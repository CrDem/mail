import torch
import triton
import triton.language as tl

def build_tile_schedule(group_sizes: torch.Tensor, num_tokens: int, BLOCK_M: int):
    device = group_sizes.device
    num_experts = group_sizes.numel()

    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(group_sizes.to(torch.int64), dim=0)

    tiles_per_expert = (group_sizes.to(torch.int64) + BLOCK_M - 1) // BLOCK_M
    tiles_cumsum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    tiles_cumsum[1:] = torch.cumsum(tiles_per_expert, dim=0)

    grid_size = triton.cdiv(num_tokens, BLOCK_M) + num_experts # upper bound

    tile_idx = torch.arange(grid_size, device=device, dtype=torch.int64)
    tile_expert = torch.searchsorted(tiles_cumsum[1:], tile_idx, right=True)
    valid = tile_expert < num_experts
    tile_expert = tile_expert.clamp(max=num_experts - 1)

    local_tile = tile_idx - tiles_cumsum[tile_expert]
    tile_row_start = torch.where(valid, offsets[tile_expert] + local_tile * BLOCK_M, torch.zeros_like(local_tile))
    tile_row_count = torch.clamp(offsets[tile_expert + 1] - tile_row_start, min=0, max=BLOCK_M)
    tile_row_count = torch.where(valid, tile_row_count, torch.zeros_like(tile_row_count))

    return (
        tile_expert.to(torch.int64),
        tile_row_start.to(torch.int64),
        tile_row_count.to(torch.int64),
        grid_size,
    )

# gmm1
@triton.jit
def _grouped_gemm1(
    x_ptr, w13_ptr, hidden_ptr,
    tile_expert_ptr, tile_row_start_ptr, tile_row_count_ptr,
    hidden_size, inter_size,
    stride_xm, stride_xk,
    stride_w13_e, stride_w13_k, stride_w13_n,
    stride_hm, stride_hn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert_id = tl.load(tile_expert_ptr + pid_m)
    row_start = tl.load(tile_row_start_ptr + pid_m)
    row_count = tl.load(tile_row_count_ptr + pid_m)
    if row_count == 0:
        return

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < row_count
    rows = tl.cast(row_start + offs_m, tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < inter_size

    w13_base = w13_ptr + expert_id * stride_w13_e

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_size

        x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        gate_ptrs = (
            w13_base + offs_k[:, None] * stride_w13_k + offs_n[None, :] * stride_w13_n
        )
        gate_w = tl.load(gate_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        up_ptrs = (
            w13_base
            + offs_k[:, None] * stride_w13_k
            + (offs_n[None, :] + inter_size) * stride_w13_n
        )
        up_w = tl.load(up_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc_gate = tl.dot(x_tile, gate_w, acc_gate)
        acc_up = tl.dot(x_tile, up_w, acc_up)

    hidden_ptrs_gate = hidden_ptr + rows[:, None] * stride_hm + offs_n[None, :] * stride_hn
    tl.store(hidden_ptrs_gate, acc_gate.to(hidden_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])
    hidden_ptrs_up = hidden_ptr + rows[:, None] * stride_hm + (offs_n[None, :] + inter_size) * stride_hn
    tl.store(hidden_ptrs_up, acc_up.to(hidden_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])

def moe_mlp_triton(
    x: torch.Tensor,            # (num_tokens, hidden_size)
    w13: torch.Tensor,          # (num_experts, hidden_size, 2*inter_size)
    w2: torch.Tensor,           # (num_experts, inter_size, hidden_size)
    group_sizes: torch.Tensor,  # (num_experts,) int
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    
    '''print(f"[FUSED_MOE_MLP] w13.shape: {w13.shape}, w13.stride: {w13.stride()}")

    print(f"[FUSED_MOE_MLP] w2.shape: {w2.shape}, w2.stride: {w2.stride()}")

    print(f"[FUSED_MOE_MLP] group_sizes.sum(): {group_sizes.sum()}")
    print(f"[FUSED_MOE_MLP] x.shape: {x.shape}")
    print(f"[FUSED_MOE_MLP] group_sizes.max(): {group_sizes.max()}")
    print(f"[FUSED_MOE_MLP] group_sizes.min(): {group_sizes.min()}")
    print(f"[FUSED_MOE_MLP] group_sizes.nonzero().shape: {group_sizes.nonzero().shape}")'''

    num_tokens, hidden_size = x.shape
    num_experts, _, up_dim = w13.shape
    inter_size = up_dim // 2


    group_sizes = group_sizes.to(device=x.device)
    tile_expert_t, tile_row_start_t, tile_row_count_t, grid_m = build_tile_schedule(
        group_sizes, num_tokens, BLOCK_M
    )

    device = x.device
    hidden = torch.empty((num_tokens, inter_size*2), dtype=x.dtype, device=device)

    '''assert int(group_sizes.sum()) == x.shape[0]
    assert (tile_row_count_t >= 0).all()
    assert (tile_row_count_t <= BLOCK_M).all()

    assert (tile_row_start_t >= 0).all()

    assert (
        tile_row_start_t + tile_row_count_t
        <= num_tokens
    ).all()'''

    grid1 = (grid_m, triton.cdiv(inter_size*2, BLOCK_N))
    _grouped_gemm1[grid1](
        x, w13, hidden,
        tile_expert_t, tile_row_start_t, tile_row_count_t,
        hidden_size, inter_size,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        hidden.stride(0), hidden.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return hidden

    out = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=device)

    grid2 = (grid_m, triton.cdiv(hidden_size, BLOCK_N))
    _grouped_gemm2_kernel[grid2](
        hidden, w2, out,
        tile_expert_t, tile_row_start_t, tile_row_count_t,
        hidden_size, inter_size,
        hidden.stride(0), hidden.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out