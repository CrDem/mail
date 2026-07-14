import torch
import triton
import triton.language as tl

@triton.jit
def _megaMOE_kernel(
    x_ptr, w13_ptr, w2_ptr, out_ptr,
    tile_expert_ptr, tile_row_start_ptr, tile_row_count_ptr,
    hidden_size, inter_size,
    stride_xm, stride_xk,
    stride_w13_e, stride_w13_k, stride_w13_n,
    stride_w2_e, stride_w2_k, stride_w2_n,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n2 = tl.program_id(1)

    expert_id = tl.load(tile_expert_ptr + pid_m)
    row_start = tl.load(tile_row_start_ptr + pid_m)
    row_count = tl.load(tile_row_count_ptr + pid_m)
    if row_count == 0:
        return

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < row_count
    rows = tl.cast(row_start + offs_m, tl.int64)

    w13_base = w13_ptr + expert_id * stride_w13_e
    w2_base = w2_ptr + expert_id * stride_w2_e

    offs_o = pid_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    o_mask = offs_o < hidden_size

    acc_out = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)

    for n0 in range(0, inter_size, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < inter_size

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, hidden_size, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < hidden_size

            x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            gate_ptrs = (
                w13_base + offs_k[:, None] * stride_w13_k + offs_n[None, :] * stride_w13_n )
            up_ptrs = (
                w13_base + offs_k[:, None] * stride_w13_k + (offs_n[None, :] + inter_size) * stride_w13_n )

            gate_w = tl.load(gate_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            up_w = tl.load(up_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc_gate = tl.dot(x_tile, gate_w, acc_gate)
            acc_up = tl.dot(x_tile, up_w, acc_up)

        # SwiGLU
        silu_gate = acc_gate * tl.sigmoid(acc_gate)
        hidden_tile = (silu_gate * acc_up).to(w2_ptr.dtype.element_ty)

        w2_ptrs = (
            w2_base + offs_n[:, None] * stride_w2_k + offs_o[None, :] * stride_w2_n )
        w2_tile = tl.load(w2_ptrs, mask=n_mask[:, None] & o_mask[None, :], other=0.0)

        acc_out = tl.dot(hidden_tile, w2_tile, acc_out)

    out_ptrs = out_ptr + rows[:, None] * stride_om + offs_o[None, :] * stride_on
    tl.store(out_ptrs, acc_out, mask=m_mask[:, None] & o_mask[None, :])

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


def megaMOE_kernel(
    x: torch.Tensor,            # (num_tokens, hidden_size)
    w13: torch.Tensor,          # (num_experts, hidden_size, 2*inter_size)
    w2: torch.Tensor,           # (num_experts, inter_size, hidden_size)
    group_sizes: torch.Tensor,  # (num_experts,) int
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_N2: int = 128,
    BLOCK_K: int = 128,
) -> torch.Tensor:

    num_tokens, hidden_size = x.shape
    num_experts, _, up_dim = w13.shape
    inter_size = up_dim // 2


    group_sizes = group_sizes.to(device=x.device)
    tile_expert_t, tile_row_start_t, tile_row_count_t, grid_m = build_tile_schedule(
        group_sizes, num_tokens, BLOCK_M
    )

    device = x.device
    out = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=device)

    grid = (grid_m, triton.cdiv(hidden_size, BLOCK_N2))
    _megaMOE_kernel[grid](
        x, w13, w2, out,
        tile_expert_t, tile_row_start_t, tile_row_count_t,
        hidden_size, inter_size,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_N2=BLOCK_N2, BLOCK_K=BLOCK_K,
    )

    return out