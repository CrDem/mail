import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_mlp_kernel(
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

    offs_n2 = pid_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    n2_mask = offs_n2 < hidden_size

    w13_base = w13_ptr + expert_id * stride_w13_e
    w2_base = w2_ptr + expert_id * stride_w2_e

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

        # SwiGLU
        silu_gate = acc_gate * tl.sigmoid(acc_gate)
        hidden_tile = (silu_gate * acc_up).to(w2_ptr.dtype.element_ty)

        w2_ptrs = (
            w2_base + offs_n[:, None] * stride_w2_k + offs_n2[None, :] * stride_w2_n
        )
        w2_tile = tl.load(w2_ptrs, mask=n_mask[:, None] & n2_mask[None, :], other=0.0)

        acc_out = tl.dot(hidden_tile, w2_tile, acc_out)

    out_ptrs = out_ptr + rows[:, None] * stride_om + offs_n2[None, :] * stride_on
    tl.store(out_ptrs, acc_out.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n2_mask[None, :])


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


def fused_moe_mlp(
    x: torch.Tensor,            # (num_tokens, hidden_size)
    w13: torch.Tensor,          # (num_experts, hidden_size, 2*inter_size)
    w2: torch.Tensor,           # (num_experts, inter_size, hidden_size)
    group_sizes: torch.Tensor,  # (num_experts,) int
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_N2: int = 32,
    BLOCK_K: int = 32,
) -> torch.Tensor:

    '''print(f"w13.shape: {w13.shape}, w13.stride: {w13.stride()}")
    print(f"w2.shape: {w2.shape}, w2.stride: {w2.stride()}")

    print(f"group_sizes.sum(): {group_sizes.sum()}")
    print(f"x.shape[0]: {x.shape[0]}")
    print(f"group_sizes.max(): {group_sizes.max()}")
    print(f"group_sizes.min(): {group_sizes.min()}")
    print(f"group_sizes.nonzero().shape: {group_sizes.nonzero().shape}")'''

    num_tokens, hidden_size = x.shape
    num_experts, _, up_dim = w13.shape
    inter_size = up_dim // 2

    group_sizes = group_sizes.to(device=x.device)
    tile_expert_t, tile_row_start_t, tile_row_count_t, grid_m = build_tile_schedule(
        group_sizes, num_tokens, BLOCK_M
    )

    '''assert int(group_sizes.sum()) == x.shape[0]

    assert (tile_row_count_t >= 0).all()
    assert (tile_row_count_t <= BLOCK_M).all()

    assert (tile_row_start_t >= 0).all()

    assert (
        tile_row_start_t + tile_row_count_t
        <= num_tokens
    ).all()'''

    device = x.device
    out = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=device)

    grid = (grid_m, triton.cdiv(hidden_size, BLOCK_N2))
    _fused_moe_mlp_kernel[grid](
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


# НИЖЕ ВАЙБКОД ДЛЯ ТЕСТА
# ----------------------------------------------------------------------
# Референс на чистом torch + самотест
# ----------------------------------------------------------------------
def reference_moe_mlp(x, w13, w2, group_sizes):
    """То же самое, что делал старый код: grouped_matmul -> swiglu -> grouped_matmul,
    но по-честному в цикле по экспертам (для проверки корректности)."""
    inter_size = w13.shape[1] // 2
    out = torch.empty(x.shape[0], w2.shape[1], dtype=x.dtype, device=x.device)
    row = 0
    for e in range(w13.shape[0]):
        n = int(group_sizes[e].item())
        if n == 0:
            continue
        xe = x[row:row + n]
        gate_up = xe.float() @ w13[e].float().T          # (n, 2*inter)
        gate, up = gate_up[:, :inter_size], gate_up[:, inter_size:]
        hidden = torch.nn.functional.silu(gate) * up
        out[row:row + n] = (hidden @ w2[e].float().T).to(x.dtype)
        row += n
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    num_experts = 4
    hidden_size = 256
    inter_size = 512
    group_sizes = torch.tensor([37, 12, 50, 21])  # неровные размеры групп специально
    num_tokens = int(group_sizes.sum())

    device = "cuda"
    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device) * 0.1
    w13 = torch.randn(num_experts, 2 * inter_size, hidden_size, dtype=torch.float16, device=device) * 0.05
    w2 = torch.randn(num_experts, hidden_size, inter_size, dtype=torch.float16, device=device) * 0.05

    out_triton = fused_moe_mlp(x, w13, w2, group_sizes)
    out_ref = reference_moe_mlp(x, w13, w2, group_sizes)

    diff = (out_triton.float() - out_ref.float()).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())
    torch.testing.assert_close(out_triton, out_ref.to(torch.float16), atol=2e-2, rtol=2e-2)
    print("OK: triton kernel matches reference")