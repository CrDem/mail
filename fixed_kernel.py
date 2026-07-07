import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Раскладка весов в sglang (важно!):
#   w13.shape = (num_experts, hidden_size, 2*inter_size)   -> (E, K, N)
#   w2.shape  = (num_experts, inter_size,  hidden_size)    -> (E, K, N)
# То есть веса уже лежат как (K, N) — ровно то, что нужно для tl.dot(x, w)
# БЕЗ транспонирования (в отличие от раскладки nn.Linear.weight = (N, K),
# под которую был написан предыдущий вариант кернела с tl.trans).
# ----------------------------------------------------------------------
@triton.jit
def _fused_moe_mlp_kernel(
    x_ptr, w13_ptr, w2_ptr, out_ptr,
    # per-tile schedule
    tile_expert_ptr, tile_row_start_ptr, tile_row_count_ptr,
    # sizes
    hidden_size, inter_size,
    # strides x: (num_tokens, hidden_size)
    stride_xm, stride_xk,
    # strides w13: (num_experts, hidden_size, 2*inter_size) = (E, K, N)
    stride_w13_e, stride_w13_k, stride_w13_n,
    # strides w2: (num_experts, inter_size, hidden_size) = (E, K, N)
    stride_w2_e, stride_w2_k, stride_w2_n,
    # strides out: (num_tokens, hidden_size)
    stride_om, stride_on,

    BLOCK_M: tl.constexpr,      # rows per program
    BLOCK_N: tl.constexpr,      # intermediate-dim tile
    BLOCK_K: tl.constexpr,      # hidden-dim tile (K for gemm1, и он же тайл
                                 # выходной размерности для gemm2 - reused)
    HIDDEN_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    expert_id = tl.load(tile_expert_ptr + pid)
    row_start = tl.load(tile_row_start_ptr + pid)
    row_count = tl.load(tile_row_count_ptr + pid)

    if row_count == 0:  # extra tiles
        return

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < row_count
    rows = row_start + offs_m

    # tile_expert_t/tile_row_start_t уже int64 (см. build_tile_schedule) ->
    # expert_id и rows автоматически int64, адресная арифметика ниже не
    # переполнится на больших моделях/батчах.
    w13_base = w13_ptr + expert_id * stride_w13_e
    w2_base = w2_ptr + expert_id * stride_w2_e

    for h0 in range(0, hidden_size, BLOCK_K):
        offs_h = h0 + tl.arange(0, BLOCK_K)
        h_mask = offs_h < hidden_size

        acc_out = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        # intermediate-dim tiling, tile size = BLOCK_N
        for n0 in range(0, inter_size, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < inter_size

            acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # hidden_size-dim tiling, tile size = BLOCK_K (reduction for gemm1)
            for k0 in range(0, hidden_size, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < hidden_size

                x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
                x_tile = tl.load(
                    x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                )  # (BLOCK_M, BLOCK_K)

                # w13 тайл гейта: (K=hidden, N=inter) -> (BLOCK_K, BLOCK_N), без trans
                gate_ptrs = (
                    w13_base
                    + offs_k[:, None] * stride_w13_k
                    + offs_n[None, :] * stride_w13_n
                )
                gate_w = tl.load(
                    gate_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                )  # (BLOCK_K, BLOCK_N)

                up_ptrs = (
                    w13_base
                    + offs_k[:, None] * stride_w13_k
                    + (offs_n[None, :] + inter_size) * stride_w13_n
                )
                up_w = tl.load(
                    up_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                )  # (BLOCK_K, BLOCK_N)

                # (M,K) @ (K,N) -> (M,N), напрямую, без tl.trans
                acc_gate = tl.dot(x_tile, gate_w, acc_gate)
                acc_up = tl.dot(x_tile, up_w, acc_up)

            # SwiGLU
            silu_gate = acc_gate * tl.sigmoid(acc_gate)
            hidden_chunk = (silu_gate * acc_up)  # (BLOCK_M, BLOCK_N)

            # w2 тайл: (K=inter, N=hidden) -> (BLOCK_N, BLOCK_K), без trans
            w2_ptrs = (
                w2_base
                + offs_n[:, None] * stride_w2_k
                + offs_h[None, :] * stride_w2_n
            )
            w2_tile = tl.load(
                w2_ptrs, mask=n_mask[:, None] & h_mask[None, :], other=0.0
            )  # (BLOCK_N, BLOCK_K)

            # (M,BLOCK_N) @ (BLOCK_N,BLOCK_K) -> (M,BLOCK_K), напрямую
            acc_out = tl.dot(hidden_chunk.to(w2_tile.dtype), w2_tile, acc_out)

        out_ptrs = out_ptr + rows[:, None] * stride_om + offs_h[None, :] * stride_on
        tl.store(
            out_ptrs, acc_out.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & h_mask[None, :],
        )


def build_tile_schedule(
    group_sizes: torch.Tensor, num_tokens: int, BLOCK_M: int
):
    device = group_sizes.device
    num_experts = group_sizes.numel()

    # offsets[e] - every expert starting point
    offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(group_sizes.to(torch.int64), dim=0)

    tiles_per_expert = (group_sizes.to(torch.int64) + BLOCK_M - 1) // BLOCK_M  # ceil div
    tiles_cumsum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    tiles_cumsum[1:] = torch.cumsum(tiles_per_expert, dim=0)

    grid_size = int(tiles_per_expert.sum().item())

    tile_idx = torch.arange(grid_size, device=device, dtype=torch.int64)
    tile_expert = torch.searchsorted(tiles_cumsum[1:], tile_idx, right=True)
    valid = tile_expert < num_experts
    tile_expert = tile_expert.clamp(max=num_experts - 1)

    local_tile = tile_idx - tiles_cumsum[tile_expert]  # tile num for current expert
    tile_row_start = offsets[tile_expert] + local_tile * BLOCK_M

    tile_row_count = torch.clamp(
        offsets[tile_expert + 1] - tile_row_start, min=0, max=BLOCK_M
    )
    tile_row_count = torch.where(valid, tile_row_count, torch.zeros_like(tile_row_count))

    # int64 намеренно: expert_id/rows внутри кернела участвуют в адресной
    # арифметике (expert_id * stride_w13_e и т.п.), а per-expert stride
    # для реальных размеров модели легко переполняет int32.
    return (
        tile_expert.to(torch.int64),
        tile_row_start.to(torch.int64),
        tile_row_count.to(torch.int64),
        grid_size,
    )


def fused_moe_mlp(
    x: torch.Tensor,            # (num_tokens, hidden_size)
    w13: torch.Tensor,          # (num_experts, hidden_size, 2*inter_size)  -- (E, K, N)
    w2: torch.Tensor,           # (num_experts, inter_size, hidden_size)   -- (E, K, N)
    group_sizes: torch.Tensor,  # (num_experts,) int
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
) -> torch.Tensor:

    print(f"w13.shape: {w13.shape}, w13.stride: {w13.stride()}")
    print(f"w2.shape: {w2.shape}, w2.stride: {w2.stride()}")

    print(f"group_sizes.sum(): {group_sizes.sum()}")
    print(f"x.shape[0]: {x.shape[0]}")
    print(f"group_sizes.max(): {group_sizes.max()}")
    print(f"group_sizes.min(): {group_sizes.min()}")
    print(f"group_sizes.nonzero().shape: {group_sizes.nonzero().shape}")

    num_tokens, hidden_size = x.shape
    num_experts, w13_k, up_dim = w13.shape          # (E, hidden_size, 2*inter_size)
    inter_size = up_dim // 2
    assert w13_k == hidden_size, (
        f"w13 второе измерение ({w13_k}) должно совпадать с hidden_size ({hidden_size})"
    )

    group_sizes = group_sizes.to(device=x.device)

    tile_expert_t, tile_row_start_t, tile_row_count_t, grid_size = build_tile_schedule(
        group_sizes, num_tokens, BLOCK_M
    )

    device = x.device
    out = torch.empty((num_tokens, hidden_size), dtype=torch.float16, device=device)

    assert int(group_sizes.sum()) == x.shape[0]

    assert (tile_row_count_t >= 0).all()
    assert (tile_row_count_t <= BLOCK_M).all()

    assert (tile_row_start_t >= 0).all()

    assert (
        tile_row_start_t + tile_row_count_t
        <= num_tokens
    ).all()

    grid = (grid_size,)
    _fused_moe_mlp_kernel[grid](
        x, w13, w2, out,
        tile_expert_t, tile_row_start_t, tile_row_count_t,
        hidden_size, inter_size,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HIDDEN_SIZE=hidden_size,
    )
    return out


# НИЖЕ ВАЙБКОД ДЛЯ ТЕСТА
# ----------------------------------------------------------------------
# Референс на чистом torch + самотест
# ----------------------------------------------------------------------
def reference_moe_mlp(x, w13, w2, group_sizes):
    """То же самое: grouped_matmul -> swiglu -> grouped_matmul, но по-честному
    в цикле по экспертам (для проверки корректности). w13/w2 уже в раскладке
    (K, N), поэтому без .T."""
    inter_size = w13.shape[2] // 2
    out = torch.empty(x.shape[0], w2.shape[2], dtype=x.dtype, device=x.device)
    row = 0
    for e in range(w13.shape[0]):
        n = int(group_sizes[e].item())
        if n == 0:
            continue
        xe = x[row:row + n]
        gate_up = xe.float() @ w13[e].float()             # (n, hidden) @ (hidden, 2*inter)
        gate, up = gate_up[:, :inter_size], gate_up[:, inter_size:]
        hidden = torch.nn.functional.silu(gate) * up
        out[row:row + n] = (hidden @ w2[e].float()).to(x.dtype)  # (n, inter) @ (inter, hidden)
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
    # НОВАЯ раскладка: (E, K, N)
    w13 = torch.randn(num_experts, hidden_size, 2 * inter_size, dtype=torch.float16, device=device) * 0.05
    w2 = torch.randn(num_experts, inter_size, hidden_size, dtype=torch.float16, device=device) * 0.05

    out_triton = fused_moe_mlp(x, w13, w2, group_sizes)
    out_ref = reference_moe_mlp(x, w13, w2, group_sizes)

    diff = (out_triton.float() - out_ref.float()).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())
    torch.testing.assert_close(out_triton, out_ref.to(torch.float16), atol=2e-2, rtol=2e-2)
    print("OK: triton kernel matches reference")