import torch
import triton
import triton.language as tl


@triton.jit
def _locate_tile(pid_m, group_sizes_ptr, num_experts, BLOCK_M: tl.constexpr):
    """
    Given a global M-tile index `pid_m`, figure out on the fly:
      - which expert this tile belongs to (expert_id)
      - the starting row offset for this tile (row_start)
      - how many valid rows it actually has, <= BLOCK_M (row_count)

    This replaces the host-side build_tile_schedule() pass (cumsum +
    searchsorted + several elementwise torch kernels, each with its own
    launch overhead). Every program instead walks the small group_sizes
    array once on-device. Same approach as the official Triton
    grouped-gemm tutorial (persistent "which group am I in" loop).

    Kept entirely in int32 inside the loop (group_sizes is forced to
    int32 by the caller) so loop-carried variables have a consistent
    dtype across iterations; only cast to int64 once at the very end,
    right before it's used for pointer/stride arithmetic.
    """
    running_tiles = 0
    running_rows = 0

    expert_id = 0
    row_start = 0
    row_count = 0
    assigned = 0

    for e in range(0, num_experts):
        gs = tl.load(group_sizes_ptr + e)
        tiles_e = (gs + BLOCK_M - 1) // BLOCK_M

        local_tile = pid_m - running_tiles
        hit = (local_tile >= 0) & (local_tile < tiles_e) & (assigned == 0)

        cand_row_start = running_rows + local_tile * BLOCK_M
        cand_row_count = gs - local_tile * BLOCK_M
        cand_row_count = tl.minimum(cand_row_count, BLOCK_M)
        cand_row_count = tl.maximum(cand_row_count, 0)

        expert_id = tl.where(hit, e, expert_id)
        row_start = tl.where(hit, cand_row_start, row_start)
        row_count = tl.where(hit, cand_row_count, row_count)
        assigned = tl.where(hit, 1, assigned)

        running_tiles += tiles_e
        running_rows += gs

    # tiles beyond the real schedule (pid_m >= total actual tiles, since the
    # grid is only an upper bound) simply never hit -> row_count stays 0,
    # caller does an early return, exactly like before.
    return expert_id.to(tl.int64), row_start.to(tl.int64), row_count.to(tl.int64)


# fused gmm1 (gate_up_proj) + SwiGLU
@triton.jit
def _grouped_gemm1_swiglu_kernel(
    x_ptr, w13_ptr, hidden_ptr,
    group_sizes_ptr, num_experts,
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

    expert_id, row_start, row_count = _locate_tile(pid_m, group_sizes_ptr, num_experts, BLOCK_M)
    if row_count == 0:
        return

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < row_count
    rows = tl.cast(row_start + offs_m, tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n2_base = pid_n * BLOCK_N + tl.arange(0, BLOCK_N*2)

    n_mask = offs_n < inter_size
    n2_mask = tl.broadcast_to( n_mask[:, None], (BLOCK_N, 2) ).reshape(BLOCK_N * 2)

    # gate0..gateN-1, up0..upN-1
    offs_n2 = (
        (offs_n2_base & (BLOCK_N - 1))
        + (offs_n2_base >> tl.log2(BLOCK_N)) * inter_size
    )

    w13_base = w13_ptr + expert_id * stride_w13_e

    acc_gate_up = tl.zeros((BLOCK_M, BLOCK_N*2), dtype=tl.float32)

    for k0 in range(0, hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_size

        x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        gate_up_ptrs = (
            w13_base + offs_k[:, None] * stride_w13_k + offs_n2[None, :] * stride_w13_n )

        gate_up_w = tl.load(gate_up_ptrs, mask=k_mask[:, None] & n2_mask[None, :], other=0.0)

        acc_gate_up = tl.dot(x_tile, gate_up_w, acc_gate_up)

    # split
    gate = acc_gate_up[:, :BLOCK_N]
    up   = acc_gate_up[:, BLOCK_N:]

    # SwiGLU
    silu_gate = gate * tl.sigmoid(gate)
    hidden_tile = (silu_gate * up).to(hidden_ptr.dtype.element_ty)

    hidden_ptrs = hidden_ptr + rows[:, None] * stride_hm + offs_n[None, :] * stride_hn
    tl.store(hidden_ptrs, hidden_tile, mask=m_mask[:, None] & n_mask[None, :])

# gmm2
@triton.jit
def _grouped_gemm2_kernel(
    hidden_ptr, w2_ptr, out_ptr,
    group_sizes_ptr, num_experts,
    hidden_size, inter_size,
    stride_hm, stride_hk,
    stride_w2_e, stride_w2_k, stride_w2_n,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert_id, row_start, row_count = _locate_tile(pid_m, group_sizes_ptr, num_experts, BLOCK_M)
    if row_count == 0:
        return

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < row_count
    rows = row_start + offs_m

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < hidden_size

    w2_base = w2_ptr + expert_id * stride_w2_e

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, inter_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < inter_size

        h_ptrs = hidden_ptr + rows[:, None] * stride_hm + offs_k[None, :] * stride_hk
        h_tile = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        w2_ptrs = (
            w2_base + offs_k[:, None] * stride_w2_k + offs_n[None, :] * stride_w2_n
        )
        w2_tile = tl.load(w2_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(h_tile.to(w2_tile.dtype), w2_tile, acc)

    out_ptrs = out_ptr + rows[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def fused_moe_mlp(
    x: torch.Tensor,            # (num_tokens, hidden_size)
    w13: torch.Tensor,          # (num_experts, hidden_size, 2*inter_size)
    w2: torch.Tensor,           # (num_experts, inter_size, hidden_size)
    group_sizes: torch.Tensor,  # (num_experts,) int
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
) -> torch.Tensor:

    num_tokens, hidden_size = x.shape
    num_experts, _, up_dim = w13.shape
    inter_size = up_dim // 2

    # Forced to int32 + contiguous: keeps the in-kernel scheduling loop
    # type-consistent (no int32/int64 mix across loop iterations) and
    # guarantees stride-1 pointer arithmetic for group_sizes_ptr + e.
    # This is a tiny (num_experts-element) op, negligible next to the
    # eliminated cumsum/searchsorted round trips.
    group_sizes = group_sizes.to(device=x.device, dtype=torch.int32).contiguous()

    # Upper bound on the number of M-tiles. Pure function of
    # (num_tokens, num_experts, BLOCK_M), all known on the host already -
    # no GPU sync needed, unlike the old build_tile_schedule().
    grid_m = triton.cdiv(num_tokens, BLOCK_M) + num_experts

    device = x.device
    hidden = torch.empty((num_tokens, inter_size), dtype=x.dtype, device=device)

    grid1 = (grid_m, triton.cdiv(inter_size, BLOCK_N))
    _grouped_gemm1_swiglu_kernel[grid1](
        x, w13, hidden,
        group_sizes, num_experts,
        hidden_size, inter_size,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        hidden.stride(0), hidden.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    out = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=device)

    grid2 = (grid_m, triton.cdiv(hidden_size, BLOCK_N))
    _grouped_gemm2_kernel[grid2](
        hidden, w2, out,
        group_sizes, num_experts,
        hidden_size, inter_size,
        hidden.stride(0), hidden.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out