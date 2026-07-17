import torch
import triton
import triton.language as tl

@triton.jit
def _locate_tile(pid_m, group_sizes_ptr, num_experts, BLOCK_M: tl.constexpr):
    running_tiles = tl.cast(0, tl.int64)
    running_rows = tl.cast(0, tl.int64)

    expert_id = tl.cast(0, tl.int64)
    row_start = tl.cast(0, tl.int64)
    row_count = tl.cast(0, tl.int64)
    assigned = tl.cast(0, tl.int64)

    for e in range(0, num_experts):
        gs = tl.cast(tl.load(group_sizes_ptr + e), tl.int64)
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

    return expert_id, row_start, row_count

@triton.jit
def _megaMOE_kernel(
    x_ptr, w13_ptr, w2_ptr, out_ptr,
    group_sizes_ptr,
    hidden_size, inter_size, num_experts,
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

    expert_id, row_start, row_count = _locate_tile(pid_m, group_sizes_ptr, num_experts, BLOCK_M)
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

@triton.jit
def _megaMOE_kernel_ext(
    x_ptr, w13_ptr, w2_ptr, out_ptr,
    group_sizes_ptr,
    hidden_size, inter_size, num_experts,
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

    expert_id, row_start, row_count = _locate_tile(pid_m, group_sizes_ptr, num_experts, BLOCK_M)
    if row_count == 0:
        return

    acc_out = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)

    W2_block_ptr = tl.make_block_ptr( # assert inter // BLOCK_N2 == 0
            base = w2_ptr,
            shape=(num_experts * inter_size, hidden_size),
            strides=(stride_w2_k, stride_w2_n),

            offsets=(expert_id.to(tl.int32) * inter_size + n0, pid_n2 * BLOCK_N2),
            block_shape=(BLOCK_N, BLOCK_N2),
            order=(1, 0),
        )

    Out_block_ptr = tl.make_block_ptr(
        base = out_ptr + row_start * stride_om,
        shape=(row_count.to(tl.int32), hidden_size),
        strides=(stride_om, stride_on),

        offsets=(0, pid_n2 * BLOCK_N2),
        block_shape=(BLOCK_M, BLOCK_N2),
        order=(1, 0),
    )

    for n0 in range(0, inter_size, BLOCK_N):
        X_block_ptr = tl.make_block_ptr(
            base = x_ptr + row_start * stride_xm,
            shape=(row_count.to(tl.int32), hidden_size),
            strides=(stride_xm, stride_xk),

            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )

        W13_gate_block_ptr = tl.make_block_ptr( # assert hidden // BLOCK_K == 0
            base = w13_ptr,
            shape=(num_experts * hidden_size, inter_size*2),
            strides=(stride_w13_k, stride_w13_n),

            offsets=(expert_id.to(tl.int32) * hidden_size, n0),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        W13_up_block_ptr = tl.make_block_ptr(
            base = w13_ptr,
            shape=(num_experts * hidden_size, inter_size*2),
            strides=(stride_w13_k, stride_w13_n),

            offsets=(expert_id.to(tl.int32) * hidden_size, n0 + inter_size),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(0, hidden_size, BLOCK_K):
            x_tile = tl.load(X_block_ptr, boundary_check=(0,1), padding_option="zero")
            gate_w = tl.load(W13_gate_block_ptr, boundary_check=(0,1), padding_option="zero")
            up_w = tl.load(W13_up_block_ptr, boundary_check=(0,1), padding_option="zero")

            X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_K))
            W13_gate_block_ptr = tl.advance(W13_gate_block_ptr, (BLOCK_K, 0))
            W13_up_block_ptr = tl.advance(W13_up_block_ptr, (BLOCK_K, 0))

            acc_gate = tl.dot(x_tile, gate_w, acc_gate)
            acc_up = tl.dot(x_tile, up_w, acc_up)

        # SwiGLU
        silu_gate = acc_gate * tl.sigmoid(acc_gate)
        hidden_tile = (silu_gate * acc_up).to(w2_ptr.dtype.element_ty)

        w2_tile = tl.load(W2_block_ptr, boundary_check=(0,1), padding_option="zero")
        W2_block_ptr = tl.advance(W2_block_ptr, (BLOCK_N, 0))

        acc_out = tl.dot(hidden_tile, w2_tile, acc_out)

    tl.store(Out_block_ptr, acc_out, boundary_check=(0,1))

@triton.jit
def _megaMOE_kernel_1d(
    x_ptr, w13_ptr, w2_ptr, out_ptr,
    group_sizes_ptr,
    num_tokens, hidden_size, inter_size, num_experts,
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

    expert_id, row_start, row_count = _locate_tile(pid_m, group_sizes_ptr, num_experts, BLOCK_M)
    if row_count == 0:
        return

    for n0 in range(0, inter_size, BLOCK_N):

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        X_block_ptr = tl.make_block_ptr(
            base = x_ptr + row_start * stride_xm,
            shape=(row_count.to(tl.int32), hidden_size),
            strides=(stride_xm, stride_xk),

            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )

        W13_gate_block_ptr = tl.make_block_ptr( # assert hidden // BLOCK_K == 0
            base = w13_ptr,
            shape=(num_experts * hidden_size, inter_size*2),
            strides=(stride_w13_k, stride_w13_n),

            offsets=(expert_id.to(tl.int32) * hidden_size, n0),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        W13_up_block_ptr = tl.make_block_ptr(
            base = w13_ptr,
            shape=(num_experts * hidden_size, inter_size*2),
            strides=(stride_w13_k, stride_w13_n),

            offsets=(expert_id.to(tl.int32) * hidden_size, n0 + inter_size),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        W2_block_ptr = tl.make_block_ptr( # assert inter // BLOCK_N2 == 0
            base = w2_ptr,
            shape=(num_experts * inter_size, hidden_size),
            strides=(stride_w2_k, stride_w2_n),

            offsets=(expert_id.to(tl.int32) * inter_size + n0, 0),
            block_shape=(BLOCK_N, BLOCK_N2),
            order=(1, 0),
        )

        Out_block_ptr = tl.make_block_ptr(
            base = out_ptr + row_start * stride_om,
            shape=(row_count.to(tl.int32), hidden_size),
            strides=(stride_om, stride_on),

            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_N2),
            order=(1, 0),
        )

        for k0 in range(0, hidden_size, BLOCK_K):
            x_tile = tl.load(X_block_ptr, boundary_check=(0,1), padding_option="zero")
            gate_w = tl.load(W13_gate_block_ptr, boundary_check=(0,1), padding_option="zero")
            up_w = tl.load(W13_up_block_ptr, boundary_check=(0,1), padding_option="zero")

            X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_K))
            W13_gate_block_ptr = tl.advance(W13_gate_block_ptr, (BLOCK_K, 0))
            W13_up_block_ptr = tl.advance(W13_up_block_ptr, (BLOCK_K, 0))

            acc_gate = tl.dot(x_tile, gate_w, acc_gate)
            acc_up = tl.dot(x_tile, up_w, acc_up)

        # SwiGLU
        silu_gate = acc_gate * tl.sigmoid(acc_gate)
        hidden_tile = (silu_gate * acc_up).to(w2_ptr.dtype.element_ty)

        for n2 in range(0, hidden_size, BLOCK_N2):
            acc_out = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)

            w2_tile = tl.load(W2_block_ptr, boundary_check=(0,1), padding_option="zero")
            W2_block_ptr = tl.advance(W2_block_ptr, (0, BLOCK_N2))

            acc_out = tl.dot(hidden_tile, w2_tile, acc_out)

            if (n0 == 0):
                tl.store(Out_block_ptr, acc_out, boundary_check=(0,1))
            else:
                prev_values = tl.load(Out_block_ptr, boundary_check=(0,1), padding_option="zero")
                tl.store(Out_block_ptr, prev_values + acc_out, boundary_check=(0,1))
            Out_block_ptr = tl.advance(Out_block_ptr, (0, BLOCK_N2))


def megaMOE_kernel(
    x: torch.Tensor,            # (num_tokens, hidden_size)
    w13: torch.Tensor,          # (num_experts, hidden_size, 2*inter_size)
    w2: torch.Tensor,           # (num_experts, inter_size, hidden_size)
    group_sizes: torch.Tensor,  # (num_experts,) int
    BLOCK_M: int = 64,
    BLOCK_N: int = 256,
    BLOCK_N2: int = 256,
    BLOCK_K: int = 128,
) -> torch.Tensor:

    num_tokens, hidden_size = x.shape
    num_experts, _, up_dim = w13.shape
    inter_size = up_dim // 2

    device = x.device

    group_sizes = group_sizes.to(device=device)
    #tile_expert_t, tile_row_start_t, tile_row_count_t, grid_m = build_tile_schedule(
    #    group_sizes, num_tokens, BLOCK_M
    #)
    grid_m = triton.cdiv(num_tokens, BLOCK_M) + num_experts

    out = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=device) #=torch.float32

    grid = (grid_m, triton.cdiv(hidden_size, BLOCK_N2))
    _megaMOE_kernel[grid](
        x, w13, w2, out,
        group_sizes,#tile_expert_t, tile_row_start_t, tile_row_count_t,#
        hidden_size, inter_size, num_experts,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_N2=BLOCK_N2, BLOCK_K=BLOCK_K,
    )

    '''grid = (grid_m,)
    _megaMOE_kernel_1d[grid](
        x, w13, w2, out,
        tile_expert_t, tile_row_start_t, tile_row_count_t,#group_sizes,#
        num_tokens, hidden_size, inter_size, num_experts,
        x.stride(0), x.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_N2=BLOCK_N2, BLOCK_K=BLOCK_K,
    )'''

    return out#.to(torch.bfloat16)