"""
Один Triton-кернел, заменяющий последовательность:

    hidden = grouped_matmul(x, w13)      # gate_up_proj, per-expert
    hidden = silu(gate) * up             # SwiGLU
    out    = grouped_matmul(hidden, w2)  # down_proj, per-expert

из sglang/.../unquant.py (forward_npu), базовый случай: classic SwiGLU,
fp16 вход/веса, fp32 аккумулятор, без bias / без gemm1_clamp_limit.

Токены предполагаются УЖЕ отсортированы по экспертам (как после
npu_moe_init_routing_v2), сам список размеров групп (group_sizes)
аналогичен expert_tokens. Веса лежат в layout [num_experts, N, K]
(как в оригинале: w13_weight.shape = (E, 2*inter, hidden),
                    w2_weight.shape  = (E, hidden, inter)),
т.е. как обычный nn.Linear.weight — поэтому в кернеле мы делаем x @ W^T.

Раскладка w13 (gate/up) предполагается КОНКАТЕНИРОВАННОЙ:
    w13[:, :inter, :]      -> gate
    w13[:, inter:2*inter,:]-> up
Если у вас interleaved-раскладка — поменяйте только индексацию offs_n
для gate_w/up_w, остальной кернел не меняется.
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# 1. Сам кернел
# ----------------------------------------------------------------------
@triton.jit
def _fused_moe_mlp_kernel(
    x_ptr, w13_ptr, w2_ptr, out_ptr,
    # per-tile расписание, посчитанное на хосте (см. ниже)
    tile_expert_ptr, tile_row_start_ptr, tile_row_count_ptr,
    # размеры
    hidden_size, inter_size,
    # strides
    stride_xm, stride_xk,
    stride_w13_e, stride_w13_n, stride_w13_k,
    stride_w2_e, stride_w2_n, stride_w2_k,
    stride_om, stride_on,
    # тайлы (constexpr -> компилируются в конкретный бинарник для каждой комбинации)
    BLOCK_M: tl.constexpr,      # строк (токенов) на программу
    BLOCK_N: tl.constexpr,      # кусок intermediate-размерности за итерацию
    BLOCK_K: tl.constexpr,      # кусок hidden-размерности (K для gemm1)
    HIDDEN_SIZE: tl.constexpr,  # = hidden_size, но нужен как constexpr для tl.arange
):
    # ---- аналог blockIdx.x в CUDA: какой тайл строк мы обрабатываем ----
    pid = tl.program_id(0)

    expert_id = tl.load(tile_expert_ptr + pid)
    row_start = tl.load(tile_row_start_ptr + pid)
    row_count = tl.load(tile_row_count_ptr + pid)

    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < row_count          # "хвостовой" неполный тайл строк
    rows = row_start + offs_m

    # Аккумулятор выходного тайла (BLOCK_M x hidden_size), живёт в регистрах
    # ВСЮ длину внешнего цикла по intermediate — копим инкрементально,
    # без материализации промежуточного hidden в HBM.
    offs_hout = tl.arange(0, HIDDEN_SIZE)
    acc_out = tl.zeros((BLOCK_M, HIDDEN_SIZE), dtype=tl.float32)

    w13_base = w13_ptr + expert_id * stride_w13_e
    w2_base = w2_ptr + expert_id * stride_w2_e

    # ---- внешний цикл: идём по intermediate-размерности кусками BLOCK_N ----
    for n0 in range(0, inter_size, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < inter_size

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ---- внутренний цикл: редукция по hidden_size (K для gemm1) ----
        for k0 in range(0, hidden_size, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < hidden_size

            x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(
                x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
            )

            gate_ptrs = (
                w13_base
                + offs_n[:, None] * stride_w13_n
                + offs_k[None, :] * stride_w13_k
            )
            gate_w = tl.load(
                gate_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0
            )

            up_ptrs = (
                w13_base
                + (offs_n[:, None] + inter_size) * stride_w13_n
                + offs_k[None, :] * stride_w13_k
            )
            up_w = tl.load(
                up_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0
            )

            # tl.dot -> тензорные ядра, аналог wmma/mma.sync, но без ручного
            # управления shared memory / synchronization — компилятор сам
            # разложит это на серию MMA-инструкций и просинхронизирует warps.
            acc_gate = tl.dot(x_tile, tl.trans(gate_w), acc_gate)
            acc_up = tl.dot(x_tile, tl.trans(up_w), acc_up)

        # ---- SwiGLU для этого куска intermediate-размерности ----
        silu_gate = acc_gate * tl.sigmoid(acc_gate)
        hidden_chunk = (silu_gate * acc_up).to(tl.float16)  # (BLOCK_M, BLOCK_N)

        # ---- сразу используем этот кусок как К-срез для gemm2 ----
        w2_ptrs = (
            w2_base
            + offs_hout[:, None] * stride_w2_n
            + offs_n[None, :] * stride_w2_k
        )
        w2_tile = tl.load(w2_ptrs, mask=n_mask[None, :], other=0.0)  # (H, BLOCK_N)

        acc_out = tl.dot(hidden_chunk, tl.trans(w2_tile), acc_out)

    out_ptrs = out_ptr + rows[:, None] * stride_om + offs_hout[None, :] * stride_on
    tl.store(out_ptrs, acc_out.to(tl.float16), mask=m_mask[:, None])


# ----------------------------------------------------------------------
# 2. Хостовая обёртка: строим расписание тайлов и запускаем кернел
# ----------------------------------------------------------------------
def fused_moe_mlp(
    x: torch.Tensor,            # (num_tokens, hidden_size), fp16, отсортирован по экспертам
    w13: torch.Tensor,          # (num_experts, 2*inter_size, hidden_size), fp16
    w2: torch.Tensor,           # (num_experts, hidden_size, inter_size), fp16
    group_sizes: torch.Tensor,  # (num_experts,) int, сколько токенов у каждого эксперта
    BLOCK_M: int = 32,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    assert x.dtype == torch.float16 and w13.dtype == torch.float16 and w2.dtype == torch.float16
    num_tokens, hidden_size = x.shape
    num_experts, up_dim, _ = w13.shape
    inter_size = up_dim // 2

    # --- планирование тайлов на хосте (аналог расчёта grid-а в CUDA) ---
    # Для каждого эксперта режем его строки на куски по BLOCK_M
    # и запоминаем (expert_id, row_start, кол-во строк в куске).
    tile_expert, tile_row_start, tile_row_count = [], [], []
    row_ptr = 0
    for e in range(num_experts):
        n = int(group_sizes[e].item())
        n_tiles = triton.cdiv(max(n, 1), BLOCK_M) if n > 0 else 0
        for t in range(n_tiles):
            r0 = row_ptr + t * BLOCK_M
            cnt = min(BLOCK_M, row_ptr + n - r0)
            tile_expert.append(e)
            tile_row_start.append(r0)
            tile_row_count.append(cnt)
        row_ptr += n

    device = x.device
    tile_expert_t = torch.tensor(tile_expert, dtype=torch.int32, device=device)
    tile_row_start_t = torch.tensor(tile_row_start, dtype=torch.int32, device=device)
    tile_row_count_t = torch.tensor(tile_row_count, dtype=torch.int32, device=device)

    out = torch.empty((num_tokens, hidden_size), dtype=torch.float16, device=device)

    grid = (tile_expert_t.numel(),)
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


# ----------------------------------------------------------------------
# 3. Референс на чистом torch + самотест
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