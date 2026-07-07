import time
import traceback

import torch

from fused_moe_mlp import fused_moe_mlp


def reference_moe_mlp(x, w13, w2, expert_tokens):
    """
    Полная копия текущей реализации из SGLang:
        grouped_matmul -> npu_swiglu -> grouped_matmul
    Без каких-либо изменений.
    """

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[x],
        weight=[w13],
        bias=None,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=x.dtype,
    )[0]

    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        bias=None,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=x.dtype,
    )[0]

    return hidden_states


def benchmark(hidden_size, inter_size):
    device = torch.device("npu")

    num_experts = 128

    group_sizes = [
        13, 14, 20, 13, 10, 6, 20, 14, 29, 5, 3, 2, 2, 11, 10, 16,
        60, 18, 9, 12, 14, 16, 15, 15, 11, 13, 20, 13, 22, 6, 6, 21,
        10, 29, 13, 23, 22, 11, 9, 26, 2, 13, 4, 27, 9, 25, 5, 6,
        41, 26, 5, 39, 1, 34, 24, 6, 8, 34, 14, 7, 42, 16, 15, 45,
        8, 23, 11, 15, 7, 15, 10, 6, 14, 4, 14, 30, 34, 4, 8, 10,
        10, 11, 18, 14, 28, 37, 11, 5, 14, 31, 8, 8, 5, 4, 5, 21,
        28, 15, 7, 23, 15, 6, 70, 23, 23, 6, 1, 22, 11, 12, 38, 12,
        8, 14, 11, 15, 18, 14, 12, 4, 3, 11, 2, 37, 14, 21, 41, 18,
    ]

    group_sizes = torch.tensor(group_sizes, dtype=torch.int64, device=device)

    num_tokens = int(group_sizes.sum().cpu())

    x = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.float16,
        device=device,
    )

    w13 = torch.randn(
        num_experts,
        hidden_size,
        2 * inter_size,
        dtype=torch.float16,
        device=device,
    )

    w2 = torch.randn(
        num_experts,
        inter_size,
        hidden_size,
        dtype=torch.float16,
        device=device,
    )

    #
    # correctness
    #

    ref = reference_moe_mlp(
        x,
        w13,
        w2,
        group_sizes,
    )

    out = fused_moe_mlp(
        x,
        w13,
        w2,
        group_sizes,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )

    torch.npu.synchronize()

    torch.testing.assert_close(
        out,
        ref,
        rtol=1e-2,
        atol=1e-2,
    )

    print("Correctness OK")

    #
    # warmup
    #

    fused_moe_mlp(
        x,
        w13,
        w2,
        group_sizes,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )

    torch.npu.synchronize()

    #
    # benchmark
    #

    start = time.perf_counter()

    fused_moe_mlp(
        x,
        w13,
        w2,
        group_sizes,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )

    torch.npu.synchronize()

    end = time.perf_counter()

    print(
        f"OK  hidden={hidden_size:<5} "
        f"inter={inter_size:<5} "
        f"{(end - start) * 1000:.3f} ms"
    )


def main():
    cases = [
        (128, 256),
        (512, 1024),
        (1024, 2048),
        (4096, 2048),
        (2048, 768),
    ]

    for hidden, inter in cases:
        print("=" * 70)
        print(f"Testing hidden={hidden}, inter={inter}")

        try:
            benchmark(hidden, inter)
        except Exception:
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()

hidden = torch.ops.npu.npu_grouped_matmul(
    x=[x],
    weight=[w13],
    bias=None,
    split_item=2,
    group_list_type=1,
    group_type=0,
    group_list=group_sizes,
    output_dtype=x.dtype,
)[0]

hidden = torch.ops.npu.npu_swiglu(hidden)

ref = torch.ops.npu.npu_grouped_matmul(
    x=[hidden],
    weight=[w2],
    bias=None,
    split_item=2,
    group_list_type=1,
    group_type=0,
    group_list=group_sizes,
    output_dtype=x.dtype,
)[0]

out = grouped_gemm2(
    hidden,
    w2,
    group_sizes,
)

torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)

def grouped_gemm2(
    hidden: torch.Tensor,       # (num_tokens, inter_size)
    w2: torch.Tensor,           # (num_experts, inter_size, hidden_size)
    group_sizes: torch.Tensor,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 32,
) -> torch.Tensor:

    num_tokens, inter_size = hidden.shape
    num_experts, _, hidden_size = w2.shape

    group_sizes = group_sizes.to(device=hidden.device)

    (
        tile_expert_t,
        tile_row_start_t,
        tile_row_count_t,
        grid_m,
    ) = build_tile_schedule(
        group_sizes,
        num_tokens,
        BLOCK_M,
    )

    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden.dtype,
        device=hidden.device,
    )

    grid = (grid_m, triton.cdiv(hidden_size, BLOCK_N))

    _grouped_gemm2_kernel[grid](
        hidden,
        w2,
        out,
        tile_expert_t,
        tile_row_start_t,
        tile_row_count_t,
        hidden_size,
        inter_size,
        hidden.stride(0),
        hidden.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out