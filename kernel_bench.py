import time
import traceback
import argparse

import torch

#from megaMOE_triton import megaMOE_kernel
from sgl_kernel_npu.moe.mega_moe import megaMOE_kernel


AUTOTUNE_CONFIGS = [
    # M=32
    (32, 128, 256, 128),
    (32, 128, 256, 256),
    # M=128
    (128, 64, 64, 64),
    (128, 64, 64, 128),
    (128, 64, 64, 256),
    (128, 64, 128, 64),
    (128, 64, 128, 128),
    (128, 64, 128, 256),
    (128, 64, 256, 64),
    (128, 64, 256, 128),
    (128, 64, 256, 256),
    (128, 128, 64, 64),
    (128, 128, 64, 128),
    (128, 128, 64, 256),
    (128, 128, 128, 64),
    (128, 128, 128, 128),
    (128, 128, 128, 256),
    (128, 128, 256, 64),
    (128, 128, 256, 128),
    (128, 128, 256, 256),
    (128, 256, 64, 64),
    (128, 256, 64, 128),
    (128, 256, 64, 256),
    (128, 256, 128, 64),
    (128, 256, 128, 128),
    (128, 256, 128, 256),
    (128, 256, 256, 64),
    (128, 256, 256, 128),
    (128, 256, 256, 256),
]


def reference_moe_mlp(x, w13, w2, expert_tokens):

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


def make_moe_tensors(num_tokens, hidden_size, inter_size, num_experts, device, dtype):
    x = torch.empty(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    w13 = torch.empty(
        num_experts,
        hidden_size,
        2 * inter_size,
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    w2 = torch.empty(
        num_experts,
        inter_size,
        hidden_size,
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    return x, w13, w2


def check_correctness(x, w13, w2, group_sizes, repeats=5):
    ref = None
    out = None

    for _ in range(repeats):
        ref = reference_moe_mlp(
            x,
            w13,
            w2,
            group_sizes,
        )

        torch.npu.synchronize()
        out = megaMOE_kernel(
            x,
            w13,
            w2,
            group_sizes,
        )
        torch.npu.synchronize()

    torch.npu.synchronize()

    diff_golden = (ref - out).abs()
    print(f"diff_golden (Max Diff): {diff_golden.max().item()}")
    torch.testing.assert_close(
        out,
        ref,
        rtol=0.0,
        atol=1e-2,
    )

    print("Correctness OK")


def run_moe_benchmark(
    x,
    w13,
    w2,
    group_sizes,
    warmup_iters,
    bench_iters,
    BLOCK_M=None,
    BLOCK_N=None,
    BLOCK_N2=None,
    BLOCK_K=None,
):
    """
    Прогоняет вармап + замер megaMOE_kernel.
    Если размеры блоков не заданы (None) - кернел вызывается без них.
    Возвращает суммарное время (в секундах) на bench_iters итераций.
    """

    kwargs = {}
    if None not in (BLOCK_M, BLOCK_N, BLOCK_N2, BLOCK_K):
        kwargs = dict(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_N2=BLOCK_N2,
            BLOCK_K=BLOCK_K,
        )

    #
    # warmup
    #

    for _ in range(warmup_iters):
        megaMOE_kernel(x, w13, w2, group_sizes, **kwargs)
        torch.npu.synchronize()

    torch.npu.synchronize()

    #
    # benchmark
    #

    t0 = time.perf_counter()

    for _ in range(bench_iters):
        megaMOE_kernel(x, w13, w2, group_sizes, **kwargs)
        torch.npu.synchronize()

    torch.npu.synchronize()
    t1 = time.perf_counter()

    return t1 - t0


def autotune_fused_moe(
    x,
    w13,
    w2,
    group_sizes,
    configs=AUTOTUNE_CONFIGS,
    warmup=20,
    iters=100,
):

    best_cfg = None
    best_time = float("inf")

    print("\nAutotuning...\n")

    for BLOCK_M, BLOCK_N, BLOCK_N2, BLOCK_K in configs:

        try:
            elapsed_total = run_moe_benchmark(
                x,
                w13,
                w2,
                group_sizes,
                warmup_iters=warmup,
                bench_iters=iters,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_N2=BLOCK_N2,
                BLOCK_K=BLOCK_K,
            )
            elapsed = elapsed_total / iters

            print(
                f"M={BLOCK_M:3d} "
                f"N={BLOCK_N:3d} "
                f"N2={BLOCK_N2:3d} "
                f"K={BLOCK_K:3d} "
                f"{elapsed*1e6:9.1f} us"
            )

            if elapsed < best_time:
                best_time = elapsed
                best_cfg = (
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_N2,
                    BLOCK_K,
                )

        except Exception as e:
            print(
                f"M={BLOCK_M} "
                f"N={BLOCK_N} "
                f"N2={BLOCK_N2} "
                f"K={BLOCK_K} "
                f"FAILED ({e})"
            )

    print("\nBest config:", best_cfg)
    print(f"Average: {best_time*1e6:.1f} us\n")

    return best_cfg


def benchmark(hidden_size, inter_size, checkAccuracy=True, checkPerf=True, autotune=False):
    print("=" * 70)
    print(f"Testing Kernel")
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
        0, 31, 32, 33, 63, 64, 65, 0, 31, 32, 33, 63, 64, 65, 1, 0,
    ]

    group_sizes = torch.tensor(group_sizes, dtype=torch.int64, device=device)

    num_tokens = int(group_sizes.sum().cpu())

    x, w13, w2 = make_moe_tensors(
        num_tokens,
        hidden_size,
        inter_size,
        num_experts,
        device,
        torch.bfloat16,
    )

    if checkAccuracy:
        check_correctness(x, w13, w2, group_sizes)

    if checkPerf:

        if autotune:
            BLOCK_M, BLOCK_N, BLOCK_N2, BLOCK_K = autotune_fused_moe(x, w13, w2, group_sizes)
        else:
            BLOCK_M = BLOCK_N = BLOCK_N2 = BLOCK_K = None

        elapsed_triton = run_moe_benchmark(
            x,
            w13,
            w2,
            group_sizes,
            warmup_iters=100,
            bench_iters=1000,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_N2=BLOCK_N2,
            BLOCK_K=BLOCK_K,
        )

        startNPU = time.perf_counter()
        for _ in range(1000):
            reference_moe_mlp(
                x,
                w13,
                w2,
                group_sizes,
            )
            torch.npu.synchronize()

        torch.npu.synchronize()
        endNPU = time.perf_counter()

        print(
            f"hidden={hidden_size:<5} "
            f"inter={inter_size:<5} \n"
            f"Triton kernel: {elapsed_triton * 1000:.3f} ms\n"
            f"NPU ops: {(endNPU - startNPU) * 1000:.3f} ms"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-accuracy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Проверять корректность (сравнение с reference_moe_mlp)",
    )
    parser.add_argument(
        "--check-perf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Замерять производительность",
    )
    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Автотюнить размеры блоков перед замером производительности",
    )
    args = parser.parse_args()

    cases = [
        #(128, 256),
        #(512, 1024),
        #(2560, 1280),
        #(4096, 2048),
        (2048, 768), # qwen3-30b: hidden = 2048, inter = 768
    ]

    for hidden, inter in cases:
        print("=" * 70)
        print(f"Testing hidden={hidden}, inter={inter}")

        try:
            benchmark(
                hidden,
                inter,
                checkAccuracy=args.check_accuracy,
                checkPerf=args.check_perf,
                autotune=args.autotune,
            )
        except Exception:
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()