import time
import traceback

import torch

from partially_fused_moe_mlp_triton import fused_moe_mlp, grouped_gemm2, swiglu_triton


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


def benchmark(hidden_size, inter_size, checkAccuracy=True, checkPerf=True):
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
    group_sizes = torch.tensor([
        # пустые эксперты
        0, 0,

        # совсем маленькие
        1, 2, 3, 4, 5, 7, 8,

        # около половины блока
        31, 32, 33,
        63, 64, 65,

        # вокруг BLOCK_M
        127, 128, 129,

        # чуть больше
        130, 131, 150,

        # несколько тайлов
        255, 256, 257,

        # три тайла
        383, 384, 385,

        # четыре тайла
        511, 512, 513,

        # большие
        777,
        1000,

        # опять пустые
        0, 0,

        # случайные маленькие
        17, 9, 41, 6, 11,

        # снова около BLOCK_M
        126, 127, 128, 129, 130,

        # огромные
        1023, 1024, 1025,

        # хвост
        13, 27, 2, 19, 0, 8
    ], dtype=torch.int64, device=device)

    group_sizes = torch.tensor(group_sizes, dtype=torch.int64, device=device)

    num_tokens = int(group_sizes.sum().cpu())

    x = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    w13 = torch.empty(
        num_experts,
        hidden_size,
        2 * inter_size,
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    w2 = torch.empty(
        num_experts,
        inter_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    if (checkAccuracy):
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

    if (checkPerf):

        #
        # warmup
        #

        for _ in range(100):
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
        
        startTriton = time.perf_counter()
        for _ in range(1000):
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
        endTriton = time.perf_counter()

        startNPU = time.perf_counter()
        for _ in range(1000):
            reference_moe_mlp(
                x,
                w13,
                w2,
                group_sizes,
            )

        torch.npu.synchronize()
        endNPU = time.perf_counter()

        print(
            f"hidden={hidden_size:<5} "
            f"inter={inter_size:<5} "
            f"Triton kernel: {(endTriton - startTriton):.3f} ms"
            f"NPU ops: {(endNPU - startNPU):.3f} ms"
        )

def test_swiglu(inter_size):

    device = torch.device("npu")
    num_tokens = 2048

    gate = torch.empty(
        num_tokens,
        inter_size,
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    up = torch.empty(
        num_tokens,
        inter_size,
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.5)

    out = torch.empty(
        num_tokens,
        inter_size,
        dtype=torch.bfloat16,
        device=device,
    )

    swiglu_triton(
        torch.cat((gate, up), dim=-1),
        out,
        num_tokens,
        inter_size
    )

    ref = torch.ops.npu.npu_swiglu(torch.cat((gate, up), dim=-1))

    torch.testing.assert_close(
        out,
        ref,
        rtol=0.0,
        atol=1e-2,
    )

    print("Correctness OK")

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
            benchmark(hidden, inter, checkAccuracy=False, checkPerf=True)
        except Exception:
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()