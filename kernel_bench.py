import time

import torch
import triton

from fused_moe_mlp import fused_moe_mlp


def run_case(hidden_size):
    device = "cuda"

    inter_size = hidden_size * 2
    num_experts = 128

    group_sizes = torch.zeros(num_experts, dtype=torch.int32, device=device)
    group_sizes[0] = 64
    num_tokens = int(group_sizes.sum().cpu())

    x = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )

    w13 = torch.randn(
        num_experts,
        inter_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )

    w2 = torch.randn(
        num_experts,
        hidden_size,
        inter_size,
        device=device,
        dtype=torch.float16,
    )

    # прогрев
    fused_moe_mlp(
        x,
        w13,
        w2,
        group_sizes,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    torch.cuda.synchronize()

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

    torch.cuda.synchronize()

    end = time.perf_counter()

    print(
        f"hidden={hidden_size:<5d} "
        f"OK "
        f"time={(end-start)*1000:.3f} ms"
    )


def main():
    hidden_sizes = [
        128,
        256,
        512,
        1024,
        1536,
        2048,
        2560,
        3072,
        4096,
    ]

    print("=" * 60)

    for hidden in hidden_sizes:
        print(f"Testing hidden={hidden}")

        try:
            run_case(hidden)

        except Exception as e:
            print(f"FAILED: {type(e).__name__}")
            print(e)

        print("-" * 60)


if __name__ == "__main__":
    main()