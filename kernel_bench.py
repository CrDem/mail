import torch
import traceback
import time

from fused_moe_mlp import fused_moe_mlp


def benchmark(hidden_size, inter_size):
    device = torch.device("npu")

    num_experts = 128

    #group_sizes = torch.zeros(num_experts, dtype=torch.int32, device=device)
    #group_sizes[0] = 64
    group_sizes = torch.randint(0,64,(num_experts,))
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

    fused_moe_mlp(
        x,
        w13,
        w2,
        group_sizes,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )

    if hasattr(torch, "npu"):
        torch.npu.synchronize()

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

    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    end = time.perf_counter()

    print(f"OK  hidden={hidden_size:<5} inter={inter_size:<5} {(end-start)*1000:.3f} ms")


def main():

    cases = [
        (128, 256),
        (256, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (2560, 6912),
        (3072, 8192),
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