import time
import torch

from fused_moe_mlp import fused_moe_mlp


device = torch.device("npu")

NUM_EXPERTS = 128

# (hidden_size, inter_size)
CASES = [
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (2560, 6912),
    (3072, 8192),
]

# сколько токенов одновременно пришло в MoE
TOKEN_CASES = [
    16,
    32,
    64,
    128,
    256,
]

WARMUP = 20
ITERS = 100


def sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


def make_single_expert(num_tokens):
    group_sizes = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
    group_sizes[0] = num_tokens
    return group_sizes


def make_uniform(num_tokens):
    group_sizes = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)

    base = num_tokens // NUM_EXPERTS
    rem = num_tokens % NUM_EXPERTS

    group_sizes[:] = base
    if rem:
        group_sizes[:rem] += 1

    return group_sizes


def make_random(num_tokens):
    expert = torch.randint(
        0,
        NUM_EXPERTS,
        (num_tokens,),
        device=device,
    )

    return torch.bincount(
        expert,
        minlength=NUM_EXPERTS,
    ).to(torch.int32)


def benchmark_case(
    hidden_size,
    inter_size,
    num_tokens,
    routing_name,
    group_sizes,
):
    x = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )

    w13 = torch.randn(
        NUM_EXPERTS,
        2 * inter_size,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )

    w2 = torch.randn(
        NUM_EXPERTS,
        hidden_size,
        inter_size,
        device=device,
        dtype=torch.float16,
    )

    # warmup
    for _ in range(WARMUP):
        fused_moe_mlp(
            x,
            w13,
            w2,
            group_sizes,
        )

    sync()

    start = time.perf_counter()

    for _ in range(ITERS):
        fused_moe_mlp(
            x,
            w13,
            w2,
            group_sizes,
        )

    sync()

    elapsed = (time.perf_counter() - start) / ITERS

    tok_per_sec = num_tokens / elapsed

    print(
        f"{routing_name:12}"
        f" tokens={num_tokens:4d}"
        f" time={elapsed*1000:7.3f} ms"
        f" throughput={tok_per_sec:10.0f} tok/s"
    )


def benchmark_hidden(hidden_size, inter_size):

    print("=" * 90)
    print(
        f"hidden={hidden_size}  "
        f"inter={inter_size}"
    )

    for num_tokens in TOKEN_CASES:

        benchmark_case(
            hidden_size,
            inter_size,
            num_tokens,
            "single",
            make_single_expert(num_tokens),
        )

        benchmark_case(
            hidden_size,
            inter_size,
            num_tokens,
            "uniform",
            make_uniform(num_tokens),
        )

        benchmark_case(
            hidden_size,
            inter_size,
            num_tokens,
            "random",
            make_random(num_tokens),
        )

        print()


def main():

    torch.manual_seed(0)

    for hidden, inter in CASES:
        benchmark_hidden(hidden, inter)


if __name__ == "__main__":
    main()