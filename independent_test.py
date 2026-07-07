"""
independent_arbiter_test.py

Standalone comparison harness. Does NOT modify fused_moe_mlp.py or the
benchmark file — only imports from fused_moe_mlp.py and re-implements
(verbatim copy) the NPU-native pipeline from the benchmark script, so
this is a fully separate arbiter.

Golden reference = reference_moe_mlp() already shipped inside
fused_moe_mlp.py: all matmuls done in fp32, only the *final* output is
rounded to fp16. That's the closest thing to "the true algorithm" we
have available without a fp64/exact reference, since it minimizes the
number of rounding points relative to both candidates.

For each test case we compute:
  1. golden      = reference_moe_mlp(...)                (fp32 math)
  2. triton_out  = fused_moe_mlp(...)                    (your kernel)
  3. npu_out     = npu_grouped_matmul + npu_swiglu + npu_grouped_matmul (Ascend native)

...and report which of (2)/(3) is numerically closer to (1).

Run this on the actual Ascend NPU machine (needs torch_npu +
torch.ops.npu.npu_grouped_matmul / npu_swiglu registered).
"""

import traceback
import torch

from fused_moe_mlp import fused_moe_mlp, reference_moe_mlp as golden_reference_moe_mlp


def npu_native_moe_mlp(x, w13, w2, expert_tokens):
    """
    Verbatim copy of reference_moe_mlp() from the NPU benchmark script
    (grouped_matmul -> npu_swiglu -> grouped_matmul). Not modified.
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


def compare_to_golden(name, out, golden):
    diff = (out.float() - golden.float()).abs()
    rel = diff / golden.float().abs().clamp_min(1e-6)
    return {
        "name": name,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
    }


def run_case(hidden_size, inter_size, num_experts, group_sizes_list, device, seed=0):
    torch.manual_seed(seed)

    group_sizes = torch.tensor(group_sizes_list, dtype=torch.int64, device=device)
    num_tokens = int(group_sizes.sum().item())

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    w13 = torch.randn(num_experts, hidden_size, 2 * inter_size, dtype=torch.float16, device=device)
    w2 = torch.randn(num_experts, inter_size, hidden_size, dtype=torch.float16, device=device)

    # 1. Ground truth: pure fp32 math, only final store rounds to fp16.
    golden = golden_reference_moe_mlp(x, w13, w2, group_sizes)

    # 2. Your Triton kernel.
    triton_out = fused_moe_mlp(
        x, w13, w2, group_sizes,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )

    # 3. Ascend native op pipeline.
    npu_out = npu_native_moe_mlp(x, w13, w2, group_sizes)

    if device.type == "npu":
        torch.npu.synchronize()

    stats_triton = compare_to_golden("triton (fused_moe_mlp)", triton_out, golden)
    stats_npu = compare_to_golden("npu native (grouped_matmul+swiglu)", npu_out, golden)

    print("=" * 78)
    print(f"case: hidden={hidden_size} inter={inter_size} experts={num_experts} tokens={num_tokens}")
    for s in (stats_triton, stats_npu):
        print(
            f"  {s['name']:<38} max_abs={s['max_abs']:.6f} mean_abs={s['mean_abs']:.6f} "
            f"max_rel={s['max_rel']:.6f} mean_rel={s['mean_rel']:.6f}"
        )

    winner = "triton" if stats_triton["mean_rel"] < stats_npu["mean_rel"] else "npu native"
    print(f"  -> closer to golden (by mean_rel): {winner}")

    return stats_triton, stats_npu


def main():
    has_npu = hasattr(torch, "npu") and torch.npu.is_available()
    device = torch.device("npu" if has_npu else "cpu")
    if not has_npu:
        print(
            "WARNING: torch.npu not available in this environment — run this "
            "script on the Ascend machine where torch_npu is installed and "
            "torch.ops.npu.npu_grouped_matmul / npu_swiglu are registered. "
            "The Triton kernel also targets NPU, so this will not run correctly on CPU/CUDA."
        )

    group_sizes_128 = [
        13, 14, 20, 13, 10, 6, 20, 14, 29, 5, 3, 2, 2, 11, 10, 16,
        60, 18, 9, 12, 14, 16, 15, 15, 11, 13, 20, 13, 22, 6, 6, 21,
        10, 29, 13, 23, 22, 11, 9, 26, 2, 13, 4, 27, 9, 25, 5, 6,
        41, 26, 5, 39, 1, 34, 24, 6, 8, 34, 14, 7, 42, 16, 15, 45,
        8, 23, 11, 15, 7, 15, 10, 6, 14, 4, 14, 30, 34, 4, 8, 10,
        10, 11, 18, 14, 28, 37, 11, 5, 14, 31, 8, 8, 5, 4, 5, 21,
        28, 15, 7, 23, 15, 6, 70, 23, 23, 6, 1, 22, 11, 12, 38, 12,
        8, 14, 11, 15, 18, 14, 12, 4, 3, 11, 2, 37, 14, 21, 41, 18,
    ]

    cases = [
        dict(hidden_size=128, inter_size=256, num_experts=128, group_sizes_list=group_sizes_128),
        dict(hidden_size=512, inter_size=1024, num_experts=128, group_sizes_list=group_sizes_128),
        dict(hidden_size=1024, inter_size=2048, num_experts=128, group_sizes_list=group_sizes_128),
        dict(hidden_size=4096, inter_size=2048, num_experts=128, group_sizes_list=group_sizes_128),
        dict(hidden_size=2048, inter_size=768, num_experts=128, group_sizes_list=group_sizes_128),
    ]

    totals = {"triton": 0, "npu native": 0}

    for case in cases:
        try:
            stats_triton, stats_npu = run_case(device=device, **case)
            if stats_triton["mean_rel"] < stats_npu["mean_rel"]:
                totals["triton"] += 1
            else:
                totals["npu native"] += 1
        except Exception:
            traceback.print_exc()

    print("=" * 78)
    print(
        f"SUMMARY  triton wins: {totals['triton']}   npu native wins: {totals['npu native']}"
        f"   (out of {len(cases)} cases, closeness = mean_rel to fp32 golden)"
    )


if __name__ == "__main__":
    main()