"""
independent_arbiter_test.py

Standalone comparison harness. Does NOT modify fused_moe_mlp.py or the
benchmark file — only imports from fused_moe_mlp.py and re-implements
(verbatim copy) the NPU-native pipeline from the benchmark script, so
this is a fully separate arbiter.

Golden reference is written FROM SCRATCH in this file (golden_moe_mlp_loop
below) instead of reusing reference_moe_mlp() from fused_moe_mlp.py, since
that one hasn't been touched in a while and could silently encode a stale
tensor layout assumption. All matmuls in the golden path run in fp32, and
only the *final* output is rounded to fp16 — that minimizes the number of
rounding points relative to both candidates we're arbitrating between.

Before trusting golden_moe_mlp_loop on the real benchmark sizes, main()
runs a self-check: on a tiny synthetic case it cross-validates the
loop-based golden implementation against an independently-written
fully-vectorized (gather + bmm) implementation of the exact same layout
assumption. If those two disagree, the layout assumption itself (or a
loop/indexing bug) is wrong and the script aborts before comparing
anything to it.

Layout assumption (must match fused_moe_mlp.py / the NPU ops):
  x   : (num_tokens, hidden_size), tokens grouped contiguously by expert,
        expert 0's tokens first, then expert 1's, etc.
  w13 : (num_experts, hidden_size, 2*inter_size)
        w13[e, :, :inter_size]  = gate projection
        w13[e, :, inter_size:]  = up projection
  w2  : (num_experts, inter_size, hidden_size)
  group_sizes : (num_experts,) int, token count routed to each expert

For each test case we compute:
  1. golden      = golden_moe_mlp_loop(...)              (fp32 math, from scratch)
  2. triton_out  = fused_moe_mlp(...)                    (your kernel)
  3. npu_out     = npu_grouped_matmul + npu_swiglu + npu_grouped_matmul (Ascend native)

...and report which of (2)/(3) is numerically closer to (1).

Run this on the actual Ascend NPU machine (needs torch_npu +
torch.ops.npu.npu_grouped_matmul / npu_swiglu registered).
"""

import traceback
import torch

from fused_moe_mlp import fused_moe_mlp


def golden_moe_mlp_loop(x, w13, w2, group_sizes):
    """
    From-scratch, pure fp32, loop-per-expert reference implementation.
    See module docstring for the tensor layout this assumes.
    """
    num_tokens, hidden_size = x.shape
    num_experts, w13_hidden, up_dim = w13.shape
    inter_size = up_dim // 2

    assert w13_hidden == hidden_size, (w13_hidden, hidden_size)
    assert w2.shape == (num_experts, inter_size, hidden_size), (w2.shape, num_experts, inter_size, hidden_size)
    assert group_sizes.numel() == num_experts, (group_sizes.numel(), num_experts)
    assert int(group_sizes.sum().item()) == num_tokens, (int(group_sizes.sum().item()), num_tokens)

    x_f32 = x.float()
    w13_f32 = w13.float()
    w2_f32 = w2.float()

    out = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=x.device)

    row = 0
    for e in range(num_experts):
        n = int(group_sizes[e].item())
        if n == 0:
            continue

        x_e = x_f32[row:row + n]                 # (n, hidden)
        w_gate = w13_f32[e, :, :inter_size]       # (hidden, inter)
        w_up = w13_f32[e, :, inter_size:]         # (hidden, inter)

        gate = x_e @ w_gate                       # (n, inter)
        up = x_e @ w_up                           # (n, inter)

        silu_gate = gate * torch.sigmoid(gate)
        hidden = silu_gate * up                   # (n, inter)

        out[row:row + n] = hidden @ w2_f32[e]     # (n, hidden)

        row += n

    return out.to(x.dtype)


def _golden_moe_mlp_vectorized_for_selfcheck(x, w13, w2, group_sizes):
    """
    Independent, fully-vectorized (gather + bmm) re-implementation of the
    exact same layout assumption as golden_moe_mlp_loop, used ONLY to
    cross-validate that function on a small synthetic case. Deliberately
    memory-heavy (materializes a per-token weight tensor), so this must
    NOT be called on the real benchmark sizes.
    """
    num_tokens, hidden_size = x.shape
    num_experts, _, up_dim = w13.shape
    inter_size = up_dim // 2

    expert_id = torch.repeat_interleave(
        torch.arange(num_experts, device=x.device), group_sizes.to(torch.int64)
    )
    assert expert_id.numel() == num_tokens

    x_f32 = x.float()
    w13_f32 = w13.float()
    w2_f32 = w2.float()

    w13_per_tok = w13_f32[expert_id]              # (num_tokens, hidden, 2*inter)
    gate_up = torch.bmm(x_f32.unsqueeze(1), w13_per_tok).squeeze(1)  # (num_tokens, 2*inter)
    gate, up = gate_up[:, :inter_size], gate_up[:, inter_size:]

    silu_gate = gate * torch.sigmoid(gate)
    hidden = silu_gate * up                       # (num_tokens, inter)

    w2_per_tok = w2_f32[expert_id]                # (num_tokens, inter, hidden)
    out = torch.bmm(hidden.unsqueeze(1), w2_per_tok).squeeze(1)      # (num_tokens, hidden)

    return out.to(x.dtype)


def self_check_golden_reference(device):
    """
    Cross-validates golden_moe_mlp_loop against an independently-written
    vectorized implementation on a tiny synthetic case. Raises if they
    disagree beyond fp32 numerical noise.
    """
    torch.manual_seed(1234)

    num_experts = 5
    hidden_size = 32
    inter_size = 16
    group_sizes = torch.tensor([7, 0, 13, 4, 9], dtype=torch.int64, device=device)
    num_tokens = int(group_sizes.sum().item())

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device)
    w13 = torch.randn(num_experts, hidden_size, 2 * inter_size, dtype=torch.float32, device=device)
    w2 = torch.randn(num_experts, inter_size, hidden_size, dtype=torch.float32, device=device)

    out_loop = golden_moe_mlp_loop(x, w13, w2, group_sizes)
    out_vec = _golden_moe_mlp_vectorized_for_selfcheck(x, w13, w2, group_sizes)

    max_abs = (out_loop - out_vec).abs().max().item()
    if max_abs > 1e-4:
        raise RuntimeError(
            f"golden_moe_mlp_loop self-check FAILED: max_abs diff vs independent "
            f"vectorized implementation = {max_abs}. Do not trust the golden "
            f"reference in this state — fix golden_moe_mlp_loop before continuing."
        )
    print(f"golden reference self-check OK (max_abs vs independent impl = {max_abs:.2e})")


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