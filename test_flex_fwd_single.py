#!/usr/bin/env python3
"""Single-launch harness for the FlexAttention forward kernel (Ascend a5).

Goal: exactly ONE AI Core kernel on the device, so that `msprof op` and the
simulator see nothing but `flex_attention_kernel`.

Everything is built on the CPU -- q/k/v, the block mask, the load-balance plan --
and only the buffers the kernel actually reads are copied to the device, right
before the launch.  No warm-up, no timing loop, no second launch, no autotune
(the kernel is a plain @triton.jit, not @triton.autotune).

Note on profiling: the H2D copies of the inputs and the D2H copy of the result
(only with --check) are aclrtMemcpy tasks, not AI Core kernels, so they do not
add kernels to the profile.  `torch.empty_like` / `torch.empty` are allocations.

Usage
-----
    # 1. correctness -- reference runs on the CPU after the single launch
    python test_flex_fwd_single.py --check

    # 2. perf
    msprof op --application="python test_flex_fwd_single.py"

    # 3. simulator -- same command, nothing else in the trace
    python test_flex_fwd_single.py

Default problem: batch 1, 16 q-heads / 8 kv-heads, head_dim 128, bf16,
segments 1000+6000+1000 = 8000 tokens, plain causal mask.

If `mojo_opset` is not already importable, point --repo (or MOJO_OPSET_ROOT) at
the directory that contains the `mojo_opset` package.
"""

import argparse
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# repo import
# ---------------------------------------------------------------------------

_REQUIRED_SYMBOLS = (
    "flex_attention_kernel",
    "_prepare_block_mask_attrs",
    "_get_or_build_load_balance_plan",
    "_get_mask_block_sizes",
    "_persistent_launch_config",
    "create_block_mask_patched",
    "TILE_BLOCK_SIZE",
)


def _import_kernel_module(repo_root):
    """Import the arch-dispatched flex_attention module from mojo_opset."""
    if repo_root:
        repo_root = os.path.abspath(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    import importlib

    # importing the package runs _dispatch, which registers the per-SoC module
    importlib.import_module("mojo_opset.backends.ttx.kernels.npu")
    mod = importlib.import_module("mojo_opset.backends.ttx.kernels.npu.flex_attention")

    missing = [n for n in _REQUIRED_SYMBOLS if not hasattr(mod, n)]
    if missing:
        # the dispatcher does not always merge private symbols -- take them
        # from the a5 file directly
        a5 = importlib.import_module(
            "mojo_opset.backends.ttx.kernels.npu.a5.flex_attention")
        for name in missing:
            if not hasattr(a5, name):
                raise ImportError(f"cannot find {name} in the flex_attention module")
            setattr(mod, name, getattr(a5, name))
    return mod


# ---------------------------------------------------------------------------
# problem construction (CPU only) -- mirrors tests/.../build_problem
# ---------------------------------------------------------------------------

FULL_MASK_MODALITIES = ("image_gen", "image_vae")


def build_problem_cpu(batch, q_head, kv_head, head_dim, data_lens, data_types,
                      sliding_window, global_window, dtype, seed):
    """The same tensors the real harness's build_problem makes, but on the CPU."""
    torch.manual_seed(seed)
    dev = "cpu"

    sample_lens = [sum(s) for s in data_lens]
    cu_seqlens = torch.tensor([0, *torch.tensor(sample_lens).cumsum(0).tolist()],
                              dtype=torch.int32, device=dev)
    total_s = int(cu_seqlens[-1].item())

    segment_ids = torch.repeat_interleave(
        torch.arange(len(sample_lens), device=dev, dtype=torch.int32),
        torch.tensor(sample_lens, device=dev),
    ).long()
    doc_start = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff()).long()

    # modality: a positive id inside a "full attention" segment, -1 otherwise
    modality_parts = []
    next_id = 1
    for sample_types, sample_lens_i in zip(data_types, data_lens):
        for seg_type, seg_len in zip(sample_types, sample_lens_i):
            if seg_type in FULL_MASK_MODALITIES:
                modality_parts.append(torch.full((seg_len,), next_id, dtype=torch.long))
                next_id += 1
            else:
                modality_parts.append(torch.full((seg_len,), -1, dtype=torch.long))
    modality = torch.cat(modality_parts)

    q = torch.rand(batch, q_head, total_s, head_dim, device=dev, dtype=dtype)
    k = torch.rand(batch, kv_head, total_s, head_dim, device=dev, dtype=dtype)
    v = torch.rand(batch, kv_head, total_s, head_dim, device=dev, dtype=dtype)

    return {
        "q": q, "k": k, "v": v,
        "segment_ids": segment_ids, "modality": modality, "doc_start": doc_start,
        "cu_seqlens": cu_seqlens, "total_s": total_s,
        "sliding_window": sliding_window, "global_window": global_window,
        "num_q_heads": q_head, "num_kv_heads": kv_head, "head_dim": head_dim,
    }


# ---------------------------------------------------------------------------
# mask_mods -- same definitions as tests/accuracy/functions/test_flex_attention.py
# ---------------------------------------------------------------------------


def mask_mod_causal(problem):
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return mask_mod


def mask_mod_full(problem):
    document_ids = problem["segment_ids"]
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        samedoc_causal = same_doc & causal
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return samedoc_causal | same_img
    return mask_mod


def mask_mod_sparse(problem):
    segment_ids = problem["segment_ids"]
    modality = problem["modality"]
    doc_start = problem["doc_start"]
    window_size = problem["sliding_window"]
    global_size = problem["global_window"]

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = segment_ids[q_idx] == segment_ids[kv_idx]
        causal = q_idx >= kv_idx
        window = causal & ((q_idx - kv_idx) <= window_size)
        glob = (causal & (kv_idx >= doc_start[q_idx])
                & (kv_idx < doc_start[q_idx] + global_size))
        sparse = same_doc & (window | glob)
        is_img = modality[q_idx] > 0
        same_img = is_img & (modality[q_idx] == modality[kv_idx])
        return sparse | same_img
    return mask_mod


def mask_mod_cross(problem):
    modality = problem["modality"]

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        is_video = modality[q_idx] > 0
        same_video = is_video & (modality[q_idx] == modality[kv_idx])
        return causal | same_video
    return mask_mod


MASK_MODS = {
    "causal": mask_mod_causal,
    "full": mask_mod_full,
    "sparse": mask_mod_sparse,
    "cross": mask_mod_cross,
}


def as_2d_mask_fn(mask_mod):
    """Adapt a (b, h, q_idx, kv_idx) mask_mod to a (q_idx, kv_idx) callable."""
    def fn(q_idx, kv_idx):
        return mask_mod(0, 0, q_idx, kv_idx)
    return fn


# ---------------------------------------------------------------------------
# CPU reference -- runs AFTER the launch and never touches the device
# ---------------------------------------------------------------------------


def reference_rows(q_h, k_h, v_h, mask_fn, sm_scale, lo, hi, chunk):
    """Dense fp32 attention for one head over q rows [lo, hi).

    Mirrors the kernel's definition exactly at the mathematical level:
      s    = (q @ k^T) * sm_scale, masked to -inf
      lse  = logsumexp(s)
      out  = softmax(s) @ v,   all-masked rows -> out 0, lse -inf
    """
    kv_len = k_h.shape[0]
    out = torch.empty(hi - lo, v_h.shape[1], dtype=torch.float32)
    lse = torch.empty(hi - lo, dtype=torch.float32)

    k_f = k_h.float()
    v_f = v_h.float()
    kv_idx = torch.arange(kv_len, dtype=torch.int64)[None, :]

    for a in range(lo, hi, chunk):
        b = min(a + chunk, hi)
        q_idx = torch.arange(a, b, dtype=torch.int64)[:, None]
        s = q_h[a:b].float() @ k_f.t()
        s *= sm_scale
        keep = mask_fn(q_idx, kv_idx)
        s = s.masked_fill(~keep, float("-inf"))
        lse[a - lo:b - lo] = torch.logsumexp(s, dim=-1)
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        out[a - lo:b - lo] = p @ v_f
    return out, lse


def pick_row_windows(num_q_blocks, n_blocks, block_m, q_len):
    """Evenly spaced q-block windows, so the check covers cheap and dense rows."""
    if n_blocks <= 0 or n_blocks >= num_q_blocks:
        picks = list(range(num_q_blocks))
    else:
        step = (num_q_blocks - 1) / max(1, n_blocks - 1) if n_blocks > 1 else 1
        picks = sorted({int(round(i * step)) for i in range(n_blocks)})
    windows = []
    for qb in picks:
        lo = qb * block_m
        hi = min(lo + block_m, q_len)
        if lo < hi:
            windows.append((qb, lo, hi))
    return windows


def run_check(prob, out_dev, lse_dev, mask_fn, sm_scale, gqa, num_q_blocks,
              block_m, n_heads, n_blocks, chunk, rtol, atol):
    q, k, v = prob["q"], prob["k"], prob["v"]
    batch, q_heads, q_len, _ = q.shape

    out_cpu = out_dev.cpu().float()
    lse_cpu = lse_dev.cpu().float()

    if n_heads <= 0 or n_heads >= q_heads:
        head_list = list(range(q_heads))
    else:
        step = max(1, q_heads // n_heads)
        head_list = list(range(0, q_heads, step))[:n_heads]

    windows = pick_row_windows(num_q_blocks, n_blocks, block_m, q_len)
    total_rows = sum(hi - lo for _, lo, hi in windows)
    print(f"[check] heads={head_list}")
    print(f"[check] q blocks={[w[0] for w in windows]}  ({total_rows} rows each head)")

    worst_out_abs = worst_out_rel = worst_lse_abs = 0.0
    failures = 0
    for z in range(batch):
        for hq in head_list:
            hkv = hq // gqa
            t0 = time.time()
            max_abs = max_rel = max_lse = 0.0
            ok = True
            for _, lo, hi in windows:
                ref_o, ref_l = reference_rows(q[z, hq], k[z, hkv], v[z, hkv],
                                              mask_fn, sm_scale, lo, hi, chunk)
                got_o = out_cpu[z, hq, lo:hi]
                got_l = lse_cpu[z, hq, lo:hi]

                d_abs = (got_o - ref_o).abs()
                d_rel = d_abs / ref_o.abs().clamp_min(1e-6)
                d_lse = torch.nan_to_num((got_l - ref_l).abs(),
                                         nan=0.0, posinf=0.0, neginf=0.0)

                max_abs = max(max_abs, float(d_abs.max()))
                max_rel = max(max_rel, float(d_rel.max()))
                max_lse = max(max_lse, float(d_lse.max()))
                ok = ok and torch.allclose(got_o, ref_o, rtol=rtol, atol=atol)
                ok = ok and torch.allclose(got_l, ref_l, rtol=rtol, atol=atol,
                                           equal_nan=True)

            worst_out_abs = max(worst_out_abs, max_abs)
            worst_out_rel = max(worst_out_rel, max_rel)
            worst_lse_abs = max(worst_lse_abs, max_lse)
            if not ok:
                failures += 1
            verdict = "OK" if ok else "FAIL"
            print(f"  z={z} hq={hq:2d} (hkv={hkv})  out max|d|={max_abs:.4e} "
                  f"max rel={max_rel:.4e}  lse max|d|={max_lse:.4e}  "
                  f"{verdict}   ({time.time() - t0:.1f}s)")

    print(f"[check] worst: out abs={worst_out_abs:.4e} rel={worst_out_rel:.4e} "
          f"lse abs={worst_lse_abs:.4e}   (rtol={rtol} atol={atol})")
    if failures:
        print(f"[check] FAILED on {failures} head(s)")
    else:
        print("[check] PASSED")
    return failures == 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # problem
    ap.add_argument("--lens", default="1000,6000,1000",
                    help="segment lengths of one sample (default 1000,6000,1000)")
    ap.add_argument("--types", default="text,image_gen,text",
                    help="segment types, same count as --lens; only used by the "
                         "full/sparse/cross masks")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--qh", type=int, default=16, help="q heads")
    ap.add_argument("--kvh", type=int, default=8, help="kv heads")
    ap.add_argument("--d", type=int, default=128, help="head dim")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--mask", default="causal", choices=sorted(MASK_MODS),
                    help="mask_mod (default causal: q_idx >= kv_idx)")
    ap.add_argument("--sw", type=int, default=1024, help="sliding window, sparse mask")
    ap.add_argument("--gw", type=int, default=4, help="global window, sparse mask")
    ap.add_argument("--seed", type=int, default=0)
    # plumbing
    ap.add_argument("--repo", default=os.environ.get("MOJO_OPSET_ROOT", ""),
                    help="directory containing the mojo_opset package")
    ap.add_argument("--dump-args", action="store_true",
                    help="print every kernel argument at launch time, using the "
                         "same formatter as dump_flex_args so the output is "
                         "directly diffable against the mojo_opset run")
    ap.add_argument("--dump-file", default=None, metavar="PATH",
                    help="also write that dump to PATH (implies --dump-args)")
    ap.add_argument("--us", type=float, default=None, metavar="T",
                    help="a measured kernel duration in us (e.g. from msprof op); "
                         "the script turns it into sustained TFLOPS so two runs "
                         "can be compared on equal terms")
    ap.add_argument("--via-impl", action="store_true",
                    help="diagnostic A/B: move the block mask to the device and let "
                         "flex_attention_fwd_impl build the args itself instead of "
                         "calling the kernel directly.  Adds a couple of tiny host "
                         "kernels (a zeros fill, maybe one add) -- in msprof look "
                         "only at flex_attention_kernel's own duration")
    # compile options forwarded to the kernel
    ap.add_argument("--intra", type=int, default=3, help="intra_cache_num")
    ap.add_argument("--inter", type=int, default=2, help="inter_cache_num")
    ap.add_argument("--ssbuf", default="default", choices=["default", "on", "off"],
                    help="enable_dynamic_cv_pipeline; 'default' passes nothing, "
                         "matching flex_attention_fwd_impl")
    ap.add_argument("--opt", action="append", default=[], metavar="K=V",
                    help="extra compile option, repeatable "
                         "(e.g. --opt ub_multi_buffer_num=4)")
    ap.add_argument("--bare", action="store_true",
                    help="pass no compile options at all (ignores --intra/--inter)")
    # correctness
    ap.add_argument("--check", action="store_true", help="run the CPU reference check")
    ap.add_argument("--check-heads", type=int, default=2,
                    help="how many q heads to verify, <=0 = all (default 2)")
    ap.add_argument("--check-qblocks", type=int, default=6,
                    help="how many 128-row q blocks to verify, evenly spaced "
                         "across the sequence, <=0 = all (default 6)")
    ap.add_argument("--check-chunk", type=int, default=128,
                    help="q rows per reference matmul chunk")
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=2e-2)
    return ap.parse_args()


def build_compile_options(args):
    if args.bare:
        return {}
    copts = dict(
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True,
        intra_cache_num=args.intra,
        inter_cache_num=args.inter,
        enable_cross_if_fusion=True,
        enable_buffer_insert_optimization=True,
        enable_ub_refine_opt=True,
    )
    if args.ssbuf != "default":
        copts["enable_dynamic_cv_pipeline"] = (args.ssbuf == "on")
    for item in args.opt:
        key, _, val = item.partition("=")
        if val.lower() in ("true", "false"):
            parsed = val.lower() == "true"
        else:
            try:
                parsed = int(val)
            except ValueError:
                parsed = val
        copts[key] = parsed
    return copts


def main():
    args = parse_args()

    # ===== stage 0: imports ==================================================
    import torch_npu  # noqa: F401  registers the npu backend
    fa = _import_kernel_module(args.repo)

    data_lens = [[int(x) for x in args.lens.split(",")]] * args.batch
    data_types = [[t.strip() for t in args.types.split(",")]] * args.batch
    if len(data_lens[0]) != len(data_types[0]):
        raise SystemExit("--lens and --types must have the same number of entries")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # ===== stage 1: problem, on the CPU ======================================
    t0 = time.time()
    prob = build_problem_cpu(args.batch, args.qh, args.kvh, args.d,
                             data_lens, data_types, args.sw, args.gw,
                             dtype, args.seed)
    q_cpu, k_cpu, v_cpu = prob["q"], prob["k"], prob["v"]
    batch, q_heads, q_len, head_dim = q_cpu.shape
    _, kv_heads, kv_len, v_head_dim = k_cpu.shape
    gqa = q_heads // kv_heads if q_heads >= kv_heads else 1
    sm_scale = 1.0 / (head_dim ** 0.5)
    print(f"[cpu ] problem: Z={batch} Hq={q_heads} Hkv={kv_heads} M={q_len} "
          f"D={head_dim} dtype={args.dtype} mask={args.mask} "
          f"({time.time() - t0:.2f}s)")

    # ===== stage 2: block mask, on the CPU ===================================
    t0 = time.time()
    mask_mod = MASK_MODS[args.mask](prob)
    block_m = block_n = fa.TILE_BLOCK_SIZE
    block_mask = fa.create_block_mask_patched(
        mask_mod, B=1, H=1, Q_LEN=q_len, KV_LEN=kv_len,
        device="cpu", BLOCK_SIZE=(block_m, block_n),
    )
    sparse_q, sparse_kv = fa._get_mask_block_sizes(block_mask)
    num_q_blocks = (q_len + sparse_q - 1) // sparse_q
    num_kv_blocks = (kv_len + sparse_kv - 1) // sparse_kv
    print(f"[cpu ] block mask: SPARSE_Q={sparse_q} SPARSE_KV={sparse_kv} "
          f"q_blocks={num_q_blocks} kv_blocks={num_kv_blocks} "
          f"({time.time() - t0:.2f}s)")

    # ===== stage 3: block-mask attrs + load-balance plan, on the CPU =========
    t0 = time.time()
    bm = fa._prepare_block_mask_attrs(block_mask, q_cpu, num_q_blocks,
                                      sparse_q, sparse_kv)
    for key in ("kv_num_blks", "kv_idx", "full_kv_num_blks", "full_kv_idx",
                "dense_mask", "packed_partial_mask", "partial_mask_offsets"):
        if bm[key].device.type != "cpu":
            raise RuntimeError(f"bm[{key!r}] is not on the CPU -- prep leaked to device")

    plan = fa._get_or_build_load_balance_plan(
        block_mask, bm, batch, q_heads, kv_heads, num_q_blocks, num_kv_blocks,
        sparse_q // block_m, 1, torch.device("cpu"), False,
    )
    if plan.q_task_ids.device.type != "cpu":
        raise RuntimeError("q_task_ids is not on the CPU -- plan leaked to device")

    n_partial = int(bm["kv_num_blks"].sum())
    n_full = int(bm["full_kv_num_blks"].sum())
    print(f"[cpu ] plan: use_q_task_list={plan.use_q_task_list}  "
          f"partial blocks={n_partial} full blocks={n_full} "
          f"packed tiles={bm['packed_partial_mask'].shape[0]} "
          f"({time.time() - t0:.2f}s)")

    num_tasks_raw = num_q_blocks * batch * q_heads
    grid, num_tasks = fa._persistent_launch_config(num_tasks_raw)
    print(f"[cpu ] grid={grid} num_tasks={num_tasks} (raw {num_tasks_raw})")

    # ----- work accounting: makes any measured duration interpretable --------
    # the kernel clamps each per-q-block count exactly like this
    cap = max((kv_len + block_n - 1) // block_n, 1)
    kv_mult = sparse_kv // block_n
    part_per_qb = bm["kv_num_blks"].reshape(-1).clamp(max=cap // max(kv_mult, 1))
    full_per_qb = bm["full_kv_num_blks"].reshape(-1).clamp(max=cap // max(kv_mult, 1))
    visited_per_head = int((part_per_qb.sum() + full_per_qb.sum()) * kv_mult)
    # two 128x128x128 matmuls per visited (q block, kv block) pair
    gflop = visited_per_head * 2 * 2 * (block_m * block_n * head_dim) / 1e9 \
        * batch * q_heads
    total_pairs = visited_per_head * batch * q_heads
    n_cores = grid[0]
    print(f"[cpu ] work: visited (q,kv) block pairs per head = {visited_per_head} "
          f"(partial {int(part_per_qb.sum())} + full {int(full_per_qb.sum())})")
    print(f"[cpu ] work: total = {total_pairs} pairs = {gflop:.1f} GFLOP "
          f"over Z*Hq={batch * q_heads} heads")
    print(f"[cpu ] work: per core = {total_pairs / n_cores:.0f} pairs "
          f"= {gflop / n_cores:.2f} GFLOP, {num_tasks / n_cores:.1f} tasks "
          f"(cores={n_cores})")
    if args.us:
        print(f"[cpu ] work: {args.us:.1f} us for {gflop:.1f} GFLOP "
              f"=> {gflop / (args.us * 1e-6) / 1e3:.1f} TFLOPS aggregate, "
              f"{gflop / n_cores / (args.us * 1e-6) / 1e3:.2f} TFLOPS per core")

    # ===== stage 4: stage the inputs on the NPU ==============================
    t0 = time.time()
    dev = torch.device("npu")
    q = q_cpu.contiguous().to(dev)
    k = k_cpu.contiguous().to(dev)
    v = v_cpu.contiguous().to(dev)
    d_kv_num_blks = bm["kv_num_blks"].to(dev)
    d_kv_idx = bm["kv_idx"].to(dev)
    d_full_kv_num_blks = bm["full_kv_num_blks"].to(dev)
    d_full_kv_idx = bm["full_kv_idx"].to(dev)
    d_q_task_ids = plan.q_task_ids.to(dev)
    d_dense_mask = bm["dense_mask"].to(dev)
    d_packed = bm["packed_partial_mask"].to(dev)
    d_offsets = bm["partial_mask_offsets"].to(dev)

    output = torch.empty_like(q)
    lse = torch.empty((batch, q_heads, q_len), dtype=torch.float32, device=dev)
    torch.npu.synchronize()
    print(f"[npu ] inputs staged ({time.time() - t0:.2f}s)")

    # ===== stage 5: compile options ==========================================
    copts = build_compile_options(args)

    if args.dump_args or args.dump_file:
        # Same formatter the mojo_opset side uses, so the two dumps diff cleanly.
        try:
            import dump_flex_args
        except ImportError:
            print("[args] dump_flex_args.py not found next to this script -- "
                  "copy it alongside and retry")
        else:
            dump_flex_args.install(args.dump_file, limit=1, module=fa)

    # ===== stage 6: THE single launch ========================================
    if args.via_impl:
        # Diagnostic path: hand the device-side block mask to the real host
        # wrapper and let it assemble the kernel arguments itself.  If this and
        # the direct call below give the same duration, the direct call is a
        # faithful reproduction of the production path.
        for name in ("kv_num_blocks", "kv_indices",
                     "full_kv_num_blocks", "full_kv_indices",
                     "q_num_blocks", "q_indices",
                     "full_q_num_blocks", "full_q_indices",
                     "packed_partial_mask", "partial_mask_offsets",
                     "partial_block_table"):
            attr = getattr(block_mask, name, None)
            if isinstance(attr, torch.Tensor):
                setattr(block_mask, name, attr.to(dev))
        # drop the CPU-built plan so the wrapper rebuilds it for the device
        for name in ("_load_balance_plan_cache_key", "_load_balance_plan"):
            if hasattr(block_mask, name):
                delattr(block_mask, name)
        print("[npu ] launching via flex_attention_fwd_impl (diagnostic A/B)")
        t0 = time.time()
        output, lse = fa.flex_attention_fwd_impl(q, k, v, block_mask)
        torch.npu.synchronize()
        print(f"[npu ] done (wall {time.time() - t0:.3f}s, includes JIT compile)")
        if args.check:
            ok = run_check(prob, output, lse, as_2d_mask_fn(mask_mod), sm_scale, gqa,
                           num_q_blocks, block_m,
                           args.check_heads, args.check_qblocks, args.check_chunk,
                           args.rtol, args.atol)
            return 0 if ok else 1
        return 0

    print("[npu ] launching flex_attention_kernel -- 1 launch, no warm-up")
    t0 = time.time()
    fa.flex_attention_kernel[grid](
        q, k, v,
        d_kv_num_blks, d_kv_idx, d_full_kv_num_blks, d_full_kv_idx,
        d_q_task_ids,
        d_dense_mask, d_dense_mask.stride(2), d_dense_mask.stride(3),
        d_packed, d_offsets,
        d_packed.stride(0), d_packed.stride(1), d_packed.stride(2),
        d_offsets.stride(2),
        output, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        d_kv_idx.stride(2),
        SM_SCALE=sm_scale,
        QK_HEAD_DIM=head_dim,
        V_HEAD_DIM=v_head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_TASKS=num_tasks,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_HEAD=q_heads,
        SPARSE_Q_BLOCK_SIZE=sparse_q,
        SPARSE_KV_BLOCK_SIZE=sparse_kv,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        GQA_SHARED_HEADS=gqa,
        HAS_FULL_BLOCKS=True,
        USE_PACKED_PARTIAL_MASK=bm["use_packed_partial_mask"],
        USE_Q_TASK_LIST=plan.use_q_task_list,
        **copts,
    )
    torch.npu.synchronize()
    print(f"[npu ] done (wall {time.time() - t0:.3f}s, includes JIT compile)")

    # ===== stage 7: optional correctness, on the CPU =========================
    if args.check:
        ok = run_check(prob, output, lse, as_2d_mask_fn(mask_mod), sm_scale, gqa,
                       num_q_blocks, block_m,
                       args.check_heads, args.check_qblocks, args.check_chunk,
                       args.rtol, args.atol)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
