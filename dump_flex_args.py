#!/usr/bin/env python3
"""Dump every argument of `flex_attention_kernel` at launch time.

Purpose: get a comparable record of the kernel invocation from two different
hosts -- the mojo_opset test suite and the standalone `test_flex_fwd_single.py`
-- so that `diff` finds the difference instead of anyone guessing.

The dump contains:
  * grid and every positional argument, labelled with the kernel's own
    parameter names (tensors as shape/dtype/stride/device, scalars by value)
  * every keyword argument, split into kernel constexprs and compile options
  * work accounting reconstructed from KV_NUM_BLKS / FULL_KV_NUM_BLKS: how many
    (q block, kv block) pairs the kernel will actually visit.  That is the
    number which, divided by the measured duration, is comparable across runs.

It launches no kernels: device tensors are brought to the host with `.cpu()`
(a memcpy, not an AI Core task) before anything is summed.

Use inside the mojo_opset suite
-------------------------------
Put this file on PYTHONPATH and add two lines at the top of your test:

    import dump_flex_args
    dump_flex_args.install("real.flexdump")

or without touching the test file:

    FLEX_DUMP=real.flexdump python -c "import dump_flex_args, runpy; \
dump_flex_args.install(); runpy.run_path('your_test.py', run_name='__main__')"

For pytest:

    FLEX_DUMP=real.flexdump python -c "import sys, dump_flex_args, pytest; \
dump_flex_args.install(); sys.exit(pytest.main(sys.argv[1:]))" -k your_test -s

Then compare with the standalone script:

    python test_flex_fwd_single.py --dump-file mine.flexdump
    diff real.flexdump mine.flexdump
"""

import os
import sys

_installed = False
_dumped = 0


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _is_tensor(obj):
    return hasattr(obj, "shape") and hasattr(obj, "stride") and hasattr(obj, "dtype")


def _fmt_tensor(t):
    try:
        stride = tuple(t.stride())
    except Exception:
        stride = "?"
    try:
        contig = bool(t.is_contiguous())
    except Exception:
        contig = "?"
    return (f"shape={tuple(t.shape)} dtype={t.dtype} stride={stride} "
            f"dev={t.device.type} contig={contig}")


def _host_list(t):
    """Flat python list of an int tensor, host side, no kernels."""
    try:
        return t.detach().cpu().reshape(-1).tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# work accounting -- mirrors the kernel's own trip-count arithmetic
# ---------------------------------------------------------------------------


def work_lines(named):
    """Reconstruct how many (q block, kv block) pairs the kernel will visit.

    Kernel source, per task:
        q_start               = task_id % NUM_Q_BLOCKS
        SPARSE_Q_MULTIPLE     = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE    = SPARSE_KV_BLOCK_SIZE // BLOCK_N
        q_sparse_idx          = q_start // SPARSE_Q_MULTIPLE
        kv_num_blocks         = load(KV_NUM_BLKS + q_sparse_idx)
        block_n_end           = min(kv_num_blocks * SPARSE_KV_MULTIPLE,
                                    max(cdiv(KV_LEN, BLOCK_N), 1))
    and the same again with FULL_KV_NUM_BLKS when HAS_FULL_BLOCKS.
    """
    out = []
    try:
        kv_counts = _host_list(named["KV_NUM_BLKS"])
        full_counts = _host_list(named["FULL_KV_NUM_BLKS"])
        kv_len = int(named["KV_LEN"])
        block_m = int(named["BLOCK_M"])
        block_n = int(named["BLOCK_N"])
        head_dim = int(named["QK_HEAD_DIM"])
        sparse_q = int(named["SPARSE_Q_BLOCK_SIZE"])
        sparse_kv = int(named["SPARSE_KV_BLOCK_SIZE"])
        num_q_blocks = int(named["NUM_Q_BLOCKS"])
        num_tasks = int(named["NUM_TASKS"])
        has_full = bool(named["HAS_FULL_BLOCKS"])
    except Exception as exc:
        return [f"work: unavailable ({type(exc).__name__}: {exc})"]

    q_mult = max(sparse_q // block_m, 1)
    kv_mult = max(sparse_kv // block_n, 1)
    cap = max(-(-kv_len // block_n), 1)                     # cdiv(KV_LEN, BLOCK_N)

    def trips(counts):
        if counts is None:
            return 0, []
        per_qb = []
        for q_start in range(num_q_blocks):
            sparse_idx = q_start // q_mult
            n = counts[sparse_idx] if sparse_idx < len(counts) else 0
            per_qb.append(min(n * kv_mult, cap))
        return sum(per_qb), per_qb

    part_total, part_per_qb = trips(kv_counts)
    if has_full:
        full_total, full_per_qb = trips(full_counts)
    else:
        full_total, full_per_qb = 0, [0] * num_q_blocks

    per_head = part_total + full_total
    heads = (num_tasks / num_q_blocks) if num_q_blocks else 0
    total_pairs = per_head * heads
    gflop = total_pairs * 2 * 2 * (block_m * block_n * head_dim) / 1e9

    out.append(f"work: num_q_blocks={num_q_blocks} SPARSE_Q_MULTIPLE={q_mult} "
               f"SPARSE_KV_MULTIPLE={kv_mult} trip_cap={cap}")
    out.append(f"work: Z*Hq={heads:g}  (NUM_TASKS/NUM_Q_BLOCKS)")
    out.append(f"work: pairs per head = {per_head}  "
               f"(partial {part_total} + full {full_total})")
    out.append(f"work: pairs total    = {total_pairs:g}")
    out.append(f"work: matmul         = {gflop:.1f} GFLOP")
    out.append(f"work: rows covered   = {num_q_blocks * block_m} of Q_LEN="
               f"{int(named.get('Q_LEN', 0))}")
    out.append(f"work: trips/qblock partial = {part_per_qb}")
    out.append(f"work: trips/qblock full    = {full_per_qb}")
    if gflop:
        out.append("work: divide GFLOP by the measured duration to compare runs")
    return out


# ---------------------------------------------------------------------------
# the dump itself
# ---------------------------------------------------------------------------


def format_call(arg_names, args, kwargs, grid):
    out = [f"grid: {grid}", f"positional: {len(args)}"]

    named = {}
    for i, val in enumerate(args):
        name = arg_names[i] if arg_names and i < len(arg_names) else f"arg{i}"
        named[name] = val
        rendered = _fmt_tensor(val) if _is_tensor(val) else repr(val)
        out.append(f"  [{i:2d}] {name:24s} {rendered}")

    known = set(arg_names or ())
    kernel_kw, opt_kw = {}, {}
    for key, val in kwargs.items():
        if key in ("grid", "warmup", "num_warps", "num_stages", "num_ctas"):
            continue
        (kernel_kw if key in known else opt_kw)[key] = val
    named.update(kernel_kw)

    out.append("kernel keywords:")
    for key in sorted(kernel_kw):
        val = kernel_kw[key]
        out.append(f"  {key:24s} = "
                   + (_fmt_tensor(val) if _is_tensor(val) else repr(val)))
    out.append("compile options:")
    for key in sorted(opt_kw):
        out.append(f"  {key:40s} = {opt_kw[key]!r}")

    out.extend(work_lines(named))
    return "\n".join(out)


def emit(text, path=None, label="flex_attention_kernel"):
    blob = (f"===== {label} launch ".ljust(60, "=") + "\n"
            + text + "\n" + "=" * 60 + "\n")
    sys.stdout.write(blob)
    sys.stdout.flush()
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(blob)


# ---------------------------------------------------------------------------
# interception
# ---------------------------------------------------------------------------


def install(path=None, limit=1, module=None):
    """Patch flex_attention_kernel so the first `limit` launches are dumped.

    `path` defaults to $FLEX_DUMP.  `limit=0` dumps every launch.
    """
    global _installed
    if _installed:
        return
    path = path or os.environ.get("FLEX_DUMP") or None
    if path and os.path.exists(path):
        os.remove(path)

    if module is None:
        import importlib
        importlib.import_module("mojo_opset.backends.ttx.kernels.npu")
        module = importlib.import_module(
            "mojo_opset.backends.ttx.kernels.npu.flex_attention")

    kernel = getattr(module, "flex_attention_kernel")
    arg_names = tuple(getattr(kernel, "arg_names", ()) or ())
    original = kernel.run

    def run(*args, **kwargs):
        global _dumped
        if limit == 0 or _dumped < limit:
            _dumped += 1
            try:
                text = format_call(arg_names, args, kwargs, kwargs.get("grid"))
            except Exception as exc:
                text = f"(dump failed: {type(exc).__name__}: {exc})"
            emit(text, path)
        return original(*args, **kwargs)

    kernel.run = run
    _installed = True
    print(f"[dump_flex_args] installed on {module.__name__}.flex_attention_kernel"
          + (f", writing {path}" if path else ""))


def install_deferred(path=None, limit=1, kernel_name="flex_attention_kernel"):
    """Arm the dump without importing anything.

    Wraps builtins.__import__ and patches the kernel the moment the module that
    owns it shows up in sys.modules.  Safe to call before torch / torch_npu /
    triton exist, so it can run from a launcher ahead of the test.
    """
    import builtins

    state = {"done": False}
    real_import = builtins.__import__

    def try_patch():
        for name, mod in list(sys.modules.items()):
            if mod is None or "flex_attention" not in name:
                continue
            kernel = getattr(mod, kernel_name, None)
            if kernel is None or not hasattr(kernel, "run"):
                continue
            install(path, limit=limit, module=mod)
            state["done"] = True
            builtins.__import__ = real_import
            return True
        return False

    def hooked_import(name, *args, **kwargs):
        mod = real_import(name, *args, **kwargs)
        if not state["done"] and ("flex_attention" in name or "mojo_opset" in name):
            try_patch()
        return mod

    builtins.__import__ = hooked_import
    print(f"[dump_flex_args] armed (deferred); will patch {kernel_name} on import"
          + (f", writing {path}" if (path or os.environ.get('FLEX_DUMP')) else ""))


if os.environ.get("FLEX_DUMP_AUTO") == "1":
    try:
        install_deferred()
    except Exception as exc:
        print(f"[dump_flex_args] auto-arm failed: {exc}")
