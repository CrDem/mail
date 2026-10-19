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
        have = sorted(k for k in named if not k.startswith("arg"))
        return [f"work: unavailable ({type(exc).__name__}: {exc})",
                f"work: names seen = {have}"]

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


_patched_ids = set()
_class_patched = False
_in_launch = False
_installing = False


def _dump_once(arg_names, args, kwargs, path, limit, origin):
    global _dumped
    if limit and _dumped >= limit:
        return
    _dumped += 1
    try:
        text = format_call(arg_names, args, kwargs, kwargs.get("grid"))
    except Exception as exc:
        text = f"(dump failed: {type(exc).__name__}: {exc})"
    emit(f"origin: {origin}" + chr(10) + text, path)


def _patch_instance(kernel, path, limit, origin):
    """Wrap one JITFunction object's run()."""
    if id(kernel) in _patched_ids or not hasattr(kernel, "run"):
        return False
    original = kernel.run
    names = tuple(getattr(kernel, "arg_names", ()) or ())

    def run(*args, **kwargs):
        global _in_launch
        _in_launch = True
        try:
            _dump_once(names, args, kwargs, path, limit, origin)
            return original(*args, **kwargs)
        finally:
            _in_launch = False

    kernel.run = run
    _patched_ids.add(id(kernel))
    return True


def _patch_class(path, limit, kernel_name):
    """Wrap JITFunction.run itself and filter by kernel name.

    This is the primary hook: it does not care which arch module (a2 / a5 /
    the dispatched public module) owns the kernel object.
    """
    global _class_patched
    if _class_patched:
        return False
    try:
        from triton.runtime.jit import JITFunction
    except Exception:
        return False
    original = JITFunction.run

    def run(self, *args, **kwargs):
        if not _in_launch and getattr(self.fn, "__name__", "") == kernel_name:
            origin = f"{getattr(self.fn, '__module__', '?')}.{kernel_name} [class hook]"
            _dump_once(tuple(getattr(self, "arg_names", ()) or ()),
                       args, kwargs, path, limit, origin)
        return original(self, *args, **kwargs)

    JITFunction.run = run
    _class_patched = True
    return True


def _candidate_kernels(kernel_name):
    """Every distinct kernel object reachable under that name, dispatched first."""
    found = []
    seen = set()
    preferred = "mojo_opset.backends.ttx.kernels.npu.flex_attention"
    names = [preferred] + [n for n in sorted(sys.modules)
                           if n != preferred and n.endswith(".flex_attention")]
    for name in names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        kernel = getattr(mod, kernel_name, None)
        if kernel is None or id(kernel) in seen:
            continue
        seen.add(id(kernel))
        found.append((name, kernel))
    return found


def install(path=None, limit=1, module=None, kernel_name="flex_attention_kernel"):
    """Arm the dump on everything that could launch the kernel.

    Patches JITFunction.run (arch-agnostic) plus every distinct kernel object
    found, so it does not matter whether a2, a5 or the dispatched module is the
    one actually used.  `limit=0` dumps every launch.
    """
    global _installed, _installing
    if _installing:          # _patch_class imports triton, which re-enters the
        return False         # import hook; ignore the nested call
    path = path or os.environ.get("FLEX_DUMP") or None
    if not _installed and path and os.path.exists(path):
        os.remove(path)

    _installing = True
    try:
        return _install_inner(path, limit, module, kernel_name)
    finally:
        _installing = False


def _install_inner(path, limit, module, kernel_name):
    global _installed
    done = []
    if _patch_class(path, limit, kernel_name):
        done.append("triton.runtime.jit.JITFunction.run [class hook]")

    if module is not None:
        kernel = getattr(module, kernel_name, None)
        if kernel is not None and _patch_instance(
                kernel, path, limit, f"{module.__name__}.{kernel_name}"):
            done.append(f"{module.__name__}.{kernel_name}")
    else:
        for name, kernel in _candidate_kernels(kernel_name):
            if _patch_instance(kernel, path, limit, f"{name}.{kernel_name}"):
                done.append(f"{name}.{kernel_name}")

    if done:
        _installed = True
        for what in done:
            print(f"[dump_flex_args] hooked {what}"
                  + (f", writing {path}" if path else ""))
    return bool(done)


def install_deferred(path=None, limit=1, kernel_name="flex_attention_kernel"):
    """Arm the dump without importing anything.

    Wraps builtins.__import__ and installs the hooks as soon as triton (and the
    kernel modules) show up.  Safe to call before torch / torch_npu / triton
    exist, so it can run from a launcher ahead of the test.
    """
    import builtins

    path = path or os.environ.get("FLEX_DUMP") or None
    if path and os.path.exists(path):
        os.remove(path)

    real_import = builtins.__import__
    state = {"armed": True}

    def try_patch():
        install(path, limit=limit, kernel_name=kernel_name)
        # the class hook alone covers every arch module, so stop once it is in
        if _class_patched and state["armed"]:
            state["armed"] = False
            builtins.__import__ = real_import

    def hooked_import(name, *args, **kwargs):
        mod = real_import(name, *args, **kwargs)
        if state["armed"] and ("flex_attention" in name or "mojo_opset" in name
                               or name == "triton" or name.startswith("triton.")):
            try_patch()
        return mod

    builtins.__import__ = hooked_import
    print(f"[dump_flex_args] armed (deferred) for {kernel_name}"
          + (f", writing {path}" if path else ""))


if os.environ.get("FLEX_DUMP_AUTO") == "1":
    try:
        install_deferred()
    except Exception as exc:
        print(f"[dump_flex_args] auto-arm failed: {exc}")


# ---------------------------------------------------------------------------
# launcher mode: python dump_flex_args.py <target.py> [args...]
# ---------------------------------------------------------------------------


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage: python dump_flex_args.py <target.py> [args...]")
        return 2

    import runpy

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    install_deferred(os.environ.get("FLEX_DUMP"),
                     limit=int(os.environ.get("FLEX_DUMP_LIMIT", "1")))

    target = os.path.abspath(argv[1])
    if not os.path.exists(target):
        print(f"[dump_flex_args] target not found: {target}")
        return 2
    target_dir = os.path.dirname(target)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    sys.argv = [target] + list(argv[2:])
    print(f"[dump_flex_args] running {target}")
    runpy.run_path(target, run_name="__main__")
    if _dumped == 0:
        print("[dump_flex_args] WARNING: the kernel was never launched, or the "
              "patch did not attach -- no dump was written")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
