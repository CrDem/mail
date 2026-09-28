"""Does inter-core double buffering pay on a one-way main loop?

Both arms run with the CV pipeline ON and differ in exactly one thing, the
inter-core buffer count, passed as a compile option:

    buf_slot_num_of_veccore    -> intra  (ssbuffer.intra_buf_count)
    buf_slot_num_of_crosscore  -> inter  (ssbuffer.inter_core_buf_count)

Defaults in BufferCountManager.cpp are intra=3, inter=2, load=1, so both arms
pin intra=1 and vary only inter.

Kernels:

  oneway_cc   VECTOR -> CUBE every iteration, nothing coming back. The pipeline
              takes it only because the chained matmul's cast folds into a
              fixpipe whose destination is L1 -- CUBE-local traffic that
              isEveryMainLoopOneWayByOpKind counts as cross-core while
              MarkMainLoop ignores it (isL1Fixpipe, MarkMainLoop.cpp:50).
              This is the kernel we expect inter=2 to do nothing for.

  roundtrip   A real C->V->C round trip. Positive control: if inter=2 does not
              pay here either, the measurement is not sensitive and the flat
              result on oneway_cc means nothing.

Verified on the IR (BLK=64, K=64, this exact source) before running:

    oneway_cc  inter 1 -> 2 : copies 1->2, cross_buffer ifs 0->6,  fixpipe stays 1
    roundtrip  inter 1 -> 2 : copies 1->2, cross_buffer ifs 0->12, fixpipe 1->2

Run:
    ASCEND_RT_VISIBLE_DEVICES=1 python test_inter_buffer_oneway.py
    PROFILE=0 python test_inter_buffer_oneway.py       # wall clock only
    BLK=128 K=32 NPROG=28 python test_inter_buffer_oneway.py
"""

import csv
import glob
import os
import shutil
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

BLK = int(os.environ.get("BLK", 64))
K = int(os.environ.get("K", 64))
NPROG = int(os.environ.get("NPROG", 28))
WARMUP = int(os.environ.get("WARMUP", 10))
ITERS = int(os.environ.get("ITERS", 50))
PROFILE = os.environ.get("PROFILE", "1") != "0"
PROF_DIR = os.environ.get("PROF_DIR", "./prof_inter_buffer")

CONFIGS = [(1, 1), (1, 2)]


@triton.jit
def oneway_cc(A, B, O, K: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BLK)
    on = tl.arange(0, BLK)
    idx = om[:, None] * BLK + on[None, :]
    base = pid * K * BLK * BLK
    acc = tl.zeros((BLK, BLK), dtype=tl.float32)
    for k in range(0, K):
        a = tl.load(A + base + idx + k * BLK * BLK)
        v = tl.exp(a.to(tl.float32)).to(tl.float16)     # VECTOR
        b = tl.load(B + base + idx + k * BLK * BLK)
        p = tl.dot(v, b)                                # CUBE consumes v
        acc = tl.dot(p.to(tl.float16), b, acc)          # CUBE, L0C -> L1 -> L0C
    tl.store(O + pid * BLK * BLK + idx, acc)


@triton.jit
def roundtrip(A, B, O, K: tl.constexpr, BLK: tl.constexpr):
    pid = tl.program_id(0)
    om = tl.arange(0, BLK)
    on = tl.arange(0, BLK)
    idx = om[:, None] * BLK + on[None, :]
    base = pid * K * BLK * BLK
    acc = tl.zeros((BLK, BLK), dtype=tl.float32)
    for k in range(0, K):
        a = tl.load(A + base + idx + k * BLK * BLK)
        b = tl.load(B + base + idx + k * BLK * BLK)
        p = tl.dot(a, b)                                # CUBE
        pv = tl.exp(p).to(tl.float16)                   # VECTOR
        acc += tl.dot(pv, b)                            # CUBE -> VECTOR
    tl.store(O + pid * BLK * BLK + idx, acc)


KERNELS = [("oneway_cc", oneway_cc), ("roundtrip", roundtrip)]


def make_inputs():
    torch.manual_seed(0)
    a = (torch.randn(NPROG, K, BLK, BLK) * 0.05).to(torch.float16).npu()
    b = (torch.randn(NPROG, K, BLK, BLK) * 0.05).to(torch.float16).npu()
    out = torch.zeros(NPROG, BLK, BLK, dtype=torch.float32).npu()
    return a, b, out


def launch(kernel, a, b, out, intra, inter):
    kernel[(NPROG, )](a, b, out, K=K, BLK=BLK,
                      buf_slot_num_of_veccore=intra,
                      buf_slot_num_of_crosscore=inter)


def wall_clock(kernel, intra, inter):
    a, b, out = make_inputs()
    for _ in range(WARMUP):
        launch(kernel, a, b, out, intra, inter)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        launch(kernel, a, b, out, intra, inter)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / ITERS, out.cpu()


def read_kernel_details(out_dir):
    """Average the duration column of kernel_details.csv.

    Column names move between CANN versions, so the duration and name columns
    are found by substring rather than assumed.
    """
    paths = glob.glob(os.path.join(out_dir, "**", "kernel_details.csv"),
                      recursive=True)
    if not paths:
        return None
    values = []
    with open(paths[0], newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        dur_col = next((c for c in fields if "duration" in c.lower()), None)
        name_col = next((c for c in fields if "name" in c.lower()), None)
        if dur_col is None:
            return None
        for row in reader:
            name = (row.get(name_col) or "") if name_col else ""
            lowered = name.lower()
            if name and not ("triton" in lowered or "oneway" in lowered
                             or "roundtrip" in lowered):
                continue
            try:
                values.append(float(row[dur_col]))
            except (TypeError, ValueError):
                continue
    if not values:
        return None
    return sum(values) / len(values)


def device_time_us(kernel, intra, inter, tag):
    """Average device duration of the kernel, read back from the profiler."""
    out_dir = os.path.join(PROF_DIR, tag)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    a, b, out = make_inputs()
    for _ in range(WARMUP):
        launch(kernel, a, b, out, intra, inter)
    torch.npu.synchronize()

    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
    )
    with prof:
        for _ in range(ITERS):
            launch(kernel, a, b, out, intra, inter)
            prof.step()
    torch.npu.synchronize()

    return read_kernel_details(out_dir)


def main():
    print("BLK=%d K=%d programs=%d  %d iters after %d warmup  profiler=%s"
          % (BLK, K, NPROG, ITERS, WARMUP, "on" if PROFILE else "off"))
    print()

    results = {}
    for label, kernel in KERNELS:
        checksums = []
        for intra, inter in CONFIGS:
            ms, res = wall_clock(kernel, intra, inter)
            us = None
            if PROFILE:
                try:
                    us = device_time_us(kernel, intra, inter,
                                        "%s_intra%d_inter%d" % (label, intra, inter))
                except Exception as ex:  # noqa: BLE001
                    print("    profiler failed (%s: %s); wall clock still valid"
                          % (type(ex).__name__, str(ex)[:120]))
            results[(label, intra, inter)] = (ms, us)
            checksums.append(res.double().sum().item())
            print("%-12s intra%d/inter%d   wall %8.4f ms   device %s"
                  % (label, intra, inter, ms,
                     ("%9.2f us" % us) if us is not None else "    n/a   "))

        if abs(checksums[0] - checksums[1]) > 1e-6:
            print("    results differ between the configs (%r vs %r) -- that is "
                  "a correctness problem and outranks the timing" % tuple(checksums))
        print()

    def gain(label):
        one = results[(label, 1, 1)]
        two = results[(label, 1, 2)]
        pick = 1 if (one[1] is not None and two[1] is not None) else 0
        before, after = one[pick], two[pick]
        return (before - after) / before * 100.0

    g_oneway = gain("oneway_cc")
    g_control = gain("roundtrip")
    print("inter=2 vs inter=1:   one-way %+.1f%%    control %+.1f%%"
          % (g_oneway, g_control))
    print()

    if g_control < 3.0:
        print("The control barely moved, so this setup cannot tell whether double "
              "buffering pays at all. Raise K or BLK and repeat before reading "
              "anything into the one-way number.")
    elif g_oneway < 3.0:
        print("Double buffering pays on the round trip and not on the one-way loop: "
              "guard the CUBE->CUBE case, i.e. give the op-kind count the "
              "isL1Fixpipe filter MarkMainLoop already uses.")
    else:
        print("Double buffering pays on the one-way loop too, so one-way "
              "communication is worth pipelining: propose allowing it, with this "
              "kernel minus the chained matmul as the example.")


if __name__ == "__main__":
    main()
