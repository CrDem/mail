"""A buffer-carried cross-core dependency that the dynamic CV pipeline races on.

What the kernel does, once per iteration:

    p  = dot(a, b)                  # CUBE
    pv = (p + k).to(f16)            # VECTOR, a different value every iteration
    store(WS, pv)                   # VECTOR writes the workspace
    w  = load(WS)                   # CUBE reads the same workspace back
    acc += dot(pv, b) - dot(w, b)   # zero exactly when w is this iteration's pv

`w` and `pv` hold the same numbers, so both dots take identical inputs and the
difference is exactly zero -- no tolerance needed. Any non-zero element means
the CUBE side read the workspace while it held some other iteration's value.

Why this races. The handoff from VECTOR to CUBE goes through memory, not
through a value the pipeline routes, so InterCoreTransferAndSync classifies it
as a memory dependency: it emits one sync_block_set / sync_block_wait pair on
PIPE_MTE2 and nothing else. A real transfer gets five sync ops -- "data ready"
forward, "buffer free" back, a credit before the loop and a drain after it --
plus a rotatable buffer from AllocMultiCache. The memory dependency gets no
back-signal and no second buffer, because the buffer belongs to the kernel.
While producer and consumer stay in the same pipeline stage that is enough.
Once AddControlFlowCondition puts them in different stages, the VECTOR core of
iteration i+1 overwrites the workspace while the CUBE core of iteration i is
still reading it.

Confirmed on the IR (ttadapter, upstream e28f03f70): the pipeline applies --
two scope.scope, three linalg.matmul, the store/load pair is not folded -- and
the memdep sync ops land inside both loops:

    vector loop 104..269, sync_block_set [<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] at 173
    cube   loop 303..526, sync_block_wait[<CUBE>,   <PIPE_MTE3>, <PIPE_MTE2>] at 422

Expected outcome:

    test_memdep_race_ssbuffer_off  -> passes
    test_memdep_race_ssbuffer_on   -> fails (default enables the pipeline on
                                      this target)

Run:
    pytest -s test_ssbuf_memdep_race.py
or
    python test_ssbuf_memdep_race.py
"""

import torch
import torch_npu  # noqa: F401  (registers the npu device)
import triton
import triton.language as tl

BLK = 128
K = 8


@triton.jit
def ssbuf_memdep_race(A, B, WS, O, K: tl.constexpr, BLK: tl.constexpr):
    om = tl.arange(0, BLK)
    on = tl.arange(0, BLK)
    idx = om[:, None] * BLK + on[None, :]
    acc = tl.zeros((BLK, BLK), dtype=tl.float32)
    for k in range(0, K):
        a = tl.load(A + idx + k * BLK * BLK)
        b = tl.load(B + idx + k * BLK * BLK)
        p = tl.dot(a, b)
        pv = (p + k.to(tl.float32)).to(tl.float16)
        tl.store(WS + idx, pv)
        w = tl.load(WS + idx)
        acc += tl.dot(pv, b) - tl.dot(w, b)
    tl.store(O + idx, acc)


def run_once(enable_cv_pipeline):
    """Run the kernel once; returns the output tile on the host.

    `enable_cv_pipeline=None` leaves the option alone, i.e. the default, which
    on this target turns the dynamic CV pipeline on.
    """
    torch.manual_seed(0)
    a = (torch.randn(K, BLK, BLK) * 0.05).to(torch.float16).npu()
    b = (torch.randn(K, BLK, BLK) * 0.05).to(torch.float16).npu()
    ws = torch.zeros(BLK, BLK, dtype=torch.float16).npu()
    out = torch.full((BLK, BLK), float("nan"), dtype=torch.float32).npu()

    kwargs = {}
    if enable_cv_pipeline is not None:
        kwargs["enable_dynamic_cv_pipeline"] = enable_cv_pipeline

    ssbuf_memdep_race[(1, )](a, b, ws, out, K=K, BLK=BLK, **kwargs)
    torch.npu.synchronize()
    return out.cpu()


def report(name, res):
    bad = torch.nonzero(res != 0)
    worst = res.abs().max().item()
    print("%-34s max|acc| = %-12g non-zero elements = %d / %d"
          % (name, worst, bad.shape[0], res.numel()))
    if bad.shape[0]:
        i, j = bad[0].tolist()
        print("    first mismatch at [%d, %d]: %g" % (i, j, res[i, j].item()))
    return bad.shape[0] == 0


def test_memdep_race_ssbuffer_off():
    """Reference run: the same kernel without the dynamic CV pipeline."""
    res = run_once(enable_cv_pipeline=False)
    assert report("ssbuffer OFF", res), \
        "the kernel is wrong even without the CV pipeline -- the test itself " \
        "is broken, not the pipeline"


def test_memdep_race_ssbuffer_on():
    """Default behaviour. Expected to fail: this is the bug being reported."""
    res = run_once(enable_cv_pipeline=None)
    assert report("ssbuffer ON (default)", res), \
        "the CUBE core read the workspace while it held another iteration's " \
        "value: the buffer-carried VECTOR->CUBE dependency has only a forward " \
        "signal and an unrotated buffer, so pipelining it is a " \
        "write-after-read race"


if __name__ == "__main__":
    print("BLK = %d, K = %d" % (BLK, K))
    off = run_once(enable_cv_pipeline=False)
    ok_off = report("ssbuffer OFF", off)
    on = run_once(enable_cv_pipeline=None)
    ok_on = report("ssbuffer ON (default)", on)

    print()
    if ok_off and not ok_on:
        print("REPRODUCED: correct without the pipeline, wrong with it.")
    elif ok_off and ok_on:
        print("No race observed. The producer and the consumer probably ended "
              "up in the same pipeline stage; try a larger K, or check that "
              "the pipeline actually applied (TRITON_KERNEL_DUMP=1 and look "
              "for scope.scope plus a PIPE_MTE2 sync pair).")
    else:
        print("The reference run is already wrong -- fix the test before "
              "reading anything into the second run.")
