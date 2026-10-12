# Correct Multi-Buffering of Vector-Side Store-Out

## Summary

The store-out buffers allocated by ssbuffer are not recognized by the downstream
multi-buffer pass in NPU-IR, because they do not satisfy that pass's
alloc-placement requirement. As a result the loop remains serialized even when
several buffers are allocated. Relocating the alloc and attaching an explicit
multi-buffer mark makes the two mechanisms cooperate: ssbuffer stays enabled and
keeps its role, while buffer rotation is performed by NPU-IR. Overlap is
achieved and execution time decreases from 35 to 28.

## 1. Configurations Evaluated

| # | Configuration | Buffer rotation | Result |
|---|---|---|---|
| A | No multi-buffering | none | Store-out is serialized; src→dst wait dominates |
| B | Multi-cache inside ssbuffer | allocated but not recognized downstream | Still serialized; UB consumed without benefit |
| C | **Adopted:** alloc relocation + explicit mark | performed by NPU-IR | Overlap achieved; 35 → 28 |

## 2. Configuration A — Baseline

With a single store-out buffer, the write of iteration `i+1` is ordered against
the read of iteration `i` through that buffer. The resulting wait on src→dst is
on the critical path, and src→dst itself is largely idle due to crossbar
contention. The kernel is therefore limited by serialized store-out rather than
by computation.

> **[Trace 1 — baseline: src→dst wait and crossbar idle time]**

## 3. Configuration B — Multi-Cache Inside ssbuffer

Several tensors are allocated for store-out through ssbuffer's own multi-cache.
The buffers are created, but **NPU-IR does not recognize them as a
multi-buffer**. The observable symptom is that only **one event id** is assigned
on the MTE3 synchronization, i.e. the downstream compiler still assumes a single
buffer. The write of iteration `i+1` remains ordered against the read of
iteration `i` (WAR through the same buffer) and the loop stays in lock-step.
UB capacity is consumed with no gain.

### Root Cause

The `hivm-mark-multi-buffer` pattern for `hivm.hir.store` / `hivm.hir.load`
requires the alloc to have a valid parent loop
(`lib/Dialect/HIVM/Transforms/MarkMultiBuffer.cpp`):

```cpp
auto *allocOp = utils::tracebackMemRef(v).getDefiningOp();
if (!utils::isAllocLikeOp(allocOp))            return failure();
if (isMarked(allocOp))                         return failure();
auto parentLoop = mlir::hivm::getParentLoop(allocOp->getResult(0));
if (!parentLoop) {
  LLVM_DEBUG(DBGS() << " allocOp has no proper parent loop.\n");
  return failure();                            // exit taken in configuration B
}
```

`getParentLoop` (`lib/Dialect/HIVM/Utils/Utils.cpp:1481`) resolves the parent
loop as follows:

```cpp
LoopLikeOpInterface getParentLoop(Value val) { return getParentLoopImpl(val, nullptr); }

LoopLikeOpInterface getParentLoopImpl(Value val, LoopLikeOpInterface consumerLoop) {
  auto *valDefOp = val.getDefiningOp();
  LoopLikeOpInterface parentLoop = valDefOp->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) return consumerLoop;                 // no enclosing loop -> nullptr
  if (isConsumedInLoop(val, parentLoop)) consumerLoop = parentLoop;
  ...
  if (auto idx = getYieldValueIdx(val, yieldedValues)) {
    ...
    return getParentLoopImpl(result, consumerLoop);      // follows the loop-carried chain
  }
  ...
}
```

ssbuffer hoists its allocs **above the main loop**, into the scope body. In that
position `getParentOfType<LoopLikeOpInterface>()` returns null, `getParentLoop`
returns nullptr, and the pattern exits before any marking occurs. Consequently no
`multi_buffer` mark is produced, `PlanMemory` does not create additional physical
buffers, `EnableMultiBuffer` has nothing to expand, and intra-core
synchronization observes a single buffer.

The limiting factor is therefore the **placement of the alloc**, not the number
of buffers allocated.

## 4. Configuration C — Adopted Approach

ssbuffer remains enabled and retains its responsibilities: splitting work across
cores, forming compute blocks, and inserting cross-core synchronization. One
responsibility is delegated: rotation of the store-out buffers is performed by
NPU-IR.

Two modifications are required:

1. **Relocation of the alloc into the main loop**, so that `getParentLoop`
   returns a non-null loop and the pattern proceeds to marking. Currently applied
   manually.
2. **Explicit multi-buffer mark** on that alloc — `annotation.mark` with
   `hivm.multi_buffer = N`. This is the same mark `hivm-mark-multi-buffer` would
   produce; supplying it directly removes the dependency on automatic detection.

The remainder of the flow is the standard NPU-IR path and requires no additional
work:

```
annotation.mark {hivm.multi_buffer = N}
        │
        ▼  PlanMemory        : allocates N physical buffers;
        │                      emits one pointer_cast carrying N addresses
        ▼  EnableMultiBuffer : counter mod N + arith.select;
        │                      the operation stays single, its address becomes dynamic
        ▼  intra-core sync   : observes N independent buffers, assigns separate
                               event ids, and no longer serializes iterations
```

The final step removes the lock-step: consecutive iterations write to different
slots, so the loop-carried WAR dependency no longer exists.

## 5. Results

Store-out of iteration `i` overlaps with computation of iteration `i+1`, and the
src→dst wait leaves the critical path.

> **[Trace 2 — configuration C: store-out overlapped with computation]**

| Configuration | Time *(unit: TBD)* | Gain |
|---|---|---|
| A — no multi-cache | 35 | — |
| C — N = 2 | 30 | 14% |
| C — N > 2 | 28 | 20% |

## 6. Remaining Work

The alloc relocation must be automated as a dedicated pass before productization.
The principal constraint is already identified: a buffer relocated into the loop
must become **loop-carried**, i.e. threaded through `iter_args`, otherwise it is
re-created on every iteration and the rotation loses its anchor. This is
consistent with `getParentLoopImpl`, which resolves the parent loop by following
the chain of yielded values.

Two interactions the pass must account for:

- `HoistTightlyCoupledAlloc` performs the inverse transformation, hoisting
  tightly-coupled allocs out of inner regions. The new pass must either be
  restricted to store-out buffers that are not tightly-coupled, or be ordered
  unambiguously with respect to it.
- ssbuffer hoists allocs above the loop by design; its own block assignment
  relies on that placement. Relocation must preserve the validity of that
  assignment.

## 7. Limitations

- Validated on one kernel class in which vector-side store-out is the
  bottleneck. Portability to other shapes is not measured.
- UB consumption increases by a factor of N for the relocated buffers. On denser
  kernels the buffers may not fit, in which case `UBOverflowChecker` removes the
  marks or UB overflow is reported.
- The manual modification is not reproducible in CI until the pass described in
  Section 6 is available.
- Measurement units in Section 5 are to be confirmed; run-to-run variance should
  be added.

## 8. Rationale

ssbuffer is neither disabled nor bypassed. It continues to perform the tasks for
which it is well suited. A single task — rotation of store-out buffers — is
delegated to the pass designed for it, which additionally has information not
available at the ssbuffer level: the physical buffer addresses assigned after
memory planning.

The change is incremental, consisting of one alloc relocation and one mark. The
existing path is unaffected, the change is trivially revertible, and the benefit
is measurable: 35 → 28.
