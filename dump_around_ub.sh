#!/bin/bash
# IR right before and right after ub-usage-opt, plus what it did to the block ids.
#
# Safe as a two-step run because ub-usage-opt sits inside ComputeBlockOpt, well
# before inter-core-transfer-and-sync -- that one misbehaves when the pipeline
# is split across separate triton-opt processes, these do not.
#
# Usage:
#   ./dump_around_ub.sh kernel.ttir.mlir [outdir]
#
# kernel.ttir.mlir comes from a run with TRITON_KERNEL_DUMP=1 (or from
# AOTcompile.py, which writes it directly).

set -u
TTIR="${1:?usage: $0 kernel.ttir.mlir [outdir]}"
OUT="${2:-./ub_dump}"
OPT="${TRITON_OPT:-$(python3 -c 'import triton, os; print(os.path.join(os.path.dirname(triton.__file__), "_C", "triton-opt"))')}"
[ -x "$OPT" ] || { echo "triton-opt not found at $OPT (set TRITON_OPT)"; exit 1; }
mkdir -p "$OUT"

FRONT='triton-control-flow-opt,triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-mode=simd_simt_template compile-on-910-95=true},triton-to-annotation,triton-to-unstructure{compile-mode=simd_simt_template compile-on-910-95=true force-scalarize-mode=false},triton-to-hivm,triton-to-hfusion,triton-to-llvm,bubble-up-operation{enable-aggressive-mode=true},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},triton-to-linalg{compile-mode=simd_simt_template compile-on-910-95=true enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true},merge-concat-load-buffer'

# Everything ComputeBlockOpt runs ahead of UBUsageOpt, in its own order
# (ComputeBlockOptPass.cpp), prefixed by PreCheckAvailable / StandardizeOp /
# PlanComputeBlock.
BEFORE_LIST=(
  pre-check-dynamic-cv-pipeline-available
  ssbuf-standardize-op
  op-classifier
  plan-cube-block
  plan-vector-block
  reorder-ops-by-block-id
  unify-alloc-block
  reorder-ops-by-block-id
  merge-vector-if-block
  reorder-ops-by-block-id
  reorder-ops-by-block-id
  exp-load-pattern
  reorder-ops-by-block-id
)

# Pass names drift between builds, so drop anything this binary does not know
# rather than failing the whole pipeline on one unknown name.
HELP="$("$OPT" --help 2>&1)"
KEEP=()
for p in "${BEFORE_LIST[@]}"; do
  if printf '%s' "$HELP" | grep -q -- "--$p"; then
    KEEP+=("$p")
  else
    echo "note: this build has no pass '$p', skipping it"
  fi
done
BEFORE=$(IFS=,; echo "${KEEP[*]}")

"$OPT" "$TTIR" "--pass-pipeline=any($FRONT,$BEFORE)" -o "$OUT/before_ub.mlir" 2> "$OUT/before.err"
[ -s "$OUT/before_ub.mlir" ] || { echo "the prefix failed, see $OUT/before.err"; head -5 "$OUT/before.err"; exit 1; }

# The pass prints its decisions on stdout ("maybe need opt", "Change ... TO n",
# "Find cycle ..."), so keep that next to the IR.
"$OPT" "$OUT/before_ub.mlir" "--pass-pipeline=any(ub-usage-opt)" -o "$OUT/after_ub.mlir" \
  > "$OUT/pass_stdout.txt" 2> "$OUT/after.err"
[ -s "$OUT/after_ub.mlir" ] || { echo "ub-usage-opt failed, see $OUT/after.err"; head -5 "$OUT/after.err"; exit 1; }

ids () { grep -o 'ssbuffer\.block_id = [0-9]*' "$1" | grep -o '[0-9]*$' | sort -n -u | tr '\n' ' '; }
vec_ids () {
  grep 'ssbuffer\.core_type = "VECTOR"' "$1" | grep -o 'ssbuffer\.block_id = [0-9]*' \
    | grep -o '[0-9]*$' | sort -n -u | tr '\n' ' '
}
count_per_block () {
  grep -o 'ssbuffer\.block_id = [0-9]*' "$1" | grep -o '[0-9]*$' | sort -n | uniq -c \
    | awk '{printf "%s:%s ", $2, $1}'
}

echo
echo "before ub-usage-opt:"
echo "  all block ids   : $(ids "$OUT/before_ub.mlir")"
echo "  vector block ids: $(vec_ids "$OUT/before_ub.mlir")"
echo "  ops per block   : $(count_per_block "$OUT/before_ub.mlir")"
echo "after ub-usage-opt:"
echo "  all block ids   : $(ids "$OUT/after_ub.mlir")"
echo "  vector block ids: $(vec_ids "$OUT/after_ub.mlir")"
echo "  ops per block   : $(count_per_block "$OUT/after_ub.mlir")"
echo
echo "what the pass said: $OUT/pass_stdout.txt"
echo "line diff of the IR: $(diff "$OUT/before_ub.mlir" "$OUT/after_ub.mlir" | wc -l) lines"
echo "files: $OUT/before_ub.mlir $OUT/after_ub.mlir"
