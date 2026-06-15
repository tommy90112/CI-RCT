#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Tracer-algorithm ablation on Elliptic++ (joint).
#
# 1:1 with the baseline eval command — ONLY --tracer_algorithm varies (and, for
# the two greedy arms, --prefer_reachable_depth to show the lookahead patch's
# effect). All arms share the SAME checkpoint, graph, CE scores, threshold,
# max_hops and --prefer_root_types, so any metric difference is attributable to
# the SEARCH ALGORITHM alone. See ../tracer_ablation_plan.md §4.
#
# Usage (run from the CI-RCT/ directory):
#     bash scripts/run_tracer_ablation.sh
#     MAX_EXPLAIN=300 bash scripts/run_tracer_ablation.sh   # fast first pass
#     CKPT=path/to.pt bash scripts/run_tracer_ablation.sh
#
# NOTE: the global-optimal arms (dag_dp/dijkstra) explore the whole ancestor
# cone, not a single path like greedy. With --max_hops 20, --ce_threshold 1e-4
# and --include_addr_addr the cone can be large, so do a MAX_EXPLAIN=300 smoke
# run FIRST to gauge per-arm runtime before committing to the full 2000.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CKPT="${CKPT:-checkpoints/ci_rct_elliptic++_joint_best.pt}"
MAX_EXPLAIN="${MAX_EXPLAIN:-2000}"
OUTDIR="${OUTDIR:-logs/elliptic/ablation}"
# evaluate.py defaults --device to cpu; the model forward over the full
# 822k-node / 2.87M-edge graph runs 7x here (once per arm), so default to GPU.
# Override with DEVICE=cpu for a no-GPU box.
DEVICE="${DEVICE:-cuda}"
mkdir -p "$OUTDIR"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "Set CKPT=... or restore ci_rct_elliptic++_joint_best.pt into checkpoints/." >&2
  exit 1
fi

COMMON=(
  --variant joint --dataset elliptic++ --data_root data
  --checkpoint "$CKPT"
  --include_addr_addr true --num_hgt_layers 3
  --max_explain "$MAX_EXPLAIN" --max_hops 20 --node_limit 1000000 --ce_threshold 0.0001
  --threshold_tuning val --threshold_objective fraud_f1
  --prefer_root_types wallet
  --device "$DEVICE"
  --lfpn_mode both --debug
)

run() {  # $1 = arm name; rest = extra args
  local name="$1"; shift
  local log="$OUTDIR/eval_${name}.log"
  echo ">>> [$name] -> $log"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES=0 python evaluate.py "${COMMON[@]}" "$@" 2>&1 | tee "$log"
  echo "WALLCLOCK_SECONDS=$((SECONDS - t0))" | tee -a "$log"
}

# ── Arms ──────────────────────────────────────────────────────────────────────
# greedy WITHOUT lookahead = the "before" state (the dd18 dead-end risk).
run greedy_nolookahead --tracer_algorithm greedy   --prefer_reachable_depth 0
# greedy WITH lookahead = current patched baseline (matches your existing log).
run greedy_lookahead   --tracer_algorithm greedy   --prefer_reachable_depth 3
# Proposed main method — global-optimal, NO lookahead patch.
run dag_dp_product     --tracer_algorithm dag_dp    --tracer_objective product --prefer_reachable_depth 0
run dag_dp_sum         --tracer_algorithm dag_dp    --tracer_objective sum     --prefer_reachable_depth 0
# Strong comparison: same optimum as dag_dp, validates correctness.
run dijkstra           --tracer_algorithm dijkstra  --prefer_reachable_depth 0
# Weight-blind lower bounds.
run bfs                --tracer_algorithm bfs        --prefer_reachable_depth 0
run dfs                --tracer_algorithm dfs        --prefer_reachable_depth 0

echo
echo "All arms done. Summarise with:"
echo "    python scripts/parse_tracer_ablation.py $OUTDIR"
