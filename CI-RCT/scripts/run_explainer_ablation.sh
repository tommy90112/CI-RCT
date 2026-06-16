#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Explainability (Metric C) ablation on Elliptic++ (joint).
#
# Orthogonal to run_tracer_ablation.sh: there we fixed the explainer and varied
# the SEARCH algorithm; here we fix the (validated) greedy tracer and vary the
# EXPLANATION MECHANISM (--explainer). All arms share the SAME checkpoint, graph,
# CE scores, tracer settings and LFPN ground truth, so any Metric C (EA/ER)
# difference is attributable to the attribution mechanism alone.
#
# Arms (all already wired in model/explainers.py):
#   ce_only   — raw |CE| greedy chain (no Shapley)            → value of Shapley
#   phi_sym   — symmetric Causal Shapley                       → baseline for asym
#   phi_asym  — MAIN: asymmetric Causal Shapley (do-interv.)   → our method
#   cxgnn_ncm — CXGNN / GNN-NCM (ECCV 2024) external baseline  → vs SOTA causal
#
# Usage (from CI-RCT/):
#     bash scripts/run_explainer_ablation.sh
#     MAX_EXPLAIN=300 bash scripts/run_explainer_ablation.sh   # fast first pass
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CKPT="${CKPT:-checkpoints/ci_rct_elliptic++_joint_best.pt}"
MAX_EXPLAIN="${MAX_EXPLAIN:-2000}"
OUTDIR="${OUTDIR:-logs/elliptic/explainer_ablation}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$OUTDIR"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
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

run() {  # $1 = explainer name
  local name="$1"
  local log="$OUTDIR/eval_${name}.log"
  echo ">>> [explainer=$name] -> $log"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES=0 python evaluate.py "${COMMON[@]}" --explainer "$name" 2>&1 | tee "$log"
  echo "WALLCLOCK_SECONDS=$((SECONDS - t0))" | tee -a "$log"
}

run ce_only
run phi_sym
run phi_asym
run cxgnn_ncm

echo
echo "Summarise with:  python scripts/parse_explainer_ablation.py $OUTDIR"
