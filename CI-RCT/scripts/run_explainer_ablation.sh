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
# Arms (all already wired in model/explainers.py). Block-B design = isolate one
# attribution ingredient per arm, all sharing the SAME readout/target/field:
#   saliency  — Grad×Input, NO intervention (correlational)    → value of CAUSALITY
#   ce_only   — raw |CE| greedy chain (no Shapley)            → value of Shapley
#   phi_sym   — symmetric Causal Shapley                       → baseline for asym
#   phi_asym  — MAIN: asymmetric Causal Shapley (do-interv.)   → our method
# The phi_asym − saliency gap is the headline "intervention beats correlation"
# result; phi_asym − ce_only isolates Shapley; phi_asym − phi_sym isolates
# temporal asymmetry.
#
# Usage (from CI-RCT/):
#     bash scripts/run_explainer_ablation.sh
#     MAX_EXPLAIN=300 bash scripts/run_explainer_ablation.sh   # fast first pass
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CKPT="${CKPT:-checkpoints_txwallet_fix/ci_rct_elliptic++_joint_best.pt}"
MAX_EXPLAIN="${MAX_EXPLAIN:-2000}"
OUTDIR="${OUTDIR:-logs/elliptic/explainer_ablation}"
DEVICE="${DEVICE:-cuda}"
# Each Shapley coalition is a FULL backbone forward, so uncapped high-in-degree
# nodes make phi_asym/phi_sym intractable (a single 300-target phi_sym ran >16h).
# Cap parents to top-k by |CE| and trim phi_sym's permutations. Override via env.
SHAPLEY_TOPK="${SHAPLEY_TOPK:-8}"
SHAPLEY_PERM="${SHAPLEY_PERM:-16}"
# Each Shapley coalition is a FULL-graph backbone forward (coalition_value.py),
# so symmetric Shapley (phi_sym, ~10x more coalitions than asym) stays expensive
# even with top-k. phi_sym is only the symmetric-vs-asymmetric ABLATION baseline,
# so run it on a small target subset; the main method phi_asym runs at full N.
PHI_SYM_MAX="${PHI_SYM_MAX:-50}"
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
  --prefer_root_types wallet --prefer_reachable_depth 3
  --ncm_baseline "${NCM_BASELINE:-marginal}" --tracer_score "${TRACER_SCORE:-ce_signed}"
  --device "$DEVICE"
  --shapley_topk "$SHAPLEY_TOPK" --shapley_permutations "$SHAPLEY_PERM"
  --lfpn_mode both --debug
)

run() {  # $1 = explainer name; rest = extra overrides (e.g. --max_explain N)
  local name="$1"; shift
  local log="$OUTDIR/eval_${name}.log"
  echo ">>> [explainer=$name] -> $log"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="${GPU:-0}" python evaluate.py "${COMMON[@]}" --explainer "$name" "$@" 2>&1 | tee "$log"
  echo "WALLCLOCK_SECONDS=$((SECONDS - t0))" | tee -a "$log"
}

# Main method + cheap arms first, at full N. saliency is one fwd+bwd per target
# (no coalitions), so it runs at full N as fast as ce_only.
run saliency
run ce_only
run phi_asym
# Symmetric ablation baseline LAST, on a small subset (expensive). NOTE: for the
# phi_asym-vs-phi_sym comparison, re-run phi_asym at the same PHI_SYM_MAX so both
# are scored on identical targets.
run phi_sym --max_explain "$PHI_SYM_MAX"

echo
echo "Summarise with:  python scripts/parse_explainer_ablation.py $OUTDIR"
