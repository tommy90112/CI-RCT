#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# φ perturbation-stability probe (strengthened Metric D) across Full + 3 loss arms.
#
# The legacy Phi-Stability metric saturates at σ=0.01 and cannot distinguish an
# L_stab-trained model from the −L_stab ablation. This probe sweeps σ, averages
# K draws, and reports drift at each chain's pivot — so L_stab's effect should
# show as Full staying flatter than no_stab as σ grows.
#
# Fast: skips LFPN + 2000-chain tracing (--stability_probe_only). Minutes/ckpt.
#
# Usage (from CI-RCT/, on the server GPU box):
#     GPU=1 bash scripts/run_phi_stability_probe.sh
#     GPU=1 MAX_EXPLAIN=2000 bash scripts/run_phi_stability_probe.sh   # fuller pass
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FULL_CKPT="${FULL_CKPT:-checkpoints_txwallet_fix/ci_rct_elliptic++_joint_best.pt}"
ABL_ROOT="${ABL_ROOT:-checkpoints_abl}"
OUTDIR="${OUTDIR:-logs/elliptic/phi_probe}"
DEVICE="${DEVICE:-cuda}"
MAX_EXPLAIN="${MAX_EXPLAIN:-500}"
SWEEP="${SWEEP:-0.01,0.05,0.1,0.2,0.5}"
DRAWS="${DRAWS:-5}"
mkdir -p "$OUTDIR"

COMMON=(
  --variant joint --dataset elliptic++ --data_root data
  --include_addr_addr true --num_hgt_layers 3
  --node_limit 1000000 --ce_threshold 0.0001
  --ncm_baseline marginal --device "$DEVICE"
  --stability_probe_only --max_explain "$MAX_EXPLAIN"
  --phi_noise_sweep "$SWEEP" --phi_noise_draws "$DRAWS"
)

probe() {  # $1 = name; $2 = checkpoint path
  local name="$1" ckpt="$2"
  local log="$OUTDIR/probe_${name}.log"
  if [[ ! -f "$ckpt" ]]; then
    echo "WARN: [$name] checkpoint not found, skipping: $ckpt" >&2; return
  fi
  echo ">>> [$name] φ-probe -> $log"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u evaluate.py "${COMMON[@]}" \
    --checkpoint "$ckpt" 2>&1 | tee "$log"
}

probe full    "$FULL_CKPT"
probe no_ncm  "$ABL_ROOT/no_ncm/ci_rct_elliptic++_joint_best.pt"
probe no_stab "$ABL_ROOT/no_stab/ci_rct_elliptic++_joint_best.pt"
probe no_gan  "$ABL_ROOT/no_gan/ci_rct_elliptic++_joint_best.pt"

echo
echo "Done. Compare the 'pivot rel-drift' column across:"
echo "  grep -A8 'φ-stability probe' $OUTDIR/probe_{full,no_stab}.log"
echo "Expectation: Full stays FLATTER than no_stab as σ grows (L_stab works)."
