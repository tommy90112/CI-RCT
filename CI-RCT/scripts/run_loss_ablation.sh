#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Loss-term (module-NECESSITY) ablation on Elliptic++ (joint).  See ablation_plan.md §A.
#
# Each arm RETRAINS from scratch and changes EXACTLY ONE of three loss knobs vs
# the frozen Full config (ablation_plan.md §0).  All other hyper-parameters are
# byte-identical, so any metric change is attributable to the removed module:
#
#   arm       changed knob            evaluate on   proves
#   ───────   ────────────────────    ───────────   ─────────────────────────────
#   no_ncm    --lambda_ncm 0.0        Metric B      M2b NCM 監督必要(CE 失判別→RCP 崩)
#   no_stab   --lambda_stability 0.0  Metric D      M2c 穩定損失必要(φ 相鄰步震盪)
#   no_gan    --use_gan false         Metric A      M4 GAN 偽裝增強對偵測有貢獻
#
# The three knobs are listed IN FULL on every arm (not just the changed one) so a
# reviewer can verify at a glance that only one differs from Full.
#
# ⚠️ TRAIN with --ncm_baseline type_mean ONLY.  marginal hangs during training
#    (per-step MLP over 822k wallets); it is an EVAL-time choice.  See handoff.
#
# Usage (run from the CI-RCT/ directory, on the server GPU box):
#     bash scripts/run_loss_ablation.sh                 # all 3 arms, sequential
#     ARM=no_ncm bash scripts/run_loss_ablation.sh      # just one arm
#     GPU=1 ARM=no_ncm bash scripts/run_loss_ablation.sh # pin to GPU 1 (A6000)
#     EPOCHS=400 bash scripts/run_loss_ablation.sh       # override epoch count
#
# Each arm is a full ~400-epoch train.  Prefer running them on separate GPUs in
# parallel (three shells: GPU=0/1/2 ARM=no_ncm/no_stab/no_gan) over sequential.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EPOCHS="${EPOCHS:-400}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints_abl}"
LOGDIR="${LOGDIR:-logs/elliptic}"
DEVICE="${DEVICE:-cuda}"
RUN_EVAL="${RUN_EVAL:-1}"   # 1 = auto-evaluate each arm after training; 0 = train only
mkdir -p "$LOGDIR"

# Frozen Full config (ablation_plan.md §0) MINUS the three ablated knobs, which
# each arm supplies explicitly below.
# ⚠️ TRAIN baseline MUST be type_mean — marginal hangs during training (per-step
#    MLP over 822k wallets).  marginal is used only at EVAL time (see EVAL_BASE).
BASE=(
  --variant joint --dataset elliptic++ --data_root data
  --epochs "$EPOCHS"
  --include_addr_addr true --hidden_dim 128 --num_hgt_layers 3
  --lambda_adversarial 0.1
  --use_reconstruction false --symmetric_joint true --lambda_aux_detection 1.0
  --ncm_baseline type_mean --ncm_edge_balance sqrt
  --device "$DEVICE"
)

# Frozen EVAL config (ablation_plan.md §0).  Numbers-only: NO --dump_* / --shapley_topk,
# since these arms are a quantitative comparison, not a viz/case-study export.
# marginal + ce_signed here is correct and matches the main result.
EVAL_BASE=(
  --variant joint --dataset elliptic++ --data_root data
  --include_addr_addr true --num_hgt_layers 3 --max_explain 2000 --max_hops 20
  --node_limit 1000000 --ce_threshold 0.0001 --threshold_tuning val
  --threshold_objective fraud_f1 --prefer_root_types wallet
  --prefer_reachable_depth 3 --lfpn_mode both --debug
  --ncm_baseline marginal --tracer_score ce_signed
  --device "$DEVICE"
)

run() {  # $1 = arm name; rest = the three knobs (lambda_ncm / lambda_stability / use_gan)
  local name="$1"; shift
  local ckpt_dir="$CKPT_ROOT/$name"
  local ckpt="$ckpt_dir/ci_rct_elliptic++_joint_best.pt"
  local log="$LOGDIR/abl_${name}.log"
  mkdir -p "$ckpt_dir"
  echo ">>> [$name] retrain -> ckpt=$ckpt_dir  log=$log"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u train.py "${BASE[@]}" "$@" \
    --checkpoint_dir "$ckpt_dir" 2>&1 | tee "$log"
  echo "WALLCLOCK_SECONDS=$((SECONDS - t0))" | tee -a "$log"
  echo ">>> [$name] train done. ckpt: $ckpt"

  if [[ "$RUN_EVAL" == "1" ]]; then
    local elog="$LOGDIR/eval_abl_${name}.log"
    if [[ ! -f "$ckpt" ]]; then
      echo "WARN: [$name] checkpoint not found, skipping eval: $ckpt" >&2
      return
    fi
    echo ">>> [$name] eval -> $elog"
    local e0=$SECONDS
    CUDA_VISIBLE_DEVICES="${GPU:-0}" python -u evaluate.py "${EVAL_BASE[@]}" \
      --checkpoint "$ckpt" 2>&1 | tee "$elog"
    echo "WALLCLOCK_SECONDS=$((SECONDS - e0))" | tee -a "$elog"
    echo ">>> [$name] eval done. log: $elog"
  fi
}

# ── Arms ─────────────────────────────────────────────────────────────────────
# Full (reference) values are: --lambda_ncm 0.3 --lambda_stability 0.5 --use_gan true
# Each arm changes EXACTLY ONE knob; the other two keep their Full value.
run_arm() {  # $1 = arm name
  case "$1" in
    no_ncm)  run no_ncm  --lambda_ncm 0.0 --lambda_stability 0.5 --use_gan true  ;;
    no_stab) run no_stab --lambda_ncm 0.3 --lambda_stability 0.0 --use_gan true  ;;
    no_gan)  run no_gan  --lambda_ncm 0.3 --lambda_stability 0.5 --use_gan false ;;
    *) echo "ERROR: unknown ARM=$1 (want: no_ncm|no_stab|no_gan)" >&2; exit 1 ;;
  esac
}

if [[ -n "${ARM:-}" ]]; then
  run_arm "$ARM"
else
  # Priority order per ablation_plan.md §C: no_ncm is the ⭐ strongest arm.
  run_arm no_ncm
  run_arm no_stab
  run_arm no_gan
fi

echo
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "All arms done (train + eval). Per-arm metrics in: $LOGDIR/eval_abl_{no_ncm,no_stab,no_gan}.log"
  echo "Compare against Full (checkpoints_txwallet_fix). Report RCP@depth>=1, not raw RCP."
else
  echo "All arms trained (RUN_EVAL=0). Evaluate later with the frozen EVAL block, e.g.:"
  echo "  RUN_EVAL=1 ARM=no_ncm bash scripts/run_loss_ablation.sh   # (re-runs train too)"
  echo "  or call evaluate.py directly on $CKPT_ROOT/<arm>/ci_rct_elliptic++_joint_best.pt"
fi
