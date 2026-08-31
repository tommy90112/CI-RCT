#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Joint-loss weight sweep on Elliptic++ (variant=joint).
#
# 目標：在 pooled F1 ≥ 0.80 的前提下，最大化 root-cause tracing（Metric B：
#       Root Cause Precision / Hit Rate）與 φ-stability（Metric D）。
#
# 方法：coordinate search（繞著 baseline 各軸單獨掃一個參數），共 7 組：
#       baseline = λ1(adv)=0.1  λ2(stab)=0.5  λ3(ncm)=0.3
#         · λ1 ∈ {0.05, 0.1, 0.5}
#         · λ2 ∈ {0.1, 0.5, 1.0}
#         · λ3 ∈ {0.1, 0.3, 0.5}
#       每組 = 1 次全量重訓（400 epochs, GAN）+ 1 次評估。
#
# 重要方法學：
#   * 選模 / 選參數一律用 VALIDATION（train.py 以 val fraud_f1 選 checkpoint；
#     evaluate.py 以 --threshold_tuning val 調門檻）。test 只在最後報一次，
#     不可拿 test 指標挑贏家（避免對 test 過擬合）。
#   * 每組 checkpoint 存到獨立 --checkpoint_dir，避免互相覆蓋。
#   * 評估刻意不帶 --dump_phi / --dump_chains（那是給前端用的慢速匯出）；
#     Metric A/B/D 仍會完整輸出。
#
# 用法（在 CI-RCT/ 目錄下執行）：
#     bash scripts/run_lambda_sweep.sh
#     EPOCHS=400 DEVICE=cuda bash scripts/run_lambda_sweep.sh
#     # 先快速試水溫（少量 epoch + 少量 chain），確認指令通再跑整晚：
#     EPOCHS=20 MAX_EXPLAIN=300 bash scripts/run_lambda_sweep.sh
#
# 中斷可續跑：已存在 checkpoint 的組會跳過訓練，只重評估。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 可由環境變數覆蓋的旋鈕 ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-400}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
MAX_EXPLAIN="${MAX_EXPLAIN:-2000}"     # 評估時追蹤的 chain 數（與 baseline 一致）
DATA_ROOT="${DATA_ROOT:-data}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/lambda_sweep}"
LOG_DIR="${LOG_DIR:-logs/elliptic/lambda_sweep}"
SUMMARY="${SUMMARY:-$LOG_DIR/results_summary.md}"

mkdir -p "$CKPT_ROOT" "$LOG_DIR"

# ── 掃描配置：tag  λ_adv  λ_stab  λ_ncm ─────────────────────────────────────
# baseline 置頂；其餘每行只動一個參數（coordinate search）。
CONFIGS=(
  "baseline   0.1   0.5   0.3"   # ← 中心點，三軸共用
  "adv0.05    0.05  0.5   0.3"   # λ1 ↓
  "adv0.5     0.5   0.5   0.3"   # λ1 ↑
  "stab0.1    0.1   0.1   0.3"   # λ2 ↓
  "stab1.0    0.1   1.0   0.3"   # λ2 ↑
  "ncm0.1     0.1   0.5   0.1"   # λ3 ↓
  "ncm0.5     0.1   0.5   0.5"   # λ3 ↑
)

# ── 訓練（除非 checkpoint 已存在）─────────────────────────────────────────────
train_one() {
  local tag="$1" l1="$2" l2="$3" l3="$4"
  local ckpt_dir="$CKPT_ROOT/$tag"
  local ckpt="$ckpt_dir/ci_rct_elliptic++_joint_best.pt"
  local log="$LOG_DIR/train_${tag}.log"
  mkdir -p "$ckpt_dir"

  if [[ -f "$ckpt" ]]; then
    echo ">>> [train:$tag] checkpoint 已存在，跳過訓練：$ckpt"
    return 0
  fi

  echo ">>> [train:$tag] λ1=$l1 λ2=$l2 λ3=$l3 → $log"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --variant joint --dataset elliptic++ --data_root "$DATA_ROOT" \
    --epochs "$EPOCHS" --use_gan true \
    --include_addr_addr true --hidden_dim 128 --num_hgt_layers 3 \
    --lambda_adversarial "$l1" --lambda_stability "$l2" --lambda_ncm "$l3" \
    --use_reconstruction false \
    --symmetric_joint true --lambda_aux_detection 1.0 \
    --checkpoint_dir "$ckpt_dir" \
    --device "$DEVICE" 2>&1 | tee "$log"
  echo "WALLCLOCK_SECONDS=$((SECONDS - t0))" | tee -a "$log"
}

# ── 評估（tracer = greedy + lookahead，與 baseline dump log 1:1）──────────────
eval_one() {
  local tag="$1"
  local ckpt="$CKPT_ROOT/$tag/ci_rct_elliptic++_joint_best.pt"
  local log="$LOG_DIR/eval_${tag}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "!!! [eval:$tag] 找不到 checkpoint，跳過：$ckpt" >&2
    return 0
  fi

  echo ">>> [eval:$tag] → $log"
  local t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" python evaluate.py \
    --variant joint --dataset elliptic++ --data_root "$DATA_ROOT" \
    --checkpoint "$ckpt" \
    --include_addr_addr true --num_hgt_layers 3 \
    --max_explain "$MAX_EXPLAIN" --max_hops 20 --node_limit 1000000 --ce_threshold 0.0001 \
    --threshold_tuning val --threshold_objective fraud_f1 \
    --tracer_algorithm greedy --prefer_reachable_depth 3 \
    --prefer_root_types wallet \
    --lfpn_mode both --debug \
    --device "$DEVICE" 2>&1 | tee "$log"
  echo "WALLCLOCK_SECONDS=$((SECONDS - t0))" | tee -a "$log"
}

# ── 跑全部 ───────────────────────────────────────────────────────────────────
for row in "${CONFIGS[@]}"; do
  read -r tag l1 l2 l3 <<< "$row"
  train_one "$tag" "$l1" "$l2" "$l3"
  eval_one  "$tag"
done

# ── 彙總成表（從各 eval log 抓關鍵指標）─────────────────────────────────────
echo
echo ">>> 彙總 → $SUMMARY"
{
  echo "# Lambda Sweep 結果彙總"
  echo
  echo "判準：pooled F1 ≥ 0.80 前提下，最大化 Root-Cause Precision / Hit + φ-stability。"
  echo "（選參數看的是這張表，但正式論文數字 = 贏家那組的 test 指標，只報一次。）"
  echo
  echo "| tag | λ1 | λ2 | λ3 | pooled F1 | fraud F1 | AUC | RC Prec | RC Hit | Chain Valid | φ-stab std |"
  echo "|-----|----|----|----|-----------|----------|-----|---------|--------|-------------|------------|"
  for row in "${CONFIGS[@]}"; do
    read -r tag l1 l2 l3 <<< "$row"
    log="$LOG_DIR/eval_${tag}.log"
    [[ -f "$log" ]] || { echo "| $tag | $l1 | $l2 | $l3 | (no log) |||||||"; continue; }
    pf1=$(grep -m1 -E "^  F1 " "$log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    ff1=$(grep -m1 -E "^  Fraud F1" "$log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    auc=$(grep -m1 -E "^  Auc" "$log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    rcp=$(grep -m1 -E "Root Cause Precision  " "$log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    rch=$(grep -m1 -E "Root Cause Hit Rate" "$log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    cv=$(grep -m1 -E "Chain Validity" "$log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    phi=$(grep -m1 -E "Phi Stability Std" "$log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "| $tag | $l1 | $l2 | $l3 | ${pf1:-—} | ${ff1:-—} | ${auc:-—} | ${rcp:-—} | ${rch:-—} | ${cv:-—} | ${phi:-—} |"
  done
} | tee "$SUMMARY"

echo
echo "完成。彙總表：$SUMMARY"
echo "挑贏家：先看 pooled F1 那欄 ≥ 0.80，再在符合者中比 RC Prec / RC Hit（越高越好）與 φ-stab std（越小越好）。"
