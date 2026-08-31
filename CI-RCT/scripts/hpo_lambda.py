#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# 自動超參數搜尋（Optuna / TPE）for joint-loss 權重 λ1, λ2, λ3。
#
# 你不給固定數字，只給「範圍」；Optuna 依前幾組結果推斷下一組該試哪裡。
#
# 目標函數（你選的：F1 門檻 + 最大化 RC）：
#     val_pooled_F1 < gate(0.80)  →  回傳 (F1 - gate)   # 負值；越接近門檻排越前
#     否則                        →  回傳 RC_prec + RC_hit − α·φ_std   # 越大越好
#
# 成本控制（proxy）：
#     搜尋階段每個 trial 只訓練 --proxy_epochs（預設 120）、評估 --proxy_max_explain
#     （預設 800），全部在 VAL 上算（--eval_split val），test 完全不碰。
#     找到贏家後，對贏家做一次 --final_epochs（預設 400）全量重訓，並在 TEST 上
#     報一次正式數字（--eval_split test）。
#
# 方法學保證：
#     * 搜尋全程只用 val；test 只在最後贏家報一次 → 無 test 洩漏。
#     * 每個 trial 的 checkpoint 存到獨立資料夾，互不覆蓋。
#     * study 存進 SQLite，中斷可續跑（重跑同一指令即接續）。
#
# 用法（在 CI-RCT/ 目錄下執行）：
#     pip install optuna
#     python scripts/hpo_lambda.py --n_trials 25 --device cuda
#     # 先試水溫（確認串得起來，每個 trial 幾分鐘）：
#     python scripts/hpo_lambda.py --n_trials 3 --proxy_epochs 15 \
#         --proxy_max_explain 200 --device cuda
#     # 只搜尋、先不要自動重訓贏家：
#     python scripts/hpo_lambda.py --n_trials 25 --no_final
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import re
import subprocess
import sys

try:
    import optuna
except ImportError:
    sys.exit("缺少 optuna，請先 `pip install optuna` 再執行。")


# ── 從 evaluate.py 的輸出 log 解析指標 ───────────────────────────────────────
_PATTERNS = {
    "pooled_f1": re.compile(r"^\s*F1\s+:\s*([0-9.]+)", re.M),
    "fraud_f1": re.compile(r"^\s*Fraud F1\s+:\s*([0-9.]+)", re.M),
    "auc": re.compile(r"^\s*Auc\s+:\s*([0-9.]+)", re.M),
    "rc_precision": re.compile(r"Root Cause Precision\s+:\s*([0-9.]+)", re.M),
    "rc_hit": re.compile(r"Root Cause Hit Rate\s+:\s*([0-9.]+)", re.M),
    "phi_std": re.compile(r"Phi Stability Std\s+:\s*([0-9.]+)", re.M),
}


def parse_metrics(text):
    """把 evaluate.py 的整段輸出解析成 {metric: float}。缺項回傳 None。"""
    out = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(text)
        out[key] = float(m.group(1)) if m else None
    return out


def run_logged(cmd, log_path, env):
    """執行子程序，輸出同時寫進 log 檔；回傳 (returncode, 完整輸出字串)。"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    chunks = []
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        for line in proc.stdout:
            f.write(line)
            chunks.append(line)
        proc.wait()
    return proc.returncode, "".join(chunks)


def build_train_cmd(args, l1, l2, l3, ckpt_dir, epochs):
    return [
        sys.executable, "train.py",
        "--variant", "joint", "--dataset", "elliptic++",
        "--data_root", args.data_root,
        "--epochs", str(epochs), "--use_gan", "true",
        "--include_addr_addr", "true",
        "--hidden_dim", "128", "--num_hgt_layers", "3",
        "--lambda_adversarial", f"{l1:.6g}",
        "--lambda_stability", f"{l2:.6g}",
        "--lambda_ncm", f"{l3:.6g}",
        "--use_reconstruction", "false",
        "--symmetric_joint", "true", "--lambda_aux_detection", "1.0",
        "--checkpoint_dir", ckpt_dir,
        "--device", args.device,
    ]


def build_eval_cmd(args, ckpt, max_explain, eval_split):
    return [
        sys.executable, "evaluate.py",
        "--variant", "joint", "--dataset", "elliptic++",
        "--data_root", args.data_root,
        "--checkpoint", ckpt,
        "--include_addr_addr", "true", "--num_hgt_layers", "3",
        "--max_explain", str(max_explain), "--max_hops", "20",
        "--node_limit", "1000000", "--ce_threshold", "0.0001",
        "--threshold_tuning", "val", "--threshold_objective", "fraud_f1",
        "--tracer_algorithm", "greedy", "--prefer_reachable_depth", "3",
        "--prefer_root_types", "wallet",
        "--lfpn_mode", "strict",        # 搜尋只需 Metric A/B/D；strict 比 both 快
        "--eval_split", eval_split,
        "--device", args.device,
        "--debug",
    ]


def objective_factory(args, env):
    def objective(trial):
        # ── 搜尋空間（log-uniform，因為三者跨越一個數量級）──────────────────
        l1 = trial.suggest_float("lambda_adversarial", 0.01, 1.0, log=True)
        l2 = trial.suggest_float("lambda_stability", 0.05, 2.0, log=True)
        l3 = trial.suggest_float("lambda_ncm", 0.05, 1.0, log=True)

        tag = f"trial{trial.number:03d}"
        ckpt_dir = os.path.join(args.ckpt_root, tag)
        ckpt = os.path.join(ckpt_dir, "ci_rct_elliptic++_joint_best.pt")
        train_log = os.path.join(args.log_dir, f"train_{tag}.log")
        eval_log = os.path.join(args.log_dir, f"eval_{tag}.log")

        # ── 訓練（proxy epochs）─────────────────────────────────────────────
        if not os.path.isfile(ckpt):
            rc, _ = run_logged(
                build_train_cmd(args, l1, l2, l3, ckpt_dir, args.proxy_epochs),
                train_log, env,
            )
            if rc != 0 or not os.path.isfile(ckpt):
                trial.set_user_attr("status", "train_failed")
                print(f"  [{tag}] 訓練失敗 (rc={rc})，跳過。見 {train_log}")
                return -10.0

        # ── 評估（在 VAL 上）────────────────────────────────────────────────
        rc, out = run_logged(
            build_eval_cmd(args, ckpt, args.proxy_max_explain, "val"),
            eval_log, env,
        )
        if rc != 0:
            trial.set_user_attr("status", "eval_failed")
            print(f"  [{tag}] 評估失敗 (rc={rc})，跳過。見 {eval_log}")
            return -10.0

        m = parse_metrics(out)
        for k, v in m.items():
            trial.set_user_attr(k, v)

        f1 = m["pooled_f1"]
        rcp, rch, phi = m["rc_precision"], m["rc_hit"], m["phi_std"]
        if f1 is None or rcp is None or rch is None:
            trial.set_user_attr("status", "parse_failed")
            print(f"  [{tag}] 指標解析失敗，跳過。見 {eval_log}")
            return -10.0

        phi = phi if phi is not None else 0.0
        # ── 目標：F1 門檻 + 最大化 RC ────────────────────────────────────────
        if f1 < args.f1_gate:
            score = f1 - args.f1_gate          # 負；越接近門檻排越前
            status = "below_gate"
        else:
            score = rcp + rch - args.alpha_phi * phi
            status = "ok"
        trial.set_user_attr("status", status)
        trial.set_user_attr("score", score)
        print(f"  [{tag}] λ=({l1:.3g},{l2:.3g},{l3:.3g}) "
              f"valF1={f1:.4f} RCp={rcp:.4f} RCh={rch:.4f} "
              f"φstd={phi:.4f} → score={score:.4f} ({status})")
        return score

    return objective


def retrain_winner(args, env, best):
    l1 = best.params["lambda_adversarial"]
    l2 = best.params["lambda_stability"]
    l3 = best.params["lambda_ncm"]
    ckpt_dir = os.path.join(args.ckpt_root, "winner_full")
    ckpt = os.path.join(ckpt_dir, "ci_rct_elliptic++_joint_best.pt")
    train_log = os.path.join(args.log_dir, "train_winner_full.log")
    eval_log = os.path.join(args.log_dir, "eval_winner_full_test.log")

    print(f"\n>>> 全量重訓贏家 λ=({l1:.4g},{l2:.4g},{l3:.4g}) "
          f"@ {args.final_epochs} epochs → {train_log}")
    rc, _ = run_logged(
        build_train_cmd(args, l1, l2, l3, ckpt_dir, args.final_epochs),
        train_log, env,
    )
    if rc != 0 or not os.path.isfile(ckpt):
        print(f"!!! 贏家全量重訓失敗 (rc={rc})，見 {train_log}")
        return

    print(f">>> 贏家在 TEST 上正式評估 → {eval_log}")
    rc, out = run_logged(
        build_eval_cmd(args, ckpt, args.final_max_explain, "test"),
        eval_log, env,
    )
    m = parse_metrics(out)
    print("\n========= 贏家 TEST 指標（論文要報這組）=========")
    print(f"  λ1={l1:.4g}  λ2={l2:.4g}  λ3={l3:.4g}")
    print(f"  pooled F1 = {m.get('pooled_f1')}  fraud F1 = {m.get('fraud_f1')}  "
          f"AUC = {m.get('auc')}")
    print(f"  RC precision = {m.get('rc_precision')}  "
          f"RC hit = {m.get('rc_hit')}  φ-std = {m.get('phi_std')}")
    print(f"  完整 log：{eval_log}")


def main():
    p = argparse.ArgumentParser(description="Optuna HPO for joint-loss λ weights")
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--proxy_epochs", type=int, default=200)
    p.add_argument("--proxy_max_explain", type=int, default=800)
    p.add_argument("--final_epochs", type=int, default=400)
    p.add_argument("--final_max_explain", type=int, default=2000)
    p.add_argument("--f1_gate", type=float, default=0.80,
                   help="pooled F1 門檻；低於此值的配置被罰。")
    p.add_argument("--alpha_phi", type=float, default=1.0,
                   help="φ-stability std 在目標中的權重（次要 tie-breaker）。")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--ckpt_root", type=str, default="checkpoints/hpo")
    p.add_argument("--log_dir", type=str, default="logs/elliptic/hpo")
    p.add_argument("--study_name", type=str, default="lambda_hpo")
    p.add_argument("--storage", type=str,
                   default="sqlite:///logs/elliptic/hpo/optuna.db")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_final", action="store_true",
                   help="只搜尋，不自動全量重訓贏家。")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_root, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,        # 中斷可續跑
    )
    study.optimize(objective_factory(args, env), n_trials=args.n_trials)

    print(f"\n完成 {len(study.trials)} trials。")
    best = study.best_trial
    print("========= 最佳配置（依 val 目標）=========")
    print(f"  λ1={best.params['lambda_adversarial']:.4g}  "
          f"λ2={best.params['lambda_stability']:.4g}  "
          f"λ3={best.params['lambda_ncm']:.4g}")
    print(f"  val score = {best.value:.4f}  "
          f"(valF1={best.user_attrs.get('pooled_f1')}, "
          f"RCp={best.user_attrs.get('rc_precision')}, "
          f"RCh={best.user_attrs.get('rc_hit')})")

    # 全部 trial 匯出成 CSV，方便事後比較。
    csv_path = os.path.join(args.log_dir, "hpo_trials.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"  全部 trial → {csv_path}")

    if not args.no_final:
        retrain_winner(args, env, best)


if __name__ == "__main__":
    main()
