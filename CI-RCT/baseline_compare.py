"""
P0 diagnostic: feature-discriminability sanity check for MG24.

CI-RCT pilot training on UNSW-MG24 gives Test F1 ~0.98 / AUC ~1.000 even
under size-stratified hybrid split. This script tests the hypothesis that
the high score is driven by CICFlowMeter feature discriminability — not by
graph structure — by running classical ML baselines on the same flow_node
features and the same train/val/test split.

Expected outcome (which we want to confirm):
    LogReg/RF F1 within a few points of CI-RCT 0.98
    → high F1 is dataset-driven, not model-driven
    → paper should pivot its contribution claim to explainability metrics
      (RCP / CCV / Pearl ground-truth) rather than raw detection F1.

Optional --feature_ablation lets you remove "tool-fingerprint" columns
(Dst Port, Flow Duration, IAT statistics, packet counts) and re-train.
A large F1 drop confirms those features carry direct label signal.

Usage
─────
    python baseline_compare.py \
        --data_root data \
        --split_mode hybrid \
        --mg24_subsample_ddos 0.1

    # With feature ablation:
    python baseline_compare.py \
        --feature_ablation "Dst Port,Src Port,Flow Duration,Flow IAT Mean"

Reference: unsw_mg24_plan.md DD-8 (split strategy)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--data_root", default="data")
    p.add_argument("--split_mode", default="hybrid",
                   choices=("row", "by_file", "hybrid"))
    p.add_argument("--mg24_subsample_ddos", type=float, default=0.1)
    p.add_argument("--mg24_min_host_flows", type=int, default=5)
    p.add_argument("--mg24_prune_external", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature_ablation", type=str, default="",
                   help="Comma-separated CICFlowMeter feature names to drop "
                        "(before z-scoring). Empty = no ablation.")
    p.add_argument("--max_train_rows", type=int, default=0,
                   help="Subsample train rows for speed (0 = no subsample). "
                        "RandomForest on 900k rows is slow; 200k is enough.")
    p.add_argument("--skip_rf", action="store_true",
                   help="Skip RandomForest (much slower than LogReg).")
    return p.parse_args()


def _build_flows_with_ablation(
    flows_df,
    ablation_cols: List[str],
):
    """
    Recompute the flow feature matrix while dropping `ablation_cols`.

    Returns (X, dropped_cols_actually_present).
    """
    from utils.mg24_loader import _FLOW_NON_FEATURE_COLS, _log1p_zscore
    import pandas as pd

    non_features = set(_FLOW_NON_FEATURE_COLS) | set(ablation_cols)
    feature_cols = [
        c for c in flows_df.columns
        if c not in non_features
        and pd.api.types.is_numeric_dtype(flows_df[c])
    ]
    dropped_actual = [c for c in ablation_cols if c in flows_df.columns]
    arr = flows_df[feature_cols].astype(np.float32).values
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)
    arr = _log1p_zscore(arr)
    return arr, dropped_actual, feature_cols


def _score_split(
    name: str,
    clf,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> List[Tuple[str, str, float, float, float]]:
    """Predict and report F1 / AUC / AUPRC on val + test."""
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

    rows = []
    for split_name, X_, y_ in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = clf.predict(X_)
        try:
            y_score = clf.predict_proba(X_)[:, 1]
        except AttributeError:
            y_score = clf.decision_function(X_)
        f1 = f1_score(y_, y_pred)
        auc = roc_auc_score(y_, y_score)
        ap = average_precision_score(y_, y_score)
        rows.append((name, split_name, f1, auc, ap))
        print(
            f"  {name:<14} {split_name:<5} "
            f"F1={f1:.4f}  AUC={auc:.4f}  AUPRC={ap:.4f}"
        )
    return rows


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print(f"Loading MG24 (split_mode={args.split_mode}, "
          f"subsample_ddos={args.mg24_subsample_ddos}, seed={args.seed})...")
    from utils.mg24_loader import (
        build_edges,
        load_mg24_data,
        to_pyg_hetero_data,
    )

    mg24 = load_mg24_data(
        root=os.path.join(args.data_root, "unsw_mg24"),
        subsample_ddos=args.mg24_subsample_ddos,
        seed=args.seed,
        prune_external_hosts=args.mg24_prune_external,
        min_host_flows=args.mg24_min_host_flows,
        verbose=True,
    )
    edges = build_edges(mg24)
    hd = to_pyg_hetero_data(
        mg24, edges, seed=args.seed, split_mode=args.split_mode,
    )

    # Feature ablation (rebuild features without specified columns).
    ablation = [c.strip() for c in args.feature_ablation.split(",") if c.strip()]
    if ablation:
        X, dropped, feature_cols = _build_flows_with_ablation(mg24.flows, ablation)
        print(f"\nFeature ablation: dropped {len(dropped)} columns: {dropped}")
        print(f"Remaining feature count: {len(feature_cols)}")
    else:
        X = hd["flow_node"].x.numpy()
        print(f"\nUsing full feature set: {X.shape[1]} columns")

    y = hd["flow_node"].y.numpy()
    train_mask = hd["flow_node"].train_mask.numpy()
    val_mask = hd["flow_node"].val_mask.numpy()
    test_mask = hd["flow_node"].test_mask.numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(
        f"\nSplit composition:\n"
        f"  Train: n={len(y_train):>8,}  mal={y_train.sum():>7,}  "
        f"mal%={y_train.mean()*100:.1f}%\n"
        f"  Val:   n={len(y_val):>8,}  mal={int(y_val.sum()):>7,}  "
        f"mal%={y_val.mean()*100:.1f}%\n"
        f"  Test:  n={len(y_test):>8,}  mal={int(y_test.sum()):>7,}  "
        f"mal%={y_test.mean()*100:.1f}%"
    )

    if args.max_train_rows > 0 and len(y_train) > args.max_train_rows:
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(len(y_train), size=args.max_train_rows, replace=False)
        X_train = X_train[sub_idx]
        y_train = y_train[sub_idx]
        print(f"  (train subsampled to {len(y_train):,} rows for speed)")

    # ── Baselines ────────────────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression

    print("\n── Baseline 1: Logistic Regression ──")
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="lbfgs", n_jobs=-1, random_state=args.seed,
    )
    lr.fit(X_train, y_train)
    _score_split("LogReg", lr, X_val, y_val, X_test, y_test)

    if not args.skip_rf:
        from sklearn.ensemble import RandomForestClassifier
        print("\n── Baseline 2: Random Forest (100 trees) ──")
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=args.seed,
        )
        rf.fit(X_train, y_train)
        _score_split("RandomForest", rf, X_val, y_val, X_test, y_test)

    # ── Reference: CI-RCT result from training log ───────────────────────────
    print(
        "\n── Reference: CI-RCT Phase 1 (hybrid, --node_limit 10000) ──\n"
        "  CI-RCT       val   F1=0.9802  AUC=1.0000\n"
        "  CI-RCT       test  F1=0.9865  AUC=1.0000"
    )
    print(
        "\nInterpretation: if LogReg/RF F1 is within ~3 points of CI-RCT, "
        "the high CI-RCT score is dataset-driven (CICFlowMeter feature "
        "discriminability). Pivot paper to explainability metrics."
    )


if __name__ == "__main__":
    main()
