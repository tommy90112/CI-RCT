"""
Feature-level fairness audit for MG24 flow detection.

Motivation
──────────
After the 4-group host_role ablation we discovered that host_node features
contribute < 1 pp to detection F1 — so the high F1 must come from flow_node's
78 CICFlowMeter features. This script audits each of those 78 features
individually to identify which (if any) are:

  (a) "Tool fingerprints"     — encode the attack tool's signature, so
                                 perfect predictors of label by themselves
                                 (Univariate AUC > 0.95)
  (b) "Label proxies"         — values strongly correlated with label even
                                 in single-feature view (top-quartile MI)
  (c) "Time/capture artifacts"— mal vs benign mean differs by orders of
                                 magnitude (class mean ratio > 10×)

A feature that is RED on all three columns is highly suspicious and should
be considered for removal or careful disclosure in §5 Limitation.

Three audits run on the SAME split used by train.py (default: hybrid).

Usage
─────
    python fairness_audit.py \
        --data_root data --split_mode hybrid \
        --mg24_subsample_ddos 0.1
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_root", default="data")
    p.add_argument("--split_mode", default="hybrid",
                   choices=("row", "by_file", "hybrid"))
    p.add_argument("--mg24_subsample_ddos", type=float, default=0.1)
    p.add_argument("--mg24_min_host_flows", type=int, default=5)
    p.add_argument("--mg24_host_role", type=str, default="full",
                   choices=("full", "no_mal_count", "zeroed",
                            "detection_excluded"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit_split", default="test",
                   choices=("train", "val", "test"),
                   help="Which split to evaluate univariate AUC on. "
                        "Use 'test' for the most honest audit.")
    p.add_argument("--top_k", type=int, default=20,
                   help="Print top-K features by univariate AUC.")
    p.add_argument("--max_rows_for_mi", type=int, default=200_000,
                   help="Cap rows used for mutual_info_classif (slow on full).")
    return p.parse_args()


def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC of using raw feature value (or its negation) as score for label=1.

    We test both directions and take max — many CICFlowMeter features are
    higher-for-benign rather than higher-for-malicious, so without this
    flip we'd see lots of misleading AUC≈0 instead of AUC≈1.
    """
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, x)
    return max(auc, 1.0 - auc)


def _class_mean_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Ratio of |mean(x | mal)| / |mean(x | benign)|.

    Robust to sign: uses abs to capture magnitude difference. A ratio > 10×
    means malicious flows have feature values an order of magnitude
    different from benign — strong univariate signal.
    """
    mean_mal = float(np.mean(x[y == 1]))
    mean_ben = float(np.mean(x[y == 0]))
    if abs(mean_ben) < 1e-12:
        return float("inf") if abs(mean_mal) > 1e-12 else 1.0
    return abs(mean_mal) / abs(mean_ben)


def _flag(value: float, threshold: float, op: str = ">") -> str:
    """Return ⚠ if value crosses threshold, else blank."""
    crossed = (value > threshold) if op == ">" else (value < threshold)
    return "⚠" if crossed else " "


def _print_table(rows: List[Tuple], header: List[str]) -> None:
    widths = [
        max(len(str(r[i])) for r in [header] + rows)
        for i in range(len(header))
    ]
    line = "  ".join(f"{header[i]:<{widths[i]}}" for i in range(len(header)))
    print(line)
    print("  ".join("─" * widths[i] for i in range(len(header))))
    for r in rows:
        print("  ".join(f"{str(r[i]):<{widths[i]}}" for i in range(len(header))))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print(f"Loading MG24 (split_mode={args.split_mode}, "
          f"host_role={args.mg24_host_role}, seed={args.seed})...")
    from utils.mg24_loader import (
        _FLOW_NON_FEATURE_COLS,
        build_edges,
        load_mg24_data,
        to_pyg_hetero_data,
    )

    mg24 = load_mg24_data(
        root=os.path.join(args.data_root, "unsw_mg24"),
        subsample_ddos=args.mg24_subsample_ddos,
        seed=args.seed,
        min_host_flows=args.mg24_min_host_flows,
        verbose=True,
    )
    edges = build_edges(mg24)
    host_features_mode = (
        args.mg24_host_role
        if args.mg24_host_role in ("full", "no_mal_count", "zeroed")
        else "full"
    )
    hd = to_pyg_hetero_data(
        mg24, edges, seed=args.seed,
        split_mode=args.split_mode,
        host_features_mode=host_features_mode,
    )

    # Use the RAW flow DataFrame so we can audit column NAMES, not just indices.
    flows = mg24.flows.reset_index(drop=True)
    feature_cols = [
        c for c in flows.columns
        if c not in _FLOW_NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(flows[c])
    ]
    print(f"\nAuditing {len(feature_cols)} flow_node feature columns "
          f"on split={args.audit_split}.")

    # Resolve mask for the requested split.
    mask = {
        "train": hd["flow_node"].train_mask,
        "val":   hd["flow_node"].val_mask,
        "test":  hd["flow_node"].test_mask,
    }[args.audit_split].numpy()
    y = hd["flow_node"].y.numpy()[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"  Audit set:  n={len(y):,}  mal={n_pos:,}  benign={n_neg:,}  "
          f"mal%={y.mean()*100:.1f}%")

    # Mutual information requires the same feature matrix.
    print("\nComputing mutual information (may take ~30s)...")
    from sklearn.feature_selection import mutual_info_classif
    X_raw = flows[feature_cols].astype(float).fillna(0).replace(
        [np.inf, -np.inf], 0).values
    X_audit = X_raw[mask]

    # Subsample for MI computation if too large.
    if len(y) > args.max_rows_for_mi:
        rng = np.random.default_rng(args.seed)
        sub = rng.choice(len(y), args.max_rows_for_mi, replace=False)
        X_for_mi, y_for_mi = X_audit[sub], y[sub]
    else:
        X_for_mi, y_for_mi = X_audit, y
    mi = mutual_info_classif(X_for_mi, y_for_mi, discrete_features=False,
                             random_state=args.seed, n_neighbors=3)

    # ── Compute per-feature stats ────────────────────────────────────────
    audit_rows = []
    for i, col in enumerate(feature_cols):
        x = X_audit[:, i]
        try:
            auc = _univariate_auc(x, y)
        except ValueError:
            auc = float("nan")  # constant column
        try:
            ratio = _class_mean_ratio(x, y)
        except Exception:
            ratio = float("nan")
        audit_rows.append((col, auc, mi[i], ratio))

    audit_rows.sort(key=lambda r: r[1], reverse=True)
    mi_q3 = float(np.quantile([r[2] for r in audit_rows], 0.75))

    # ── Print Top-K most discriminative ──────────────────────────────────
    header = ["#", "feature", "univAUC", "F", "MI", "F",
              "ratio(mal/ben)", "F", "verdict"]
    table_rows = []
    for i, (col, auc, mi_val, ratio) in enumerate(audit_rows[:args.top_k]):
        flag_auc = _flag(auc, 0.95)
        flag_mi = _flag(mi_val, mi_q3)
        flag_ratio = _flag(
            ratio if not np.isinf(ratio) else 1e9, 10.0,
        )
        red_count = sum(1 for f in (flag_auc, flag_mi, flag_ratio) if f == "⚠")
        verdict = (
            "🚨 LEAK?" if red_count >= 3
            else "⚠ check"  if red_count == 2
            else "OK"
        )
        ratio_str = (
            f"{ratio:,.2f}" if not np.isinf(ratio) and not np.isnan(ratio)
            else "inf"
        )
        table_rows.append((
            i + 1, col,
            f"{auc:.4f}", flag_auc,
            f"{mi_val:.4f}", flag_mi,
            ratio_str, flag_ratio,
            verdict,
        ))

    print(f"\n── Top-{args.top_k} most discriminative features "
          f"(by univariate AUC) ──")
    print(f"  Red flags: AUC>0.95, MI>Q3(={mi_q3:.4f}), ratio>10×")
    _print_table(table_rows, header)

    # ── Summary counters ────────────────────────────────────────────────
    n_auc_high = sum(1 for r in audit_rows if r[1] > 0.95)
    n_auc_perfect = sum(1 for r in audit_rows if r[1] > 0.99)
    n_ratio_high = sum(
        1 for r in audit_rows
        if (np.isfinite(r[3]) and r[3] > 10.0) or np.isinf(r[3])
    )
    print(
        f"\n── Summary ──\n"
        f"  Features with univariate AUC > 0.95: {n_auc_high:>3} / {len(audit_rows)}\n"
        f"  Features with univariate AUC > 0.99: {n_auc_perfect:>3} / {len(audit_rows)}\n"
        f"  Features with mal/ben mean ratio > 10×: {n_ratio_high:>3} / {len(audit_rows)}"
    )

    if n_auc_perfect > 0:
        print(
            f"\n  🚨 {n_auc_perfect} feature(s) achieve AUC > 0.99 alone — "
            "near-perfect single-column classifiers.\n"
            "  These are almost certainly tool fingerprints / capture "
            "artifacts. Consider an explicit ablation removing them and "
            "disclosing in paper §5 Limitation."
        )
    elif n_auc_high > 5:
        print(
            f"\n  ⚠ {n_auc_high} features hit AUC > 0.95 individually. "
            "Detection is inherently easy on this dataset — pivot paper "
            "contribution claim to explainability (Module 2/3), not raw F1."
        )
    else:
        print(
            "\n  ✓ No single feature is a near-perfect predictor. The high "
            "F1 must therefore come from multi-feature interactions, "
            "which is legitimate machine learning."
        )


if __name__ == "__main__":
    main()
