"""
L2 diagnostic — frozen-backbone multi-class probe.

Answers the question: does the dd13-trained backbone embedding already
encode attack-type information, or did it only learn the binary
malicious/benign signal?

Method:
  1. Load the dd13 checkpoint (or any trained CI_RCT checkpoint).
  2. Run forward() once to get h_dict (per-node-type embeddings).
  3. For each labelled node type, fit a sklearn MLP probe to predict
     attack_type from the (frozen) embedding. Report macro-F1.
  4. Compare two probe splits:
       - by_file   : row-level random split (each attack_type appears in
                     train/val/test) — gives the *ceiling* of what the
                     embedding contains.
       - by_incident: matches main training split (test attack_types
                     unseen in train) — gives the generalization *floor*.

Verdict heuristic on `by_file` macro-F1:
  >= 0.7  backbone already encodes attack-type; hierarchical = inference
          re-wiring, no retrain needed
  0.4–0.7 backbone has partial signal; fine-tune (L3) needed
  < 0.4   backbone has no attack-type signal; full retrain (L4) or skip
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import CI_RCT_Config  # noqa: E402
from model.ci_rct import CI_RCT  # noqa: E402
from utils.mg24_loader import (  # noqa: E402
    _attack_type_from_audit_host_ref,
    _build_global_incident_split,
    _incident_groups_for_flows,
    _incident_groups_for_measurements,
    _incident_groups_for_processes,
    build_edges,
    load_mg24_data,
    to_pyg_hetero_data,
)


# ── Attack-type label derivation per node type ────────────────────────────────


def attack_labels_for_flows(flows: pd.DataFrame) -> np.ndarray:
    """One label per flow_node row. Uses the dataframe's attack_type directly."""
    return flows["attack_type"].astype(str).fillna("unknown").values


def attack_labels_for_processes(processes: pd.DataFrame) -> np.ndarray:
    """
    One label per process_node row.

    - audit-derived malicious: parse host_ref → attack_type (e.g. "ddos")
    - audit-derived benign:    "benign"
    - procmon-derived:         "procmon_mal" / "procmon_benign" (no attack_type)
    """
    if processes.empty:
        return np.array([], dtype=object)
    src = processes["source"].astype(str).values
    host_ref = processes["host_ref"].astype(str).values
    is_mal = processes["is_malicious"].astype(int).values
    out = np.empty(len(processes), dtype=object)
    for i in range(len(processes)):
        if src[i] == "audit":
            if is_mal[i] == 1:
                at = _attack_type_from_audit_host_ref(host_ref[i], 1)
                out[i] = at if at else "other_malicious"
            else:
                out[i] = "benign"
        else:
            out[i] = "procmon_mal" if is_mal[i] == 1 else "procmon_benign"
    return out


def attack_labels_for_measurements(measurements: pd.DataFrame) -> np.ndarray:
    """measurement_node has no attack_type — only mal/benign per device."""
    if measurements.empty:
        return np.array([], dtype=object)
    dev = measurements["device_id"].astype(str).values
    is_mal = measurements["is_malicious"].astype(int).values
    return np.array(
        [f"{dev[i]}_{'mal' if is_mal[i] == 1 else 'benign'}" for i in range(len(dev))],
        dtype=object,
    )


# ── Probe splits ──────────────────────────────────────────────────────────────


def random_row_split(
    n: int, val_ratio: float, test_ratio: float, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-level random split returning index arrays for train/val/test."""
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def incident_row_split(
    incident_groups: np.ndarray, incident_split: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate the global incident-split assignment into per-row indices."""
    splits = np.array(
        [incident_split.get(g, "train") for g in incident_groups],
        dtype=object,
    )
    return (
        np.flatnonzero(splits == "train"),
        np.flatnonzero(splits == "val"),
        np.flatnonzero(splits == "test"),
    )


# ── Probe fit ─────────────────────────────────────────────────────────────────


def fit_probe(
    X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
    *, hidden_size: int, max_iter: int, seed: int,
) -> Dict[str, object]:
    """
    Fit a small MLP probe on (X_train, y_train), evaluate on val and test.

    Returns dict with macro-F1, accuracy, per-class F1 table, confusion
    matrix DataFrame, and the set of classes seen in train.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (
        f1_score, accuracy_score, classification_report, confusion_matrix,
    )

    # Filter degenerate splits.
    if len(train_idx) < 20 or len(test_idx) < 5:
        return {"error": f"insufficient samples (train={len(train_idx)}, test={len(test_idx)})"}

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    train_classes = sorted(set(ytr.tolist()))
    if len(train_classes) < 2:
        return {"error": f"only {len(train_classes)} class in train"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_size,),
            max_iter=max_iter,
            random_state=seed,
            early_stopping=False,
        )
        clf.fit(Xtr, ytr)

        pred_va = clf.predict(Xva) if len(val_idx) > 0 else np.array([], dtype=object)
        pred_te = clf.predict(Xte)

        report_te = classification_report(
            yte, pred_te, output_dict=True, zero_division=0,
        )
        all_labels = sorted(set(yte.tolist()) | set(pred_te.tolist()))
        cm = confusion_matrix(yte, pred_te, labels=all_labels)
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)

    out: Dict[str, object] = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_train_classes": int(len(train_classes)),
        "n_test_classes": int(len(set(yte.tolist()))),
        "n_unseen_test_classes": int(
            len(set(yte.tolist()) - set(train_classes))
        ),
        "test_macro_f1": float(f1_score(yte, pred_te, average="macro", zero_division=0)),
        "test_micro_f1": float(f1_score(yte, pred_te, average="micro", zero_division=0)),
        "test_accuracy": float(accuracy_score(yte, pred_te)),
        "per_class": pd.DataFrame(report_te).T,
        "confusion_matrix": cm_df,
    }
    if len(val_idx) > 0:
        out["val_macro_f1"] = float(
            f1_score(yva, pred_va, average="macro", zero_division=0)
        )
        out["val_accuracy"] = float(accuracy_score(yva, pred_va))
    return out


# ── Per-node-type probe driver ───────────────────────────────────────────────


def run_probe_for_node_type(
    node_type: str,
    embedding: np.ndarray,
    labels: np.ndarray,
    incident_groups: Optional[np.ndarray],
    incident_split: Optional[Dict[str, str]],
    *,
    seed: int,
    out_dir: Path,
    hidden_size: int,
    max_iter: int,
    sample_max: int,
) -> None:
    print(f"\n{'─' * 70}")
    print(f"  L2 probe — {node_type}")
    print(f"{'─' * 70}")
    print(f"  embedding shape: {embedding.shape}")
    print(f"  label distribution: "
          f"{dict(sorted(Counter(labels.tolist()).items(), key=lambda kv: -kv[1])[:10])}"
          f"{' ...' if len(set(labels)) > 10 else ''}")

    rng = np.random.default_rng(seed)

    # Subsample for tractability — multi-class MLP on 1.3M rows is slow.
    n = len(embedding)
    if n > sample_max:
        sub = rng.choice(n, size=sample_max, replace=False)
        sub.sort()
        emb_s = embedding[sub]
        lab_s = labels[sub]
        inc_s = incident_groups[sub] if incident_groups is not None else None
        print(f"  subsampled to {sample_max:,} rows for probe fit")
    else:
        emb_s = embedding
        lab_s = labels
        inc_s = incident_groups

    # ── by_file (row-level random) ──
    print("\n  [Probe A] by_file (row-level random split):")
    tr, va, te = random_row_split(len(emb_s), 0.15, 0.15, np.random.default_rng(seed))
    result_file = fit_probe(
        emb_s, lab_s, tr, va, te,
        hidden_size=hidden_size, max_iter=max_iter, seed=seed,
    )
    _print_probe_result(result_file, label="by_file")
    _save_probe_artifacts(result_file, node_type, "by_file", out_dir)

    # ── by_incident (matches main pipeline) ──
    if inc_s is not None and incident_split is not None:
        print("\n  [Probe B] by_incident (matches main training split):")
        tr, va, te = incident_row_split(inc_s, incident_split)
        result_inc = fit_probe(
            emb_s, lab_s, tr, va, te,
            hidden_size=hidden_size, max_iter=max_iter, seed=seed,
        )
        _print_probe_result(result_inc, label="by_incident")
        _save_probe_artifacts(result_inc, node_type, "by_incident", out_dir)


def _print_probe_result(result: Dict[str, object], *, label: str) -> None:
    if "error" in result:
        print(f"    [{label}] skipped: {result['error']}")
        return
    print(
        f"    train={result['n_train']:>7,}  val={result['n_val']:>7,}  "
        f"test={result['n_test']:>7,}"
    )
    print(
        f"    classes: train={result['n_train_classes']}  "
        f"test={result['n_test_classes']}  "
        f"unseen-in-test={result['n_unseen_test_classes']}"
    )
    if "val_macro_f1" in result:
        print(
            f"    val:  macro-F1 = {result['val_macro_f1']:.4f}  "
            f"accuracy = {result['val_accuracy']:.4f}"
        )
    print(
        f"    test: macro-F1 = {result['test_macro_f1']:.4f}  "
        f"micro-F1 = {result['test_micro_f1']:.4f}  "
        f"accuracy = {result['test_accuracy']:.4f}"
    )


def _save_probe_artifacts(
    result: Dict[str, object], node_type: str, split_name: str, out_dir: Path,
) -> None:
    if "error" in result:
        return
    pc: pd.DataFrame = result["per_class"]  # type: ignore[assignment]
    cm: pd.DataFrame = result["confusion_matrix"]  # type: ignore[assignment]
    pc.to_csv(out_dir / f"l2_{node_type}_{split_name}_per_class.csv")
    cm.to_csv(out_dir / f"l2_{node_type}_{split_name}_confusion.csv")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to dd13 (or similar) CI_RCT checkpoint .pt")
    parser.add_argument("--data_root", default="data/unsw_mg24")
    parser.add_argument("--output_dir", default="logs/diag_l2")
    parser.add_argument("--device", default="cuda",
                        choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample_ddos", type=float, default=0.1,
                        help="Match dd13 training config.")
    parser.add_argument("--mg24_host_role", default="zeroed",
                        choices=("full", "no_mal_count", "zeroed"))
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Must match the checkpoint's hidden_dim.")
    parser.add_argument("--num_hgt_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--type_emb_dim", type=int, default=16)
    parser.add_argument("--probe_hidden", type=int, default=128)
    parser.add_argument("--probe_max_iter", type=int, default=200)
    parser.add_argument("--sample_max", type=int, default=80000,
                        help="Probe-fit subsample cap per node type.")
    parser.add_argument("--skip_node_types", default="",
                        help="Comma-separated node types to skip "
                             "(e.g. 'measurement_node,host_node').")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[L2] device: {device}")
    skip_types = {t.strip() for t in args.skip_node_types.split(",") if t.strip()}

    # ── Data ──
    print(f"[L2] loading MG24 (subsample_ddos={args.subsample_ddos})")
    mg24 = load_mg24_data(
        Path(args.data_root),
        subsample_ddos=args.subsample_ddos,
        seed=args.seed, verbose=False,
    )
    edges = build_edges(mg24)
    data = to_pyg_hetero_data(
        mg24, edges,
        seed=args.seed,
        split_mode="by_incident",
        host_features_mode=args.mg24_host_role,
    ).to(device)
    print(
        f"[L2] node counts: "
        + ", ".join(f"{nt}={data[nt].x.size(0):,}"
                    for nt in sorted(data.node_types)
                    if data[nt].x is not None)
    )

    # ── Model + checkpoint ──
    config = CI_RCT_Config(
        hidden_dim=args.hidden_dim,
        num_hgt_layers=args.num_hgt_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        node_type_emb_dim=args.type_emb_dim,
    )
    in_channels_dict = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types)
        if data[nt].x is not None
    }
    model = CI_RCT(
        config=config,
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        use_gan=False,
    ).to(device)
    model.load_checkpoint(args.checkpoint, device=str(device))
    model.eval()
    print(f"[L2] loaded checkpoint: {args.checkpoint}")

    # ── Forward pass to extract embeddings ──
    print("[L2] running forward pass (frozen backbone)...")
    with torch.no_grad():
        _, h_dict = model.forward(data)
    embeddings: Dict[str, np.ndarray] = {
        nt: h.detach().cpu().numpy() for nt, h in h_dict.items()
    }

    # ── Per-node-type probe ──
    incident_groups_by_type: Dict[str, np.ndarray] = {
        "flow_node": _incident_groups_for_flows(mg24.flows),
        "process_node": _incident_groups_for_processes(mg24.processes),
        "measurement_node": _incident_groups_for_measurements(mg24.measurements),
    }
    rng_split = np.random.default_rng(args.seed)
    incident_split = _build_global_incident_split(
        incident_groups_by_type,
        val_ratio=0.15, test_ratio=0.15, rng=rng_split,
    )

    label_builders = {
        "flow_node": lambda: attack_labels_for_flows(mg24.flows),
        "process_node": lambda: attack_labels_for_processes(mg24.processes),
        "measurement_node": lambda: attack_labels_for_measurements(mg24.measurements),
    }

    summary_rows: List[Dict[str, object]] = []
    for ntype in ("flow_node", "process_node", "measurement_node"):
        if ntype in skip_types:
            continue
        if ntype not in embeddings:
            print(f"\n[L2] {ntype}: no embedding in h_dict, skipping")
            continue
        emb = embeddings[ntype]
        labels = label_builders[ntype]()
        if len(labels) != emb.shape[0]:
            print(f"[L2] {ntype}: row mismatch "
                  f"(labels={len(labels)} vs emb={emb.shape[0]}), skipping")
            continue
        run_probe_for_node_type(
            ntype, emb, labels,
            incident_groups=incident_groups_by_type.get(ntype),
            incident_split=incident_split,
            seed=args.seed, out_dir=out_dir,
            hidden_size=args.probe_hidden,
            max_iter=args.probe_max_iter,
            sample_max=args.sample_max,
        )

    print(f"\n[L2] all per-class / confusion artifacts written to {out_dir.resolve()}")
    print("[L2] verdict heuristic on by_file macro-F1 across node types:")
    print("  >= 0.7   backbone already encodes attack-type; hierarchical "
          "inference-only viable")
    print("  0.4–0.7  backbone has partial signal; fine-tune (L3) advised")
    print("  < 0.4    backbone lacks attack-type signal; full retrain (L4) "
          "or skip hierarchical")


if __name__ == "__main__":
    main()
