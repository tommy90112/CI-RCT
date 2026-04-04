"""
CI-RCT evaluation entry point for the crypto exchange dataset.

Evaluation dimensions:
  A. Classification:     F1-macro, AUC-ROC, Recall (fraud class)
  B. Root Cause Tracing: Root Cause Precision (RCP), Fraud Hit Rate (FHR),
                         Mean Tracing Depth (MTD)
  D. φ-Stability:        Std(φ_t − φ_{t-1}) over test fraud nodes
  V. Case Studies:       Causal chain figures saved to figures/case_studies/

Differences from evaluate.py:
  - Loads the crypto exchange HeteroData via crypto_loader
  - Computes Fraud Hit Rate (= CCV over blacklist labels, not model preds)
  - Exports chain visualizations for the top-N longest chains

Usage:
  python evaluate_crypto.py \\
      --data_root data/crypto \\
      --checkpoint checkpoints/ci_rct_crypto_best.pt \\
      --case_studies 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import recall_score

from configs.config import CI_RCT_Config
from evaluate import (
    eval_classification,
    eval_explanation_quality,
    print_section,
)
from model.causal_shapley import compute_asymmetric_causal_shapley
from model.ci_rct import CI_RCT
from model.root_cause_tracer import RootCauseTracer
from utils.chain_visualizer import draw_case_studies
from utils.crypto_loader import load_crypto_dataset
from utils.data_utils import build_typed_causal_graph_from_hetero, compute_type_offsets
from utils.metrics import compute_root_cause_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CI-RCT on the crypto exchange dataset")
    p.add_argument("--data_root",   type=str, default="data/crypto")
    p.add_argument("--checkpoint",  type=str, default=None)
    p.add_argument("--max_hops",    type=int, default=5)
    p.add_argument("--ce_threshold",type=float, default=0.1)
    p.add_argument("--top_k",       type=int, default=3)
    p.add_argument("--node_limit",  type=int, default=1000)
    p.add_argument("--max_explain", type=int, default=100,
                   help="Max test fraud nodes to run full explanation on")
    p.add_argument("--case_studies",type=int, default=5,
                   help="Number of case study chain figures to generate")
    p.add_argument("--fig_dir",     type=str, default="figures/case_studies")
    p.add_argument("--device",      type=str, default="cpu")
    # Must match training config
    p.add_argument("--hidden_dim",      type=int,   default=128)
    p.add_argument("--num_hgt_layers",  type=int,   default=3)
    p.add_argument("--num_heads",       type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--type_emb_dim",    type=int,   default=16)
    return p.parse_args()


# ── Dimension B + D ────────────────────────────────────────────────────────────

def eval_rct_and_stability(
    model,
    data,
    labels,
    test_mask,
    fraud_label_set: set,
    causal_graph,
    node_type_map: dict,
    args,
    device,
    n_case_studies: int = 5,
):
    """
    Run explanation on fraud-predicted test nodes.

    Returns
    -------
    rct_metrics     dict  RCP / FHR / MTD
    stab_metrics    dict  φ-stability
    case_data       list  dicts ready for draw_case_studies()
    """
    model.eval()
    with torch.no_grad():
        logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)

    causal_effects = model.compute_causal_effects(flat_h, causal_graph)

    tracer = RootCauseTracer(
        causal_graph=causal_graph,
        max_hops=args.max_hops,
        threshold=args.ce_threshold,
    )

    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_predicted = [
        idx for idx in test_indices
        if logits[idx].argmax().item() == 1
    ][: args.max_explain]

    if not fraud_predicted:
        print("  No fraud-predicted test nodes — skipping RCT metrics.")
        return {}, {}, []

    predicted_roots, causal_chains, phi_list = [], [], []

    for node_id in fraud_predicted:
        root, chain = tracer.trace_root_cause(node_id, causal_effects)
        predicted_roots.append(root)
        causal_chains.append(chain)

        phi = compute_asymmetric_causal_shapley(causal_effects, causal_graph, node_id)
        phi_list.append(phi)

    rct_metrics = compute_root_cause_metrics(
        predicted_roots, causal_chains, fraud_label_set
    )
    rct_metrics["num_traced"] = len(predicted_roots)

    # Rename chain_validity → fraud_hit_rate to match thesis terminology
    rct_metrics["fraud_hit_rate"] = rct_metrics.pop("chain_validity", 0.0)

    # φ-Stability
    stability_diffs = []
    for i in range(1, len(phi_list)):
        common = set(phi_list[i].keys()) & set(phi_list[i - 1].keys())
        for p in common:
            stability_diffs.append(abs(phi_list[i][p] - phi_list[i - 1][p]))

    stab_metrics = {
        "phi_stability_std":      float(np.std(stability_diffs)) if stability_diffs else 0.0,
        "phi_stability_mean_abs": float(np.mean(stability_diffs)) if stability_diffs else 0.0,
        "num_nodes_explained":    len(fraud_predicted),
    }

    # ── Case study data ──────────────────────────────────────────────────────
    # Pick chains with longest depth (most interesting for visualization)
    indexed = sorted(
        enumerate(causal_chains),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:n_case_studies]

    case_data = []
    for rank, (orig_idx, chain) in enumerate(indexed, start=1):
        target_id = fraud_predicted[orig_idx]
        root_id   = predicted_roots[orig_idx]
        title = (
            f"Case #{rank} — target=node_{target_id} "
            f"root=node_{root_id} depth={len(chain)-1}"
        )
        case_data.append({
            "chain":          chain,
            "causal_effects": causal_effects,
            "fraud_set":      fraud_label_set,
            "node_type_map":  node_type_map,
            "title":          title,
        })

    return rct_metrics, stab_metrics, case_data


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    print("Loading crypto exchange dataset …")
    data, target_type = load_crypto_dataset(args.data_root)
    data = data.to(device)

    labels    = data[target_type].y
    test_mask = data[target_type].test_mask

    # ── Build TypedCausalGraph (multi-source BFS from test fraud nodes) ──────
    print("Building TypedCausalGraph …")
    type_offsets = compute_type_offsets(data)

    fraud_test_global_ids = [
        type_offsets[target_type] + int(i)
        for i in test_mask.nonzero(as_tuple=True)[0].tolist()
        if labels[i].item() == 1
    ]

    causal_graph = build_typed_causal_graph_from_hetero(
        data,
        seed_node_ids=fraud_test_global_ids[:50],   # seed from known fraud nodes
        hop_limit=args.max_hops,
        node_limit=args.node_limit,
    )

    # node_type_map: global_id → "user" | "wallet"
    node_type_map = dict(causal_graph.node_type)

    # fraud_label_set: global IDs of ALL labeled fraud users (train + val + test)
    user_offset = type_offsets[target_type]
    fraud_label_set = {
        user_offset + int(i)
        for i in range(len(labels))
        if labels[i].item() == 1
    }

    # ── Build model ──────────────────────────────────────────────────────────
    config = CI_RCT_Config(
        dataset="crypto",
        target_node_type=target_type,
        max_hops=args.max_hops,
        ce_threshold=args.ce_threshold,
        top_k_paths=args.top_k,
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

    if args.checkpoint:
        model.load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint — evaluating randomly initialised model.")

    # ── A: Classification ────────────────────────────────────────────────────
    cls_metrics = eval_classification(model, data, labels, test_mask)
    print_section("A. Classification Metrics", cls_metrics)

    # ── B + D: Root Cause Tracing + φ-Stability ──────────────────────────────
    rct_metrics, stab_metrics, case_data = eval_rct_and_stability(
        model, data, labels, test_mask,
        fraud_label_set, causal_graph, node_type_map,
        args, device, n_case_studies=args.case_studies,
    )
    if rct_metrics:
        print_section("B. Root Cause Tracing Metrics", rct_metrics)
    if stab_metrics:
        print_section("D. φ-Stability Metrics", stab_metrics)

    # ── V: Case Study Visualizations ─────────────────────────────────────────
    if case_data:
        print(f"\nGenerating {len(case_data)} case study figures → {args.fig_dir}")
        saved = draw_case_studies(case_data, save_dir=args.fig_dir)
        for path in saved:
            print(f"  Saved: {path}")

    print(f"\n{'─' * 55}\n  Evaluation complete.\n{'─' * 55}\n")


if __name__ == "__main__":
    main()
