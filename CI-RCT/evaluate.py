"""
CI-RCT evaluation entry point.

Evaluates a trained model across four metric dimensions:
  A. Classification:         F1-macro, AUC-ROC, Recall
  B. Root Cause Tracing:     RCP, CCV, MTD
  C. Explanation Quality:    EA, ER (requires ground-truth causal labels)
  D. φ-Stability:            Std(φ_t − φ_{t-1}) over test fraud nodes

Metric C ground-truth sources (automatically selected by dataset):
  - elliptic++  : LFPN (Labeled Fraud Propagation Neighborhood), via
                  utils/lfpn_utils.py.  Both "strict" (direct initiator
                  only) and "extended" (+ k-hop labeled illicit wallets
                  via AddrAddr) are run by default; control with
                  --lfpn_mode and --lfpn_k.
  - unsw_nb15   : Granger causality on per-IP attack-flow time series,
                  via utils/granger_utils.py.
  - other       : Metric C is skipped.

Usage:
  python evaluate.py --dataset dblp --checkpoint checkpoints/ci_rct_dblp_best.pt
  python evaluate.py --dataset elliptic --checkpoint checkpoints/ci_rct_elliptic_best.pt
  python evaluate.py --dataset elliptic++ --checkpoint ... --lfpn_mode both --lfpn_k 2
"""
import argparse
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import recall_score

from configs.config import CI_RCT_Config
from model.causal_shapley import compute_asymmetric_causal_shapley
from model.ci_rct import CI_RCT
from model.root_cause_tracer import RootCauseTracer
from utils.data_utils import build_typed_causal_graph_from_hetero, compute_type_offsets
from utils.metrics import (
    compute_classification_metrics,
    compute_root_cause_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CI-RCT model")
    parser.add_argument("--dataset", type=str, default="dblp",
                        choices=["dblp", "acm", "imdb", "elliptic", "elliptic++", "unsw_nb15"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--ce_threshold", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--node_limit", type=int, default=5000,
                        help="Max nodes in TypedCausalGraph BFS (default 5000)")
    parser.add_argument("--hop_limit", type=int, default=2,
                        help="BFS hop depth for TypedCausalGraph construction (default 2)")
    parser.add_argument("--num_seeds", type=int, default=20,
                        help="Number of fraud seed nodes for causal graph BFS (default 20)")
    parser.add_argument("--max_explain", type=int, default=50,
                        help="Max test fraud nodes to run full explanation on")
    parser.add_argument("--device", type=str, default="cpu")
    # Must match training config
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--type_emb_dim", type=int, default=16)
    # Must match training data loading
    parser.add_argument("--include_addr_addr", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Must match training: include wallet→wallet edges")
    parser.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Must match training: use fraud-anchored wallet subgraph")
    parser.add_argument("--fraud_subgraph_hops", type=int, default=2)
    parser.add_argument("--max_flows", type=int, default=200_000,
                        help="Max flow records for unsw_nb15 (0 = no limit).")
    # LFPN settings (Elliptic++ Metric C)
    parser.add_argument("--lfpn_mode", type=str, default="both",
                        choices=["strict", "extended", "both"],
                        help="Ground-truth mode for Elliptic++ Metric C. "
                             "strict = direct initiators only; "
                             "extended = + k-hop labeled illicit wallets; "
                             "both = run both and report two tables.")
    parser.add_argument("--lfpn_k", type=int, default=2,
                        help="k in LFPN_k for extended mode (default 2). "
                             "Use k=1/2/3 for sensitivity analysis.")
    return parser.parse_args()


def load_dataset(name: str, root: str, **kwargs):
    if name == "dblp":
        from torch_geometric.datasets import DBLP
        return DBLP(root=os.path.join(root, "dblp"))[0], "author"
    if name == "acm":
        from torch_geometric.datasets import ACM
        return ACM(root=os.path.join(root, "acm"))[0], "paper"
    if name == "imdb":
        from torch_geometric.datasets import IMDB
        return IMDB(root=os.path.join(root, "imdb"))[0], "movie"
    if name == "elliptic":
        from utils.elliptic_loader import load_elliptic_dataset
        return load_elliptic_dataset(root)
    if name == "elliptic++":
        from utils.elliptic_plus_loader import load_elliptic_plus_dataset
        return load_elliptic_plus_dataset(
            os.path.join(root, "Elliptic++"),
            include_addr_addr=kwargs.get("include_addr_addr", False),
            fraud_subgraph=kwargs.get("fraud_subgraph", False),
            fraud_subgraph_hops=kwargs.get("fraud_subgraph_hops", 2),
        )
    if name == "unsw_nb15":
        from utils.unsw_loader import load_unsw_dataset
        return load_unsw_dataset(
            os.path.join(root, "unsw_nb15"),
            max_flows=kwargs.get("max_flows", 200_000),
        )
    raise ValueError(f"Unknown dataset: {name!r}")


def print_section(title: str, metrics: dict) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"  {label:40s}: {value:.4f}")
        else:
            print(f"  {label:40s}: {value}")


# ── Metric A: Classification ───────────────────────────────────────────────────

@torch.no_grad()
def eval_classification(model, data, labels, test_mask):
    model.eval()
    logits, _ = model.forward(data)
    preds  = logits[test_mask].argmax(dim=-1).cpu()
    scores = torch.softmax(logits[test_mask], dim=-1)[:, 1].cpu()
    y_true = labels[test_mask].cpu()

    metrics = compute_classification_metrics(preds, y_true, scores)
    # Add per-class Recall for fraud detection focus
    metrics["recall_fraud"] = float(
        recall_score(y_true.numpy(), preds.numpy(), pos_label=1, zero_division=0)
    )
    return metrics


# ── Metric B + D: Root Cause Tracing + φ-Stability ────────────────────────────

def eval_root_cause_and_stability(model, data, labels, test_mask,
                                  causal_graph, args, device,
                                  type_offsets, target_type):
    """
    Run root cause tracing on fraud-predicted test nodes.
    Computes:
      - RCP, CCV, MTD  (Dimension B)
      - φ-Stability    (Dimension D)

    Uses global node IDs (local index + type offset) to correctly
    address nodes inside the TypedCausalGraph.
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

    offset = type_offsets[target_type]
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()

    # Convert to global IDs; only keep nodes that exist in the causal graph
    fraud_predicted_global = [
        offset + idx for idx in test_indices
        if logits[idx].argmax().item() == 1
           and (offset + idx) in causal_graph.set_v
    ][: args.max_explain]

    if not fraud_predicted_global:
        print("  No fraud-predicted test nodes in causal graph — skipping RCT metrics.")
        return {}, {}

    predicted_roots, causal_chains = [], []

    # φ-Stability: compute perturbed causal effects once for the whole batch.
    _noise_sigma = 0.01
    with torch.no_grad():
        flat_h_perturbed = {
            gid: emb + torch.randn_like(emb) * _noise_sigma
            for gid, emb in flat_h.items()
        }
    causal_effects_perturbed = model.compute_causal_effects(flat_h_perturbed, causal_graph)

    stability_diffs = []

    for global_id in fraud_predicted_global:
        root, chain = tracer.trace_root_cause(global_id, causal_effects)
        predicted_roots.append(root)
        causal_chains.append(chain)

        phi_orig = compute_asymmetric_causal_shapley(causal_effects, causal_graph, global_id)
        phi_pert = compute_asymmetric_causal_shapley(causal_effects_perturbed, causal_graph, global_id)
        for p in set(phi_orig.keys()) & set(phi_pert.keys()):
            stability_diffs.append(abs(phi_orig[p] - phi_pert[p]))

    # Ground-truth fraud nodes (global IDs) for RCP / CCV
    fraud_label_set = set(
        offset + i for i in test_indices if labels[i].item() == 1
    )

    rct_metrics = compute_root_cause_metrics(
        predicted_roots, causal_chains, fraud_label_set
    )
    rct_metrics["num_traced"] = len(predicted_roots)

    stability_metrics = {
        "phi_stability_std": float(np.std(stability_diffs)) if stability_diffs else 0.0,
        "phi_stability_mean_abs": float(np.mean(stability_diffs)) if stability_diffs else 0.0,
        "num_nodes_explained": len(fraud_predicted_global),
    }

    return rct_metrics, stability_metrics


# ── Metric C: Explanation Quality (optional, requires ground truth) ────────────

def eval_explanation_quality(model, data, labels, test_mask,
                              causal_graph, args, gt_causal_nodes=None):
    """
    EA and ER metrics.  Skipped if no ground-truth causal node labels provided.

    gt_causal_nodes: dict {node_id: set_of_gt_causal_node_ids} or None

    Only GT nodes that exist in the causal graph are evaluated.  Tracing uses
    threshold=0.0 so that cross-type edges (e.g. wallet→tx) are not pruned by
    the CE threshold; the goal here is coverage, not precision filtering.
    """
    if not gt_causal_nodes:
        print("  No ground-truth causal labels — skipping explanation quality metrics.")
        return {}

    # Filter to GT nodes that are actually reachable in the causal graph
    eligible = {
        nid: gs for nid, gs in gt_causal_nodes.items()
        if nid in causal_graph.set_v
    }
    print(f"  Explanation Quality: {len(eligible)}/{len(gt_causal_nodes)} "
          f"GT transactions found in causal graph.")
    if not eligible:
        print("  No GT transactions in causal graph — skipping explanation quality metrics.")
        return {}

    model.eval()
    with torch.no_grad():
        logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)

    causal_effects = model.compute_causal_effects(flat_h, causal_graph)
    # Use threshold=0.0 so cross-type edges are not pruned during explanation
    tracer = RootCauseTracer(
        causal_graph=causal_graph,
        max_hops=args.max_hops,
        threshold=0.0,
    )

    preds_list, gts_list = [], []
    for node_id, gt_set in eligible.items():
        root, chain = tracer.trace_root_cause(node_id, causal_effects)
        preds_list.append(set(chain))
        gts_list.append(gt_set)

    from utils.metrics import compute_explanation_metrics
    return compute_explanation_metrics(preds_list, gts_list)


# ── Metric C ground-truth dispatcher ──────────────────────────────────────────

def build_gt_list(
    args:          argparse.Namespace,
    data,
    type_offsets:  Dict[str, int],
) -> List[Tuple[str, Dict[int, Set[int]]]]:
    """
    Build the Metric C ground-truth for the current dataset.

    Returns a list of (label, gt_dict) so downstream code can evaluate
    the same model under multiple GT definitions (e.g. LFPN-Strict and
    LFPN-Extended on Elliptic++).  An empty list means Metric C will
    be skipped.
    """
    gt_list: List[Tuple[str, Dict[int, Set[int]]]] = []

    if args.dataset == "unsw_nb15" and hasattr(data, "_df"):
        print("Computing Granger ground-truth for Metric C (UNSW-NB15)...")
        from utils.granger_utils import compute_granger_ground_truth
        gt = compute_granger_ground_truth(
            df=data._df,
            ip_global_offset=type_offsets.get("ip_node", 0),
            flow_global_offset=type_offsets.get("flow_node", 0),
            window_size=60,
            max_lag=3,
            p_threshold=0.05,
            verbose=True,
        )
        if gt:
            gt_list.append(("Granger", gt))

    elif args.dataset == "elliptic++":
        from utils.lfpn_utils import compute_lfpn_ground_truth

        modes = ["strict", "extended"] if args.lfpn_mode == "both" \
                else [args.lfpn_mode]

        for m in modes:
            print(f"\nComputing LFPN ground-truth (mode={m}) for Metric C...")
            gt = compute_lfpn_ground_truth(
                data_root=os.path.join(args.data_root, "Elliptic++"),
                tx_global_offset=type_offsets.get("transaction", 0),
                wallet_global_offset=type_offsets.get("wallet", 0),
                mode=m,
                k_hops=args.lfpn_k,
                include_addr_addr=args.include_addr_addr,
                fraud_subgraph=args.fraud_subgraph,
                fraud_subgraph_hops=args.fraud_subgraph_hops,
                verbose=True,
            )
            if gt:
                label = "LFPN-Strict" if m == "strict" \
                        else f"LFPN-Extended (k={args.lfpn_k})"
                gt_list.append((label, gt))

    return gt_list


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading dataset: {args.dataset}")
    data, target_type = load_dataset(
        args.dataset, args.data_root,
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
        max_flows=args.max_flows,
    )
    data = data.to(device)

    labels    = data[target_type].y
    test_mask = data[target_type].test_mask

    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_global_ids = [
        offset + i for i in test_indices if labels[i].item() == 1
    ]

    # ── Metric C ground-truth (computed BEFORE causal graph so GT tx IDs
    #    can be added as seeds, guaranteeing they appear in the graph) ─────────
    gt_list = build_gt_list(args, data, type_offsets)

    # Build causal graph seeded from test fraud nodes + all GT tx IDs across
    # every GT definition, so Metric C transactions are guaranteed reachable.
    print("\nBuilding TypedCausalGraph...")
    gt_tx_ids: List[int] = []
    for _, gt in gt_list:
        gt_tx_ids.extend(gt.keys())
    seed_ids = list(dict.fromkeys(
        fraud_global_ids[:args.num_seeds] + gt_tx_ids
    ))
    causal_graph = build_typed_causal_graph_from_hetero(
        data,
        seed_node_ids=seed_ids if seed_ids else None,
        hop_limit=args.hop_limit,
        node_limit=args.node_limit,
    )
    print(f"  Causal graph: {len(causal_graph.v)} nodes, "
          f"{len(causal_graph.edge_type_map)} directed edges")

    config = CI_RCT_Config(
        dataset=args.dataset,
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
        use_gan=False,  # evaluation always runs in inference mode
    ).to(device)

    if args.checkpoint:
        model.load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — evaluating randomly initialised model (baseline).")

    # ── Dimension A: Classification ──────────────────────────────────────────
    cls_metrics = eval_classification(model, data, labels, test_mask)
    print_section("A. Classification Metrics", cls_metrics)

    # ── Dimensions B + D: Root Cause Tracing + φ-Stability ──────────────────
    rct_metrics, stab_metrics = eval_root_cause_and_stability(
        model, data, labels, test_mask, causal_graph, args, device,
        type_offsets=type_offsets, target_type=target_type,
    )
    if rct_metrics:
        print_section("B. Root Cause Tracing Metrics", rct_metrics)
    if stab_metrics:
        print_section("D. φ-Stability Metrics", stab_metrics)

    # ── Dimension C: Explanation Quality (once per GT definition) ────────────
    if not gt_list:
        print("\n  Metric C: no ground-truth available — skipping.")
    else:
        for gt_label, gt_dict in gt_list:
            print(f"\n[Metric C — GT = {gt_label}]")
            expl_metrics = eval_explanation_quality(
                model, data, labels, test_mask, causal_graph, args,
                gt_causal_nodes=gt_dict,
            )
            if expl_metrics:
                print_section(f"C. Explanation Quality — {gt_label}",
                              expl_metrics)

    print(f"\n{'─' * 55}\n  Evaluation complete.\n{'─' * 55}\n")


if __name__ == "__main__":
    main()