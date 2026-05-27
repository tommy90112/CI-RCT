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
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import recall_score
 
from configs.config import CI_RCT_Config
from model.causal_shapley import compute_asymmetric_causal_shapley
from model.ci_rct import CI_RCT
from model.root_cause_tracer import RootCauseTracer
from utils.data_utils import (
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
    default_blocked_edge_types,
    default_rare_edge_types,
)
from utils.metrics import (
    compute_classification_metrics,
    compute_root_cause_metrics,
)
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CI-RCT model")
    parser.add_argument("--dataset", type=str, default="dblp",
                        choices=["dblp", "acm", "imdb", "elliptic", "elliptic++",
                                 "unsw_nb15", "unsw_mg24"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--ce_threshold", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--node_limit", type=int, default=5000)
    parser.add_argument("--hop_limit", type=int, default=2)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--max_explain", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--type_emb_dim", type=int, default=16)
    parser.add_argument("--include_addr_addr", type=lambda x: x.lower() == "true",
                        default=False)
    parser.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true",
                        default=False)
    parser.add_argument("--fraud_subgraph_hops", type=int, default=2)
    parser.add_argument("--max_flows", type=int, default=200_000)
    # ── UNSW-MG24 specific (mirror train.py) ─────────────────────────────────
    parser.add_argument("--mg24_subsample_ddos", type=float, default=1.0)
    parser.add_argument("--mg24_min_host_flows", type=int, default=5)
    parser.add_argument("--mg24_prune_external", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--mg24_split_mode", type=str, default="by_file",
                        choices=("row", "by_file", "hybrid", "by_incident"))
    parser.add_argument("--mg24_host_role", type=str, default="full",
                        choices=("full", "no_mal_count", "zeroed",
                                 "detection_excluded"))
    parser.add_argument("--mg24_drop_features", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lfpn_mode", type=str, default="both",
                        choices=["strict", "extended", "both"])
    parser.add_argument("--lfpn_k", type=int, default=2)
    parser.add_argument("--debug", action="store_true",
                        help="Print tracer diagnostics: CE distribution by edge "
                             "type, chain length histogram, stuck-trace analysis.")
    # ── B1: type-aware BFS sampling for rare/bridge edges ────────────────────
    parser.add_argument("--rare_edge_types", type=str, default="",
                        help="Comma-separated edge-type strings "
                             "(formatted 'src__to__dst') that the BFS "
                             "subgraph sampler must guarantee inclusion of. "
                             "Designed for sparse bridge edges crowded out "
                             "by high-degree types. Empty string falls back "
                             "to default_rare_edge_types(dataset). Pass "
                             "'none' to disable the rare-edge pass entirely.")
    parser.add_argument("--rare_edge_reserve", type=int, default=500,
                        help="Node-budget reserve for the rare-edge pass.")
    parser.add_argument("--rare_edge_max_hops", type=int, default=5,
                        help="Chain depth for the rare-edge expansion.")
    parser.add_argument("--blocked_edge_types", type=str, default="",
                        help="Comma-separated edge-type strings dropped "
                             "from BOTH BFS expansion and the final causal "
                             "graph. Symmetric to --rare_edge_types. Empty "
                             "string falls back to "
                             "default_blocked_edge_types(dataset). Pass "
                             "'none' to disable the filter entirely.")
    return parser.parse_args()


def load_dataset(name, root, **kwargs):
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
    if name == "unsw_mg24":
        # Mirrors train.py's load_dataset(unsw_mg24, ...) so that the
        # HeteroData produced for evaluation has the SAME node/edge layout
        # and SAME split masks as the training run.
        from utils.mg24_loader import (
            build_edges,
            load_mg24_data,
            to_pyg_hetero_data,
        )
        mg24 = load_mg24_data(
            root=os.path.join(root, "unsw_mg24"),
            subsample_ddos=kwargs.get("mg24_subsample_ddos", 1.0),
            seed=kwargs.get("seed", 42),
            prune_external_hosts=kwargs.get("mg24_prune_external", True),
            min_host_flows=kwargs.get("mg24_min_host_flows", 5),
            verbose=True,
        )
        edges = build_edges(mg24)
        host_role = kwargs.get("mg24_host_role", "full")
        host_features_mode = (
            host_role if host_role in ("full", "no_mal_count", "zeroed")
            else "full"
        )
        drop_raw = kwargs.get("mg24_drop_features", "") or ""
        flow_features_exclude = [
            c.strip() for c in drop_raw.split(",") if c.strip()
        ]
        hd = to_pyg_hetero_data(
            mg24, edges,
            seed=kwargs.get("seed", 42),
            split_mode=kwargs.get("mg24_split_mode", "by_file"),
            host_features_mode=host_features_mode,
            flow_features_exclude=flow_features_exclude or None,
        )
        # DD-3 primary target: flow_node (same as train.py).
        # DD-16: stash the raw MG24Data on hd so build_gt_list can derive
        # kill-chain explanation ground truth from the original DataFrames.
        hd._mg24 = mg24  # type: ignore[attr-defined]
        return hd, "flow_node"
    raise ValueError(f"Unknown dataset: {name!r}")
 
 
def print_section(title, metrics):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"  {label:40s}: {value:.4f}")
        else:
            print(f"  {label:40s}: {value}")
 
 
def _load_illicit_wallet_globals(
    data_root: str,
    wallet_global_offset: int,
    include_addr_addr: bool = False,
    fraud_subgraph: bool = False,
    fraud_subgraph_hops: int = 2,
) -> set:
    """
    Return the set of *global node IDs* of labeled illicit wallets
    (wallets_classes.csv class==1) on Elliptic++.
 
    The wallet ordering must match what elliptic_plus_loader.py produces
    under the same loader flags — we rely on lfpn_utils._rebuild_wallet_to_idx
    to do this consistently.  If the loader ever changes its filtering rules,
    that helper is the single place to update.
 
    Returns an empty set on any failure (so RCP just falls back to
    "tx-only fraud nodes" instead of crashing).
    """
    try:
        import pandas as pd
        from pathlib import Path
        from utils.lfpn_utils import _rebuild_wallet_to_idx, CLASS_ILLICIT
    except Exception as e:
        print(f"  [fraud_label_set] could not load illicit wallet helper: {e}")
        return set()
 
    root = Path(os.path.join(data_root, "Elliptic++"))
    try:
        wallets_cls = pd.read_csv(root / "wallets_classes.csv")
        wallets_cls.columns = [c.strip() for c in wallets_cls.columns]
 
        # Same readers the loader uses, needed by _rebuild_wallet_to_idx.
        wallets   = pd.read_csv(root / "wallets_features.csv", usecols=[0])
        wallets.columns = ["address"]
        txs_feat  = pd.read_csv(root / "txs_features.csv", usecols=[0])
        txs_feat.columns = ["txId"]
        txs_cls_df = pd.read_csv(root / "txs_classes.csv")
        txs_cls_df.columns = [c.strip() for c in txs_cls_df.columns]
        addr_tx   = pd.read_csv(root / "AddrTx_edgelist.csv")
        addr_tx.columns = [c.strip() for c in addr_tx.columns]
        tx_addr   = pd.read_csv(root / "TxAddr_edgelist.csv")
        tx_addr.columns = [c.strip() for c in tx_addr.columns]
        addr_addr = pd.read_csv(root / "AddrAddr_edgelist.csv")
        addr_addr.columns = [c.strip() for c in addr_addr.columns]
 
        tx_to_idx = {tid: i for i, tid in enumerate(txs_feat["txId"].tolist())}
 
        wallet_to_idx = _rebuild_wallet_to_idx(
            wallets=wallets,
            wallets_cls=wallets_cls,
            txs_cls=txs_cls_df,
            tx_to_idx=tx_to_idx,
            addr_tx=addr_tx,
            tx_addr=tx_addr,
            addr_addr=addr_addr,
            include_addr_addr=include_addr_addr,
            fraud_subgraph=fraud_subgraph,
            fraud_subgraph_hops=fraud_subgraph_hops,
            verbose=False,
        )
 
        illicit_addrs = wallets_cls.loc[
            wallets_cls["class"] == CLASS_ILLICIT, "address"
        ].astype(str).tolist()
 
        return {
            wallet_global_offset + wallet_to_idx[a]
            for a in illicit_addrs
            if a in wallet_to_idx
        }
    except Exception as e:
        print(f"  [fraud_label_set] failed to build illicit wallet set: {e}")
        return set()
 
 
@torch.no_grad()
def eval_classification(model, data, labels, test_mask):
    model.eval()
    logits, _ = model.forward(data)
    preds  = logits[test_mask].argmax(dim=-1).cpu()
    scores = torch.softmax(logits[test_mask], dim=-1)[:, 1].cpu()
    y_true = labels[test_mask].cpu()
    metrics = compute_classification_metrics(preds, y_true, scores)
    metrics["recall_fraud"] = float(
        recall_score(y_true.numpy(), preds.numpy(), pos_label=1, zero_division=0)
    )
    return metrics
 
 
def eval_root_cause_and_stability(model, data, labels, test_mask,
                                  causal_graph, args, device,
                                  type_offsets, target_type):
    model.eval()
    with torch.no_grad():
        logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)
    causal_effects = model.compute_causal_effects(flat_h, causal_graph)
 
    # ── DIAGNOSTIC 1: CE distribution by edge type ─────────────────────────
    if args.debug:
        ce_by_etype = defaultdict(list)
        for (src, dst), ce in causal_effects.items():
            etype = causal_graph.edge_type_map.get((src, dst), "unknown")
            ce_by_etype[etype].append(ce)
        print("\n[diagnostic 1/3] CE distribution by edge type")
        print(f"  (ce_threshold = {args.ce_threshold})")
        print(f"  {'edge_type':<35s} {'n':>7s}  {'mean':>9s}  {'std':>8s}  "
              f"{'min':>8s}  {'max':>8s}  {'%>thresh':>9s}")
        for etype in sorted(ce_by_etype.keys()):
            arr = np.array(ce_by_etype[etype])
            pct = float((arr > args.ce_threshold).mean()) * 100
            print(f"  {etype:<35s} {len(arr):>7d}  "
                  f"{arr.mean():>+9.4f}  {arr.std():>8.4f}  "
                  f"{arr.min():>+8.4f}  {arr.max():>+8.4f}  "
                  f"{pct:>8.2f}%")
 
    tracer = RootCauseTracer(
        causal_graph=causal_graph,
        max_hops=args.max_hops,
        threshold=args.ce_threshold,
    )
 
    offset = type_offsets[target_type]
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_predicted_global = [
        offset + idx for idx in test_indices
        if logits[idx].argmax().item() == 1
           and (offset + idx) in causal_graph.set_v
    ][: args.max_explain]
 
    if not fraud_predicted_global:
        print("  No fraud-predicted test nodes in causal graph — skipping RCT metrics.")
        return {}, {}
 
    predicted_roots, causal_chains = [], []
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
 
    # ── Build fraud_label_set ─────────────────────────────────────────────
    # On bipartite-style fraud graphs the tracer's root is typically a
    # wallet (because |CE(wallet→tx)| dominates).  RCP/CCV must therefore
    # treat *both* illicit transactions and labeled illicit wallets as
    # fraud-related — a chain ending at a known illicit wallet is a
    # successful root-cause trace, not a miss.
    fraud_label_set = set(
        offset + i for i in test_indices if labels[i].item() == 1
    )
    if args.dataset == "elliptic++":
        wallet_offset = type_offsets.get("wallet")
        illicit_wallet_globals = _load_illicit_wallet_globals(
            args.data_root,
            wallet_global_offset=wallet_offset,
            include_addr_addr=args.include_addr_addr,
            fraud_subgraph=args.fraud_subgraph,
            fraud_subgraph_hops=args.fraud_subgraph_hops,
        )
        fraud_label_set |= illicit_wallet_globals
        if args.debug:
            print(f"\n[fraud_label_set] tx fraud nodes: "
                  f"{sum(1 for i in test_indices if labels[i].item() == 1)}, "
                  f"illicit wallets added: {len(illicit_wallet_globals):,}, "
                  f"total fraud_label_set size: {len(fraud_label_set):,}")
    elif args.dataset == "unsw_mg24":
        # DAG is `device → process → host → flow`, so backward trace from a
        # fraud flow_node walks through host_node / process_node / device_node.
        # Add malicious node ids on every labelled type so reaching them counts.
        n_flow_fraud = sum(1 for i in test_indices if labels[i].item() == 1)
        per_type_added: dict = {}
        for ntype in ("process_node", "measurement_node"):
            if ntype not in data.node_types or not hasattr(data[ntype], "y"):
                continue
            ntype_offset = type_offsets.get(ntype, 0)
            ntype_labels = data[ntype].y
            added = {
                ntype_offset + i for i in range(ntype_labels.size(0))
                if int(ntype_labels[i].item()) == 1
            }
            per_type_added[ntype] = len(added)
            fraud_label_set |= added
        if args.debug:
            extras = ", ".join(
                f"{t}={n:,}" for t, n in per_type_added.items()
            )
            print(f"\n[fraud_label_set] flow fraud (test): {n_flow_fraud:,}; "
                  f"added malicious nodes: {extras}; "
                  f"total fraud_label_set size: {len(fraud_label_set):,}")
 
    rct_metrics = compute_root_cause_metrics(
        predicted_roots, causal_chains, fraud_label_set
    )
    rct_metrics["num_traced"] = len(predicted_roots)
 
    # ── DIAGNOSTIC 4: root type breakdown ──────────────────────────────────
    if args.debug:
        root_type_counts = Counter()
        root_in_fraud_by_type = Counter()
        for root in predicted_roots:
            rtype = causal_graph.node_type.get(root, "unknown")
            root_type_counts[rtype] += 1
            if root in fraud_label_set:
                root_in_fraud_by_type[rtype] += 1
        print(f"\n[diagnostic 4/4] Predicted root cause type breakdown")
        for rtype in sorted(root_type_counts.keys()):
            n_total = root_type_counts[rtype]
            n_fraud = root_in_fraud_by_type[rtype]
            pct = 100 * n_fraud / max(1, n_total)
            print(f"  root type = {rtype:<20s}: {n_total:>4d}  "
                  f"({n_fraud} in fraud_label_set, {pct:.1f}%)")
    stability_metrics = {
        "phi_stability_std": float(np.std(stability_diffs)) if stability_diffs else 0.0,
        "phi_stability_mean_abs": float(np.mean(stability_diffs)) if stability_diffs else 0.0,
        "num_nodes_explained": len(fraud_predicted_global),
    }
 
    # ── DIAGNOSTIC 2: chain depth histogram ────────────────────────────────
    if args.debug:
        chain_lens = [len(c) for c in causal_chains]
        depth_hist = Counter([l - 1 for l in chain_lens])
        n = len(causal_chains)
        print(f"\n[diagnostic 2/3] Chain depth histogram "
              f"(num_traced={n}, max_hops={args.max_hops})")
        for depth in sorted(depth_hist.keys()):
            count = depth_hist[depth]
            bar = "█" * int(40 * count / max(1, n))
            print(f"  depth={depth:>2d}: {count:>4d}  {bar}")
 
    # ── DIAGNOSTIC 3: why are length-1 (depth-0) chains stuck? ─────────────
    if args.debug:
        stuck_total = 0
        no_parents = 0
        weak_ce_count = 0
        weak_ce_max_values = []
        for chain in causal_chains:
            if len(chain) > 1:
                continue
            stuck_total += 1
            target = chain[0]
            parents = list(causal_graph.parents(target))
            if not parents:
                no_parents += 1
            else:
                best_ce = max(
                    (causal_effects.get((p, target), 0.0) for p in parents),
                    default=0.0,
                )
                if best_ce < args.ce_threshold:
                    weak_ce_count += 1
                    weak_ce_max_values.append(best_ce)
        print(f"\n[diagnostic 3/3] Stuck-at-target trace analysis "
              f"({stuck_total}/{len(causal_chains)} chains have depth 0)")
        print(f"  reason: target has no parents in causal graph : {no_parents}")
        print(f"  reason: parents exist but max CE < threshold  : {weak_ce_count}")
        if weak_ce_max_values:
            arr = np.array(weak_ce_max_values)
            print(f"    -> max-CE among stuck-but-has-parents:  "
                  f"mean={arr.mean():+.4f}  median={np.median(arr):+.4f}  "
                  f"min={arr.min():+.4f}  max={arr.max():+.4f}")
            print(f"    -> if these are mostly near 0, the relevant edge model "
                  f"is undertrained (NCM hypothesis confirmed).")
 
    return rct_metrics, stability_metrics
 
 
def eval_explanation_quality(model, data, labels, test_mask,
                              causal_graph, args, gt_causal_nodes=None):
    if not gt_causal_nodes:
        print("  No ground-truth causal labels — skipping.")
        return {}
    eligible = {
        nid: gs for nid, gs in gt_causal_nodes.items()
        if nid in causal_graph.set_v
    }
    print(f"  Explanation Quality: {len(eligible)}/{len(gt_causal_nodes)} "
          f"GT transactions found in causal graph.")
    if not eligible:
        return {}
    model.eval()
    with torch.no_grad():
        logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)
    causal_effects = model.compute_causal_effects(flat_h, causal_graph)
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
 
 
def build_gt_list(args, data, type_offsets):
    gt_list = []
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
    elif args.dataset == "unsw_mg24" and hasattr(data, "_mg24"):
        print("Computing kill-chain ground-truth for Metric C (UNSW-MG24)...")
        from utils.mg24_kill_chain_gt import compute_mg24_kill_chain_gt
        test_mask = (
            data["flow_node"].test_mask.cpu().numpy()
            if hasattr(data["flow_node"], "test_mask") else None
        )
        gt = compute_mg24_kill_chain_gt(
            mg24_data=data._mg24,  # type: ignore[attr-defined]
            type_offsets=type_offsets,
            test_mask=test_mask,
            include_devices=True,
            verbose=True,
        )
        if gt:
            gt_list.append(("KillChain", gt))
    elif args.dataset == "elliptic++":
        from utils.lfpn_utils import compute_lfpn_ground_truth
        modes = ["strict", "extended"] if args.lfpn_mode == "both" else [args.lfpn_mode]
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
 
 
def main():
    args = parse_args()
    device = torch.device(args.device)
 
    print(f"Loading dataset: {args.dataset}")
    data, target_type = load_dataset(
        args.dataset, args.data_root,
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
        max_flows=args.max_flows,
        mg24_subsample_ddos=args.mg24_subsample_ddos,
        mg24_min_host_flows=args.mg24_min_host_flows,
        mg24_prune_external=args.mg24_prune_external,
        mg24_split_mode=args.mg24_split_mode,
        mg24_host_role=args.mg24_host_role,
        mg24_drop_features=args.mg24_drop_features,
        seed=args.seed,
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
 
    gt_list = build_gt_list(args, data, type_offsets)
 
    print("\nBuilding TypedCausalGraph...")
    gt_tx_ids = []
    for _, gt in gt_list:
        gt_tx_ids.extend(gt.keys())
    seed_ids = list(dict.fromkeys(
        fraud_global_ids[:args.num_seeds] + gt_tx_ids
    ))

    # B1: resolve rare-edge-type set. Explicit --rare_edge_types wins;
    # the literal string "none" disables the pass; empty string falls back
    # to the per-dataset default from default_rare_edge_types().
    raw = args.rare_edge_types.strip()
    if raw.lower() == "none":
        rare_edge_types = set()
        source = "disabled"
    elif raw:
        rare_edge_types = {tok.strip() for tok in raw.split(",") if tok.strip()}
        source = "explicit"
    else:
        rare_edge_types = default_rare_edge_types(args.dataset)
        source = "dataset-default"

    if rare_edge_types:
        print(f"  [BFS] rare edge types ({len(rare_edge_types)}, "
              f"reserve={args.rare_edge_reserve}, "
              f"max_hops={args.rare_edge_max_hops}, src={source}): "
              f"{sorted(rare_edge_types)}")

    # Symmetric resolution for blocked_edge_types (replaces the legacy
    # block_addr_to_addr param; Elliptic-style wallet/address self-loops
    # come from default_blocked_edge_types).
    raw_b = args.blocked_edge_types.strip()
    if raw_b.lower() == "none":
        blocked_edge_types = set()
        blocked_src = "disabled"
    elif raw_b:
        blocked_edge_types = {tok.strip() for tok in raw_b.split(",") if tok.strip()}
        blocked_src = "explicit"
    else:
        blocked_edge_types = default_blocked_edge_types(args.dataset)
        blocked_src = "dataset-default"
    if blocked_edge_types:
        print(f"  [BFS] blocked edge types ({len(blocked_edge_types)}, "
              f"src={blocked_src}): {sorted(blocked_edge_types)}")

    causal_graph = build_typed_causal_graph_from_hetero(
        data,
        seed_node_ids=seed_ids if seed_ids else None,
        hop_limit=args.hop_limit,
        node_limit=args.node_limit,
        blocked_edge_types=blocked_edge_types if blocked_edge_types else None,
        rare_edge_types=rare_edge_types if rare_edge_types else None,
        rare_reserve=args.rare_edge_reserve,
        rare_max_hops=args.rare_edge_max_hops,
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
        use_gan=False,
    ).to(device)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — evaluating randomly initialised model (baseline).")
 
    cls_metrics = eval_classification(model, data, labels, test_mask)
    print_section("A. Classification Metrics", cls_metrics)
 
    rct_metrics, stab_metrics = eval_root_cause_and_stability(
        model, data, labels, test_mask, causal_graph, args, device,
        type_offsets=type_offsets, target_type=target_type,
    )
    if rct_metrics:
        print_section("B. Root Cause Tracing Metrics", rct_metrics)
    if stab_metrics:
        print_section("D. φ-Stability Metrics", stab_metrics)
 
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