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
  - other       : Metric C is skipped.

Usage:
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
    compute_fraud_f1,
    compute_root_cause_metrics,
)
from utils.threshold_utils import sweep_best_threshold
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CI-RCT model")
    parser.add_argument("--dataset", type=str, default="elliptic++",
                        choices=["elliptic++"])
    # Elliptic++ detection target: 'transaction' (default, unchanged), 'wallet',
    # or 'joint' (one model classifying both → single pooled F1 + dual-seed
    # tracing). wallet/joint require --dataset elliptic++.
    parser.add_argument("--variant", type=str, default="transaction",
                        choices=["transaction", "wallet", "joint"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--ce_threshold", type=float, default=0.1)
    # Type-aware tie-break for the tracer (opt-in; empty string = legacy
    # |CE|-only ranking). Comma-separated node types that are "root-capable"
    # (labelable malicious types). When set, the greedy search prefers
    # climbing to these over same-type relay hops (e.g. the host→host bridge),
    # recovering RCP on models whose CE landscape diverts the trace to host.
    # Example: --prefer_root_types process_node,measurement_node
    parser.add_argument("--prefer_root_types", type=str, default="")
    # DD-18 LOOKAHEAD tie-break depth. prefer_root_types alone only inspects
    # the immediate upstream's type; on MG24 a fraud flow's hub host has an
    # all-host upstream (bridge edges), so the greedy |CE|-max dead-ends at a
    # 0-parent bridge host while the branch that actually leads to a process
    # has marginally smaller |CE|. With d > 0, among threshold-passing upstream
    # candidates the tracer prefers those that can REACH a prefer_root_types
    # node within d backward hops. 0 = disabled (legacy). Needs prefer_root_types.
    # Example: --prefer_root_types process_node --prefer_reachable_depth 3
    parser.add_argument("--prefer_reachable_depth", type=int, default=0)
    # Tracer algorithm ablation. greedy = legacy byte-identical; the rest are
    # comparison arms (model/tracer_strategies, tracer_ablation_plan.md). All
    # arms share the SAME causal_effects + graph + threshold + max_hops, so only
    # the search rule differs — that is the controlled variable of the ablation.
    parser.add_argument(
        "--tracer_algorithm", type=str, default="greedy",
        choices=["greedy", "beam", "dag_dp", "dijkstra", "bfs", "dfs"],
        help="Root-cause backward-search algorithm. greedy=legacy byte-identical; "
             "dag_dp=recommended global-optimal DAG Viterbi.",
    )
    parser.add_argument(
        "--tracer_objective", type=str, default="product",
        choices=["product", "sum"],
        help="Weighted-path objective for dag_dp/dijkstra: product (max-product, "
             "cost=-log|CE|) or sum (max-sum |CE|).",
    )
    parser.add_argument(
        "--ncm_baseline", type=str, default="zero",
        choices=["zero", "type_mean", "marginal"],
        help="CE null-intervention baseline. 'zero'=legacy do(h_u=0) (OOD, "
             "saturates p_null → CE sign uninterpretable); 'type_mean'="
             "do(h_u=E[h_type]) recentres CE so sign=promote(+)/suppress(−); "
             "'marginal'=p_null=E[MLP(h)] over same-type sources (no Jensen "
             "gap, E[CE] per type is exactly 0). "
             "No retraining needed — same weights, different CE reference.",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--node_limit", type=int, default=5000)
    parser.add_argument("--hop_limit", type=int, default=2)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--max_explain", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    # NOTE: for v2 checkpoints this flag is a FALLBACK only — the layer count
    # is read back from the checkpoint's embedded arch metadata and overrides
    # this value (see main()'s arch_get). It matters only for legacy
    # bare-state_dict checkpoints, where it MUST match the training recipe:
    # train.py's default is 3, so a legacy checkpoint trained on defaults but
    # evaluated with the old default 2 left the extra HGT layer randomly
    # initialised (strict=False load), giving correct AUC but F1≈0. Default
    # raised to 3 to match train.py and make that fallback safe-by-default.
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--type_emb_dim", type=int, default=16)
    parser.add_argument("--include_addr_addr", type=lambda x: x.lower() == "true",
                        default=False)
    parser.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true",
                        default=False)
    parser.add_argument("--fraud_subgraph_hops", type=int, default=2)
    # ── Decision-threshold tuning (binary detection; no retrain needed) ──────
    # 'none' keeps the legacy argmax (==0.5) cut. 'val' sweeps a threshold on
    # the validation split to maximise --threshold_objective, then applies it
    # to the test split (test distribution never leaks into the choice).
    # --threshold >= 0 overrides the sweep with a fixed manual cut.
    parser.add_argument("--threshold_tuning", type=str, default="none",
                        choices=["none", "val"])
    parser.add_argument("--threshold_objective", type=str, default="macro_f1",
                        choices=["macro_f1", "fraud_f1"])
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="Manual class-1 probability cut in (0,1); "
                             "overrides --threshold_tuning when >= 0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lfpn_mode", type=str, default="both",
                        choices=["strict", "extended", "both"])
    parser.add_argument("--lfpn_k", type=int, default=2)
    # Metric C "groundtruth match": the CXGNN-original "exact" requires the
    # traced chain set to EQUAL the GT set, which is structurally impossible
    # here (the chain always contains the queried tx, the LFPN GT is wallet-
    # only) → always 0. "subset" instead asks whether the chain recovered ALL
    # GT nodes (per-instance perfect recall). Default subset; pass exact to
    # reproduce the CXGNN metric.
    parser.add_argument("--gt_match_mode", type=str, default="subset",
                        choices=["exact", "subset"])
    # ── Metric C explainer (ablation route A) ────────────────────────────────
    # Which explainer produces the per-fraud-node explanatory set scored by
    # Metric C. 'ce_only' (default) is the legacy raw-|CE| greedy trace — the
    # φ machinery is bypassed, byte-identical to the prior Metric C. The φ
    # variants rank each hop's parents by Causal Shapley computed from the
    # backbone do-intervention coalition value (non-additive), so 'asym' vs
    # 'sym' isolates the empirical value of temporal asymmetry.
    parser.add_argument("--explainer", type=str, default="ce_only",
                        choices=["ce_only", "phi_asym", "phi_sym",
                                 "saliency"])
    parser.add_argument("--shapley_permutations", type=int, default=64,
                        help="Monte-Carlo permutations for symmetric Shapley "
                             "(--explainer phi_sym). Exact enumeration is used "
                             "when n_parents! <= this value.")
    parser.add_argument("--shapley_topk", type=int, default=0,
                        help="Cap each hop's parents to the top-k by |CE| before "
                             "Causal Shapley (phi_asym/phi_sym). 0 = no cap "
                             "(legacy). Each coalition is a full backbone forward, "
                             "so high-in-degree nodes make phi intractable on "
                             "Elliptic++; e.g. --shapley_topk 8 bounds the count.")
    parser.add_argument("--coalition_subgraph",
                        type=lambda x: x.lower() == "true", default=True,
                        help="Forward only the readout's L-hop receptive-field "
                             "subgraph per Causal Shapley coalition instead of the "
                             "full graph (numerically identical for a local "
                             "backbone; self-checks vs full forward and reverts on "
                             "mismatch). Massive phi_asym/phi_sym speedup. "
                             "true (default) / false.")
    # Ranking signal for the MAIN root-cause tracer (Metric B / RCP), as opposed
    # to --explainer which only changes Metric C. 'ce' is byte-identical legacy.
    parser.add_argument("--tracer_score", type=str, default="ce",
                        choices=["ce", "ce_signed", "phi_asym", "phi_sym"],
                        help="Per-hop ranking signal for the MAIN root-cause "
                             "tracer (Metric B). 'ce' (default) ranks by |CE| "
                             "(byte-identical to legacy). 'phi_asym'/'phi_sym' "
                             "rank by |φ| (asymmetric/symmetric Causal Shapley via "
                             "the backbone do-intervention coalition value), making "
                             "the title's Shapley-driven-tracing claim testable on "
                             "RCP. Reuses --shapley_topk / --shapley_permutations / "
                             "--coalition_subgraph. φ readout uses the primary head, "
                             "so non-primary-type (joint) seeds fall back to |CE|. "
                             "ce_signed=以原始有號 CE 排序,只追正向 promoter"
                             "(配 type_mean baseline 使用)。")
    # Dump the exact set of traced root-cause chains (the same ones counted in
    # the depth histogram / num_traced) decoded to real Elliptic++ identities,
    # for the crime-chain viewer (viz/crime_chain*.html).
    parser.add_argument("--dump_chains", type=str, default=None,
                        help="JSON path to write the traced chains (with real "
                             "txId / wallet address) for the crime-chain viewer.")
    parser.add_argument("--dump_chains_topn", type=int, default=0,
                        help="Keep only the top-N chains (0 = all), sorted "
                             "true-positive & fraud-root & deepest first.")
    parser.add_argument("--dump_csv", type=str, default=None,
                        help="CSV path to write the traced chains as a flat "
                             "one-row-per-chain table (path / type / CE encoded "
                             "with '|'); shares the same chains & top-N filter "
                             "as --dump_chains.")
    parser.add_argument("--dump_phi", action="store_true",
                        help="Attach per-node Causal Shapley φ (causal "
                             "responsibility) to the dumped chains: phi_add "
                             "(additive CE/n) and phi_asym (true asymmetric "
                             "Shapley via the backbone coalition value). "
                             "phi_asym needs coalition forwards — slow on many "
                             "chains; tune with --shapley_topk / --max_explain.")
    parser.add_argument("--dump_feature_attribution", action="store_true",
                        help="L3: attach causal feature attribution to each "
                             "chain's φ_asym pivot node (per-feature "
                             "do-intervention CFE; saliency fallback). Requires "
                             "--dump_phi. Adds one forward/backward + a few "
                             "do-passes per chain.")
    parser.add_argument("--feat_attr_topk", type=int, default=12,
                        help="L3: number of features to report per pivot node.")
    parser.add_argument("--feat_attr_all_nodes", action="store_true",
                        help="L3: attribute EVERY chain node, not only the φ_asym "
                             "pivot. Nodes outside the target's num_layers-hop "
                             "receptive field still return empty (cannot reach the "
                             "readout). Trades compute for near-target coverage.")
    parser.add_argument("--debug", action="store_true",
                        help="Print tracer diagnostics: CE distribution by edge "
                             "type, chain length histogram, stuck-trace analysis.")
    parser.add_argument("--stability_probe_only", action="store_true",
                        help="Skip LFPN/tracing/Metric-C and run ONLY the φ "
                             "perturbation-stability probe (a σ-sweep of input "
                             "noise, K draws each). Fast comparison of L_stab's "
                             "effect across checkpoints (Full vs no_stab).")
    parser.add_argument("--phi_noise_sweep", type=str,
                        default="0.01,0.05,0.1,0.2,0.5",
                        help="Comma-separated σ values for --stability_probe_only: "
                             "Gaussian noise std added to node embeddings before "
                             "recomputing φ. Larger σ probes harder.")
    parser.add_argument("--phi_noise_draws", type=int, default=5,
                        help="Number of independent noise draws averaged per σ in "
                             "--stability_probe_only (reduces estimator variance).")
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["test", "val"],
                        help="Which split to compute Metric A/B/C/D on. "
                             "Default 'test' (report numbers). Use 'val' for "
                             "hyperparameter search so the held-out test set is "
                             "never touched during tuning. Decision thresholds "
                             "are always tuned on val regardless.")
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
    if name == "elliptic++":
        from utils.elliptic_plus_loader import load_elliptic_plus_dataset
        return load_elliptic_plus_dataset(
            os.path.join(root, "Elliptic++"),
            include_addr_addr=kwargs.get("include_addr_addr", False),
            fraud_subgraph=kwargs.get("fraud_subgraph", False),
            fraud_subgraph_hops=kwargs.get("fraud_subgraph_hops", 2),
        )
    raise ValueError(f"Unknown dataset: {name!r}")
 
 
def _parse_prefer_root_types(raw: str):
    """Parse --prefer_root_types CSV into a set, or None when empty (legacy)."""
    types = {t.strip() for t in (raw or "").split(",") if t.strip()}
    return types or None


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
    wallet_per_address: bool = False,
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
            wallet_per_address=wallet_per_address,
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
def eval_classification(model, data, labels, test_mask, threshold=None):
    """
    Binary classification metrics on the test split.

    threshold: if None, use argmax (== 0.5 cut, legacy). If a float in (0, 1),
    predict fraud where P(class-1) > threshold. AUC is threshold-independent
    (computed from scores) so it is unaffected.
    """
    model.eval()
    logits, _ = model.forward(data)
    scores = torch.softmax(logits[test_mask], dim=-1)[:, 1].cpu()
    if threshold is None:
        preds = logits[test_mask].argmax(dim=-1).cpu()
    else:
        preds = (scores > threshold).long()
    y_true = labels[test_mask].cpu()
    metrics = compute_classification_metrics(preds, y_true, scores)
    metrics["fraud_f1"] = compute_fraud_f1(y_true, preds)
    metrics["recall_fraud"] = float(
        recall_score(y_true.numpy(), preds.numpy(), pos_label=1, zero_division=0)
    )
    # Fraction predicted fraud — a degenerate threshold (≈1.0 → "everything is
    # fraud", or ≈0.0 → "nothing is fraud") shows up here immediately and
    # explains a collapsed macro F1.
    metrics["pred_fraud_rate"] = float(preds.float().mean())
    if threshold is not None:
        metrics["threshold"] = float(threshold)
    return metrics


def _load_variant_dataset(args):
    """Dispatch the loader by --variant (elliptic++ only for wallet/joint)."""
    if args.variant == "transaction":
        return load_dataset(
            args.dataset, args.data_root,
            include_addr_addr=args.include_addr_addr,
            fraud_subgraph=args.fraud_subgraph,
            fraud_subgraph_hops=args.fraud_subgraph_hops,
            seed=args.seed,
        )
    if args.dataset != "elliptic++":
        raise ValueError(
            f"--variant {args.variant} requires --dataset elliptic++ "
            f"(got {args.dataset!r})."
        )
    root = os.path.join(args.data_root, "Elliptic++")
    if args.variant == "wallet":
        from utils.elliptic_plus_wallet_loader import load_elliptic_plus_wallet_dataset
        return load_elliptic_plus_wallet_dataset(
            root, include_addr_addr=args.include_addr_addr,
            fraud_subgraph=args.fraud_subgraph,
            fraud_subgraph_hops=args.fraud_subgraph_hops,
        )
    from utils.elliptic_plus_joint_loader import load_elliptic_plus_joint_dataset
    return load_elliptic_plus_joint_dataset(
        root, include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )


def _tune_head_threshold(probs, store, objective):
    """Per-head decision threshold, swept on THIS type's own val split.

    Returns the swept threshold, or None to fall back to argmax (when no
    objective is requested or the type has no val nodes).
    """
    if objective is None:
        return None
    vmask = getattr(store, "val_mask", None)
    if vmask is None or not bool(vmask.any()):
        return None
    v_scores = probs[vmask][:, 1].cpu().numpy()
    v_true = store.y[vmask].cpu().numpy()
    thr, _ = sweep_best_threshold(v_scores, v_true, objective=objective)
    return thr


@torch.no_grad()
def eval_classification_pooled(model, data, threshold_objective=None,
                              mask_attr="test_mask"):
    """Joint variant: ONE pooled F1 over every classified type's test nodes.

    Each type is scored by its OWN head (primary for transaction, aux head for
    wallet); the (y_true, y_pred, score) triples are concatenated and a single
    set of metrics is computed.

    When ``threshold_objective`` is given (e.g. 'fraud_f1'), each head's
    decision threshold is tuned INDEPENDENTLY on that type's own val split
    before pooling. This is essential because the heads have very different
    fraud base rates: a single shared argmax (== 0.5) cut lets the high-volume
    wallet head over-predict (pred_fraud_rate ≫ true rate), which collapses the
    pooled fraud F1. ``None`` keeps the legacy argmax cut.

    Returns (metrics, per_type_test_n, per_type_info) where per_type_info maps
    each type → {fraud_f1, pred_fraud_rate, threshold}.
    """
    model.eval()
    logits_by_type, _ = model.all_logits(data)
    y_true, y_pred, scores = [], [], []
    per_type_n, per_type_info = {}, {}
    for ntype, logits in logits_by_type.items():
        mask = getattr(data[ntype], mask_attr, None)
        if mask is None or not bool(mask.any()):
            continue
        probs = torch.softmax(logits, dim=-1)
        t_scores = probs[mask][:, 1].cpu()
        t_true = data[ntype].y[mask].cpu()

        thr = _tune_head_threshold(probs, data[ntype], threshold_objective)
        t_pred = (
            (t_scores > thr).long() if thr is not None
            else probs[mask].argmax(dim=-1).cpu()
        )

        per_type_n[ntype] = int(mask.sum())
        per_type_info[ntype] = {
            "fraud_f1": compute_fraud_f1(t_true, t_pred),
            "pred_fraud_rate": float(t_pred.float().mean()),
            "threshold": 0.5 if thr is None else float(thr),
        }
        y_true.append(t_true)
        y_pred.append(t_pred)
        scores.append(t_scores)

    y_true, y_pred, scores = map(torch.cat, (y_true, y_pred, scores))
    metrics = compute_classification_metrics(y_pred, y_true, scores)
    metrics["fraud_f1"] = compute_fraud_f1(y_true, y_pred)
    metrics["pred_fraud_rate"] = float(y_pred.float().mean())
    return metrics, per_type_n, per_type_info


def phi_stability_probe(model, data, causal_graph, args, device,
                        type_offsets, target_type, extra_fraud_seeds=None):
    """φ perturbation-stability probe (Metric D, strengthened).

    The legacy Phi-Stability metric perturbs embeddings at a single small σ
    (0.01) and averages |Δφ| over ALL parents — dominated by near-zero φ, it
    saturates and cannot distinguish an L_stab-trained model from an ablated
    one. This probe instead (i) sweeps σ so degradation shows as a curve,
    (ii) averages K independent noise draws to cut estimator variance, and
    (iii) reports drift at each chain's PIVOT (argmax|φ|) — the node that
    actually carries the explanation — both absolute and relative to |φ|.

    A model regularised by L_stab should keep pivot drift flatter as σ grows.
    """
    model.eval()
    with torch.no_grad():
        _logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)
    causal_effects = model.compute_causal_effects(flat_h, causal_graph)

    # Same predicted-fraud seed selection as the main Metric-B path.
    offset = type_offsets[target_type]
    logits, _ = model.all_logits(data)
    tgt_logits = logits[target_type] if isinstance(logits, dict) else logits
    pred = tgt_logits.argmax(dim=-1)
    seeds = [
        offset + idx for idx in range(pred.size(0))
        if pred[idx].item() == 1 and (offset + idx) in causal_graph.set_v
    ]
    if extra_fraud_seeds:
        seeds.extend(g for g in extra_fraud_seeds if g in causal_graph.set_v)
    seeds = list(dict.fromkeys(seeds))[: args.max_explain]
    if not seeds:
        print("  [φ-probe] No fraud-predicted seeds in causal graph — skipping.")
        return

    # Original φ and its pivot (argmax|φ|) per seed.
    phi_orig = {s: compute_asymmetric_causal_shapley(causal_effects, causal_graph, s)
                for s in seeds}
    pivots = {}
    for s, phi in phi_orig.items():
        if phi:
            pivots[s] = max(phi.items(), key=lambda kv: abs(kv[1]))[0]

    sweep = [float(x) for x in str(args.phi_noise_sweep).split(",") if x.strip()]
    K = max(1, int(args.phi_noise_draws))
    eps = 1e-8

    print(f"\n[φ-stability probe] σ-sweep over {len(sweep)} levels, "
          f"K={K} draws, N={len(seeds)} seeds, "
          f"{len(pivots)} with a defined pivot")
    print(f"  {'σ':>6s}  {'all-parent mean|Δφ|':>20s}  "
          f"{'pivot mean|Δφ|':>16s}  {'pivot rel-drift':>16s}")

    rows = []
    for sigma in sweep:
        all_diffs, pivot_abs, pivot_rel = [], [], []
        for _ in range(K):
            with torch.no_grad():
                flat_h_pert = {
                    gid: emb + torch.randn_like(emb) * sigma
                    for gid, emb in flat_h.items()
                }
            ce_pert = model.compute_causal_effects(flat_h_pert, causal_graph)
            for s in seeds:
                phi_p = compute_asymmetric_causal_shapley(ce_pert, causal_graph, s)
                phi_o = phi_orig[s]
                for p in set(phi_o) & set(phi_p):
                    all_diffs.append(abs(phi_o[p] - phi_p[p]))
                piv = pivots.get(s)
                if piv is not None and piv in phi_p:
                    d = abs(phi_o[piv] - phi_p[piv])
                    pivot_abs.append(d)
                    pivot_rel.append(d / (abs(phi_o[piv]) + eps))
        m_all = float(np.mean(all_diffs)) if all_diffs else 0.0
        m_piv = float(np.mean(pivot_abs)) if pivot_abs else 0.0
        m_rel = float(np.mean(pivot_rel)) if pivot_rel else 0.0
        rows.append((sigma, m_all, m_piv, m_rel))
        print(f"  {sigma:>6.3f}  {m_all:>20.6f}  {m_piv:>16.6f}  {m_rel:>16.6f}")

    return rows


def eval_root_cause_and_stability(model, data, labels, test_mask,
                                  causal_graph, args, device,
                                  type_offsets, target_type,
                                  extra_fraud_seeds=None):
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
        prefer_root_types=_parse_prefer_root_types(args.prefer_root_types),
        prefer_reachable_depth=args.prefer_reachable_depth,
        tracer_algorithm=args.tracer_algorithm,
        tracer_objective=args.tracer_objective,
    )
 
    offset = type_offsets[target_type]
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_predicted_global = [
        offset + idx for idx in test_indices
        if logits[idx].argmax().item() == 1
           and (offset + idx) in causal_graph.set_v
    ]
    # Joint dual-seed: also trace from predicted-fraud nodes of other types
    # (e.g. wallet head), computed by the caller and passed in here.
    if extra_fraud_seeds:
        fraud_predicted_global.extend(
            g for g in extra_fraud_seeds if g in causal_graph.set_v
        )
        fraud_predicted_global = list(dict.fromkeys(fraud_predicted_global))
    fraud_predicted_global = fraud_predicted_global[: args.max_explain]
 
    if not fraud_predicted_global:
        print("  No fraud-predicted test nodes in causal graph — skipping RCT metrics.")
        return {}, {}
 
    predicted_roots, causal_chains = [], []
    _noise_sigma = getattr(
        getattr(model, "config", None), "phi_stability_noise_std", 0.01
    )
    with torch.no_grad():
        flat_h_perturbed = {
            gid: emb + torch.randn_like(emb) * _noise_sigma
            for gid, emb in flat_h.items()
        }
    causal_effects_perturbed = model.compute_causal_effects(flat_h_perturbed, causal_graph)
    stability_diffs = []

    # ── Optional φ-driven ranking for the MAIN tracer (Metric B / RCP) ─────────
    # --tracer_score ce (default) ⇒ |CE| ranking, byte-identical to legacy.
    # ce_signed ⇒ rank each hop by the RAW signed CE (no abs), so max() picks the
    # strongest positive promoter and negative (suppressor) parents are never
    # followed — pair with a type_mean baseline so the sign is meaningful.
    # phi_asym/phi_sym rank each hop by |φ| (asymmetric/symmetric Causal Shapley
    # via the backbone do-intervention coalition value), making the thesis title's
    # "asymmetric-Shapley-driven root-cause tracing" claim testable on RCP. We take
    # |φ| (not signed φ) to keep the magnitude convention of the |CE| path, so
    # suppressor parents (negative effect, large |φ|) are still followed and the
    # trace does not stall at depth 0.
    tracer_score = getattr(args, "tracer_score", "ce")
    readout_type = model.backbone.target_node_type
    _phi_layers = None
    if tracer_score not in ("ce", "ce_signed") and getattr(args, "coalition_subgraph", True):
        try:
            _phi_layers = len(model.backbone.hgt_layers)
        except AttributeError:
            _phi_layers = getattr(
                getattr(model, "config", None), "num_hgt_layers", None
            )

    def _phi_tracer_score_fn(seed_global):
        # φ readout reuses the primary classifier head, so it is only valid for
        # seeds of that type; non-primary (joint) seeds fall back to |CE| (None).
        if tracer_score == "ce":
            return None
        if tracer_score == "ce_signed":
            # Raw signed CE: max() over upstream picks the strongest positive
            # promoter; negative (suppressor) parents are never selected. Valid
            # for any node type, so it precedes the primary-head readout check.
            def signed_ce_fn(current, upstream):
                return {u: causal_effects.get((u, current), 0.0) for u in upstream}
            return signed_ce_fn
        if causal_graph.node_type.get(seed_global) != readout_type:
            return None
        from model.explainers import _make_phi_score_fn
        base = _make_phi_score_fn(
            model=model, data=data, causal_graph=causal_graph,
            target_node=seed_global, type_offsets=type_offsets,
            target_node_type=readout_type, fraud_class=1,
            mode=("asym" if tracer_score == "phi_asym" else "sym"),
            n_permutations=getattr(args, "shapley_permutations", 64),
            causal_effects=causal_effects,
            shapley_topk=(getattr(args, "shapley_topk", 0) or None),
            use_subgraph=getattr(args, "coalition_subgraph", True),
            num_layers=_phi_layers,
        )

        def abs_score_fn(current, upstream):
            return {u: abs(v) for u, v in base(current, upstream).items()}

        return abs_score_fn

    for global_id in fraud_predicted_global:
        root, chain = tracer.trace_root_cause(
            global_id, causal_effects,
            upstream_score_fn=_phi_tracer_score_fn(global_id),
        )
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
            wallet_per_address=args.variant in ("wallet", "joint"),
        )
        fraud_label_set |= illicit_wallet_globals
        # wallet/joint (dual-seed) traces can legitimately end at an upstream
        # illicit transaction; credit those too. Gated so the transaction
        # variant's fraud_label_set stays unchanged.
        if (target_type != "transaction" or extra_fraud_seeds) \
                and "transaction" in type_offsets and hasattr(data["transaction"], "y"):
            tx_off = type_offsets["transaction"]
            tx_y = data["transaction"].y
            fraud_label_set |= {
                tx_off + i
                for i in (tx_y == 1).nonzero(as_tuple=True)[0].tolist()
            }
        if args.debug:
            print(f"\n[fraud_label_set] tx fraud nodes: "
                  f"{sum(1 for i in test_indices if labels[i].item() == 1)}, "
                  f"illicit wallets added: {len(illicit_wallet_globals):,}, "
                  f"total fraud_label_set size: {len(fraud_label_set):,}")
 
    rct_metrics = compute_root_cause_metrics(
        predicted_roots, causal_chains, fraud_label_set
    )
    rct_metrics["num_traced"] = len(predicted_roots)

    # ── True-positive-only RCP ──────────────────────────────────────────────
    # The headline RCP above is computed over every *predicted* fraud target,
    # so a classifier false positive (a licit tx wrongly flagged) — which has
    # no real fraud trail and can never trace to a fraud root — counts as a
    # miss and depresses the tracer's measured precision. Restricting to
    # targets that are ACTUALLY illicit (label==1) isolates tracer quality
    # from classifier precision. predicted_roots is aligned 1-to-1 with
    # fraud_predicted_global, so we can recover each target's true label.
    from utils.metrics import root_cause_precision

    def _label_of_global(gid):
        """True label of a global id via its own node type (handles mixed-type
        joint seeds). For single-type variants this equals labels[gid-offset]."""
        chosen_t, chosen_off = None, -1
        for t, off in type_offsets.items():
            if off <= gid and off > chosen_off:
                chosen_t, chosen_off = t, off
        if chosen_t is None:
            return None
        y = getattr(data[chosen_t], "y", None)
        local = gid - chosen_off
        if y is None or local < 0 or local >= y.size(0):
            return None
        return int(y[local].item())

    tp_roots = [
        root
        for root, gid in zip(predicted_roots, fraud_predicted_global)
        if _label_of_global(gid) == 1
    ]
    if tp_roots:
        rct_metrics["root_cause_precision_true_pos"] = float(
            np.mean([root_cause_precision(r, fraud_label_set) for r in tp_roots])
        )
        rct_metrics["num_true_pos_traced"] = len(tp_roots)

    # ── RCP diagnostics: label-coverage ceiling & split-artifact check ─────
    # (a) A root landing on an UNKNOWN-label node scores 0 no matter what the
    #     tracer did — on Elliptic++ (mostly class-3 nodes) this caps RCP well
    #     below 1. root_labeled_ratio IS that ceiling; rcp_labeled_only is RCP
    #     restricted to chains whose root can actually be judged.
    root_labels = [_label_of_global(r) for r in predicted_roots]
    labeled_pairs = [
        (r, y) for r, y in zip(predicted_roots, root_labels) if y in (0, 1)
    ]
    rct_metrics["root_labeled_ratio"] = len(labeled_pairs) / len(predicted_roots)
    if labeled_pairs:
        rct_metrics["root_cause_precision_labeled_only"] = float(
            np.mean([
                root_cause_precision(r, fraud_label_set)
                for r, _ in labeled_pairs
            ])
        )
    # (b) The headline fraud_label_set only credits TEST-split fraud txs on
    #     the transaction variant (the 735 gate keeps it byte-identical), so a
    #     chain that correctly traces back to a train/val-split fraud tx still
    #     scores 0 — a split artifact: root-ness is a graph property, not a
    #     split property. Report the all-labeled-fraud-tx RCP alongside
    #     (idempotent on wallet/joint, where all fraud txs are already added).
    if args.dataset == "elliptic++" and "transaction" in type_offsets \
            and hasattr(data["transaction"], "y"):
        _tx_off = type_offsets["transaction"]
        _tx_y = data["transaction"].y
        all_fraud_set = fraud_label_set | {
            _tx_off + i for i in (_tx_y == 1).nonzero(as_tuple=True)[0].tolist()
        }
        rct_metrics["root_cause_precision_all_fraud_tx"] = float(
            np.mean([root_cause_precision(r, all_fraud_set)
                     for r in predicted_roots])
        )

    # ── Optional: dump the traced chains with real Elliptic++ identities ────
    # These are the SAME chains summarised by the depth histogram — here we
    # decode every node's global id back to its real txId / wallet address and
    # write them out as JSON (crime-chain viewer) and/or a flat CSV table.
    want_dump = (
        (getattr(args, "dump_chains", None) or getattr(args, "dump_csv", None))
        and args.dataset == "elliptic++"
    )
    if want_dump:
        import json as _json
        from utils.elliptic_identity import build_reverse_maps, chain_to_record
        from utils.chain_export import write_chains_csv
        print(f"\n[dump] decoding {len(causal_chains)} chains to real "
              f"txId / address …")
        idx_to_txid, idx_to_addr = build_reverse_maps(
            args.data_root,
            include_addr_addr=args.include_addr_addr,
            fraud_subgraph=args.fraud_subgraph,
            fraud_subgraph_hops=args.fraud_subgraph_hops,
            wallet_per_address=args.variant in ("wallet", "joint"),
        )
        records = [
            chain_to_record(
                chain, causal_effects, _label_of_global(gid) == 1,
                type_offsets, causal_graph, data, idx_to_txid, idx_to_addr,
            )
            for chain, gid in zip(causal_chains, fraud_predicted_global)
        ]
        # true-positive & fraud-root & deepest first (most informative on top)
        records.sort(
            key=lambda c: (c["is_true_positive"], c["root_is_fraud"], c["depth"]),
            reverse=True,
        )
        if args.dump_chains_topn > 0:
            records = records[: args.dump_chains_topn]
        if getattr(args, "dump_phi", None):
            from utils.chain_phi import attach_phi_to_records
            from model.explainers import _make_phi_score_fn
            phi_readout_type = model.backbone.target_node_type
            try:
                phi_num_layers = len(model.backbone.hgt_layers)
            except AttributeError:
                phi_num_layers = getattr(
                    getattr(model, "config", None), "num_hgt_layers", None
                )

            def _asym_phi_fn(readout_global, intervene_global):
                # Option A — per-hop rolling readout. chain_phi resolves the
                # readout node (the intervene node itself, or its nearest
                # downstream head node) and passes it here. v(S) then reads out
                # fraud probability at `readout_global` while the coalition
                # controls `intervene_global`'s parent edges, making φ_asym a
                # per-hop LOCAL causal responsibility instead of a global
                # attribution to the fixed seed — deep hops are no longer forced
                # to φ≈0 by the backbone's receptive-field horizon.
                base = _make_phi_score_fn(
                    model=model, data=data, causal_graph=causal_graph,
                    target_node=readout_global, type_offsets=type_offsets,
                    target_node_type=phi_readout_type, fraud_class=1, mode="asym",
                    n_permutations=getattr(args, "shapley_permutations", 64),
                    causal_effects=causal_effects,
                    shapley_topk=(getattr(args, "shapley_topk", 0) or None),
                    use_subgraph=getattr(args, "coalition_subgraph", True),
                    num_layers=phi_num_layers,
                )
                return base(intervene_global, None)

            print(f"[dump_phi] computing per-node φ (phi_add + phi_asym) for "
                  f"{len(records)} chains — asym uses coalition forwards, may be "
                  f"slow …", flush=True)
            _phi_log_every = max(1, len(records) // 40)

            def _phi_progress(done, total):
                if done % _phi_log_every == 0 or done == total:
                    print(f"[dump_phi]   {done}/{total} chains", flush=True)

            with torch.no_grad():
                records = attach_phi_to_records(
                    records, causal_graph=causal_graph,
                    causal_effects=causal_effects,
                    asym_phi_fn=_asym_phi_fn,
                    readout_type=phi_readout_type,
                    on_progress=_phi_progress,
                )

            # ── L3: causal feature attribution on each chain's φ_asym pivot ──
            if getattr(args, "dump_feature_attribution", None):
                from utils.feature_names import get_feature_names
                from model.feature_attribution import compute_causal_feature_attribution

                feat_names = get_feature_names(
                    os.path.join(args.data_root, "Elliptic++"),
                    wallet_per_address=args.variant in ("wallet", "joint"),
                )
                print(f"[dump_phi] L3: causal feature attribution on pivots of "
                      f"{len(records)} chains …", flush=True)
                attr_all = getattr(args, "feat_attr_all_nodes", False)
                n_attr = 0
                for ri, rec in enumerate(records):
                    nodes = rec.get("nodes", [])
                    if not nodes:
                        continue
                    # L3's readout reuses the PRIMARY classifier head, so it is
                    # only defined for chains whose target IS that type. Joint
                    # wallet-seeded chains must be skipped (mirrors the φ-tracer
                    # readout-type guard) — their global id would otherwise be
                    # misread as a transaction-local index (IndexError).
                    if causal_graph.node_type.get(nodes[0]["global"]) \
                            != phi_readout_type:
                        continue
                    # Nodes to attribute: every chain node when --feat_attr_all_nodes,
                    # else just the φ_asym pivot (node with peak |φ_asym|). Nodes
                    # outside the target's receptive field return empty features and
                    # are left unattributed (they cannot reach the readout).
                    if attr_all:
                        candidates = nodes
                    else:
                        pivot = None
                        best = 0.0
                        for nd in nodes:
                            pa = nd.get("phi_asym")
                            if pa is not None and abs(pa) > best:
                                best = abs(pa)
                                pivot = nd
                        candidates = [pivot] if pivot is not None else []
                    for nd in candidates:
                        attr = compute_causal_feature_attribution(
                            model=model, data=data, causal_graph=causal_graph,
                            target_node=nodes[0]["global"], pivot_node=nd["global"],
                            type_offsets=type_offsets, target_node_type=phi_readout_type,
                            feature_names=feat_names, fraud_class=1,
                            use_subgraph=getattr(args, "coalition_subgraph", True),
                            num_layers=phi_num_layers, top_k=args.feat_attr_topk,
                        )
                        if attr["features"]:
                            nd["feature_attribution"] = attr["features"]
                            nd["feature_attribution_method"] = attr["method"]
                            n_attr += 1
                    if (ri + 1) % _phi_log_every == 0 or ri + 1 == len(records):
                        _unit = "nodes" if attr_all else "pivots"
                        print(f"[dump_phi]   L3 {ri + 1}/{len(records)} "
                              f"({n_attr} {_unit} attributed)", flush=True)
        if getattr(args, "dump_chains", None):
            meta = {
                "dataset": "elliptic++",
                "checkpoint": os.path.basename(args.checkpoint or ""),
                "n_chains": len(records),
                "n_true_positive": sum(1 for c in records if c["is_true_positive"]),
                "n_fraud_root": sum(1 for c in records if c["root_is_fraud"]),
                "mean_depth": (round(float(np.mean([c["depth"] for c in records])), 2)
                               if records else 0.0),
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.dump_chains)), exist_ok=True)
            with open(args.dump_chains, "w") as f:
                _json.dump({"meta": meta, "chains": records}, f)
            print(f"[dump_chains] wrote {len(records)} chains → {args.dump_chains}")
        if getattr(args, "dump_csv", None):
            n_csv = write_chains_csv(records, args.dump_csv)
            print(f"[dump_csv] wrote {n_csv} chains → {args.dump_csv}")

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
 
 
def _subset_match_meaningful(preds_list, gts_list, min_feasible_frac=0.5):
    """Whether a 'subset' gt-match is structurally meaningful for this GT.

    Subset gt-match scores 1 only when the chain contains EVERY gt node
    (gt ⊆ chain). That is achievable only if the GT can physically fit inside
    the predicted chain (|gt| ≤ |chain|). Broad GT — e.g. LFPN k-hop
    neighbourhoods with |GT| up to 1000+ — can never fit a depth-bounded chain,
    so the score collapses to ~0 and MISLEADS (it reads as "0% accurate" when
    the chain is in fact recovering the precise source; see Explanation Recall).

    Returns (meaningful, feasible_frac) where feasible_frac is the fraction of
    non-empty instances for which |gt| ≤ |chain| (subset is at least size-possible).
    """
    feasible = [len(g) <= len(p) for p, g in zip(preds_list, gts_list) if p and g]
    if not feasible:
        return False, 0.0
    frac = sum(feasible) / len(feasible)
    return frac >= min_feasible_frac, frac


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
        prefer_root_types=_parse_prefer_root_types(args.prefer_root_types),
        tracer_algorithm=args.tracer_algorithm,
        tracer_objective=args.tracer_objective,
    )
    from model.explainers import build_explainer
    explainer_name = getattr(args, "explainer", "ce_only")
    explainer = build_explainer(
        explainer_name,
        model=model,
        data=data,
        causal_graph=causal_graph,
        tracer=tracer,
        type_offsets=compute_type_offsets(data),
        target_node_type=model.backbone.target_node_type,
        n_permutations=getattr(args, "shapley_permutations", 64),
        shapley_topk=(getattr(args, "shapley_topk", 0) or None),
        coalition_subgraph=getattr(args, "coalition_subgraph", True),
    )
    print(f"  Explainer: {explainer_name}")
    preds_list, gts_list = [], []
    for node_id, gt_set in eligible.items():
        preds_list.append(explainer(node_id, causal_effects))
        gts_list.append(gt_set)
    from utils.metrics import compute_explanation_metrics
    metrics = compute_explanation_metrics(
        preds_list, gts_list, gt_match_mode=args.gt_match_mode
    )
    # Suppress a structurally-degenerate subset gt-match (broad GT can never fit
    # a bounded chain → ~0, which misreads as "totally inaccurate"). Report
    # Explanation Recall as the coverage metric in that regime instead.
    if args.gt_match_mode == "subset":
        meaningful, frac = _subset_match_meaningful(preds_list, gts_list)
        if not meaningful:
            metrics.pop("gt_match_accuracy", None)
            metrics.pop("gt_match_mode", None)
            metrics["gt_match"] = (
                f"n/a — GT broader than chain "
                f"({frac:.0%} of cases size-feasible); see Recall"
            )
            print("  [note] subset gt-match suppressed: GT is broader than the "
                  "bounded causal chain, so 'recover ALL gt nodes' is "
                  "structurally ~0; Explanation Recall is the coverage metric.")
    return metrics
 
 
def build_gt_list(args, data, type_offsets):
    gt_list = []
    if args.dataset == "elliptic++":
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
                wallet_per_address=args.variant in ("wallet", "joint"),
            )
            if gt:
                label = "LFPN-Strict" if m == "strict" \
                        else f"LFPN-Extended (k={args.lfpn_k})"
                gt_list.append((label, gt))
    return gt_list
 
 
def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── Restore architecture from the checkpoint (v2 format) ────────────────
    # A v2 checkpoint stores the layer count / hidden dim / head count it was
    # trained with. We prefer those over the CLI flags so the rebuilt model
    # always matches the trained weights — eliminating the "eval default
    # num_hgt_layers (2) ≠ train default (3) → AUC ok but F1≈0" trap. Legacy
    # checkpoints return None here and the CLI flags are used unchanged.
    ckpt_arch = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_arch = CI_RCT.read_arch_metadata(args.checkpoint, device=args.device)
    if ckpt_arch is None and args.checkpoint:
        print("  [arch] checkpoint has no embedded architecture (legacy "
              "format) — using CLI flags; ensure --num_hgt_layers etc. match "
              "the training recipe or F1 may silently collapse.")

    def arch_get(key, cli_value):
        """Prefer the checkpoint's stored architecture; fall back to CLI."""
        if ckpt_arch and ckpt_arch.get(key) is not None:
            stored = ckpt_arch[key]
            if stored != cli_value:
                print(f"  [arch] {key}: checkpoint={stored} "
                      f"(overrides CLI={cli_value})")
            return stored
        return cli_value

    print(f"Loading dataset: {args.dataset} (variant={args.variant})")
    data, target_type = _load_variant_dataset(args)
    data = data.to(device)
    labels    = data[target_type].y
    # --eval_split selects which split every metric (A/B/C/D) is computed on.
    # 'test' for reported numbers; 'val' for hyperparameter search so test stays
    # untouched. Variable name kept as test_mask to avoid threading a rename
    # through every downstream eval helper.
    mask_attr = "val_mask" if args.eval_split == "val" else "test_mask"
    test_mask = getattr(data[target_type], mask_attr)
    if test_mask is None:
        raise ValueError(
            f"--eval_split={args.eval_split} requested but "
            f"data[{target_type!r}].{mask_attr} is missing."
        )

    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_global_ids = [
        offset + i for i in test_indices if labels[i].item() == 1
    ]
    # Joint: also seed the BFS from fraud wallets (connectivity for dual-seed).
    fraud_wallet_ids = []
    if (args.variant == "joint" and "wallet" in data.node_types
            and hasattr(data["wallet"], "y")):
        w_off = type_offsets["wallet"]
        fraud_wallet_ids = [
            w_off + i
            for i in (data["wallet"].y == 1).nonzero(as_tuple=True)[0].tolist()
        ]

    # LFPN ground-truth (Metric C) is expensive and irrelevant to the φ probe.
    gt_list = [] if args.stability_probe_only \
        else build_gt_list(args, data, type_offsets)

    print("\nBuilding TypedCausalGraph...")
    gt_tx_ids = []
    for _, gt in gt_list:
        gt_tx_ids.extend(gt.keys())
    seed_ids = list(dict.fromkeys(
        fraud_global_ids[:args.num_seeds]
        + fraud_wallet_ids[:args.num_seeds]
        + gt_tx_ids
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
        hidden_dim=arch_get("hidden_dim", args.hidden_dim),
        num_hgt_layers=arch_get("num_hgt_layers", args.num_hgt_layers),
        num_heads=arch_get("num_heads", args.num_heads),
        dropout=arch_get("dropout", args.dropout),
        node_type_emb_dim=arch_get("node_type_emb_dim", args.type_emb_dim),
        ncm_baseline=args.ncm_baseline,
    )
    in_channels_dict = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types)
        if data[nt].x is not None
    }
    # Restore the training-time backbone exclusion (e.g. MG24 DD-8 Fix 4 drops
    # host_node from message passing). Defaults to [] for legacy checkpoints.
    backbone_exclude_node_types = arch_get("backbone_exclude_node_types", [])
    if args.variant == "joint":
        from model.ci_rct_joint import CI_RCT_Joint
        aux_node_types = arch_get("aux_node_types", ["wallet"])
        aux_num_classes = arch_get("aux_num_classes", {}) or {
            t: int(data[t].y.max().item()) + 1 for t in aux_node_types
        }
        model = CI_RCT_Joint(
            config=config,
            metadata=data.metadata(),
            in_channels_dict=in_channels_dict,
            use_gan=False,
            backbone_exclude_node_types=backbone_exclude_node_types,
            aux_node_types=list(aux_node_types),
            aux_num_classes=aux_num_classes,
        ).to(device)
    else:
        model = CI_RCT(
            config=config,
            metadata=data.metadata(),
            in_channels_dict=in_channels_dict,
            use_gan=False,
            backbone_exclude_node_types=backbone_exclude_node_types,
        ).to(device)
    if args.checkpoint:
        # PyG's HGTConv has per-relation lazy weights that only materialise
        # on the first forward(). If load_state_dict(strict=False) is called
        # before that, those weights' state-dict keys silently skip — the
        # subsequent forward then initialises them randomly, leaving the
        # model with checkpoint weights elsewhere but random per-relation
        # heads. Symptom: AUC roughly correct (ranking preserved by trained
        # layers) but F1 collapses (argmax bias from random heads).
        # Warm up with a dummy forward, then load.
        model.eval()
        with torch.no_grad():
            model.forward(data)
        model.load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — evaluating randomly initialised model (baseline).")
 
    # Resolve the decision threshold (argmax by default; manual or val-tuned).
    # Joint tunes a SEPARATE threshold per head inside eval_classification_pooled
    # (the two heads have different base rates → no single shared cut works).
    chosen_threshold = None
    if args.variant == "joint":
        pass
    elif args.threshold >= 0.0:
        chosen_threshold = args.threshold
        print(f"\n[threshold] using manual cut = {chosen_threshold:.3f}")
    elif args.threshold_tuning == "val":
        val_mask = getattr(data[target_type], "val_mask", None)
        if val_mask is None:
            print("\n[threshold] --threshold_tuning val requested but no val_mask; "
                  "falling back to argmax.")
        else:
            model.eval()
            with torch.no_grad():
                logits, _ = model.forward(data)
            val_scores = torch.softmax(logits[val_mask], dim=-1)[:, 1].cpu().numpy()
            val_true = labels[val_mask].cpu().numpy()
            chosen_threshold, val_obj = sweep_best_threshold(
                val_scores, val_true, objective=args.threshold_objective
            )
            print(f"\n[threshold] val-tuned cut = {chosen_threshold:.3f} "
                  f"({args.threshold_objective}={val_obj:.4f} on val)")

    # ── A. Classification ───────────────────────────────────────────────────
    extra_fraud_seeds = None
    if args.variant == "joint":
        # Per-head val-tuned thresholds when --threshold_tuning val is set
        # (a shared 0.5 cut lets the wallet head over-predict and collapse the
        # pooled fraud F1); otherwise legacy argmax.
        pooled_obj = (
            args.threshold_objective if args.threshold_tuning == "val" else None
        )
        cls_metrics, per_type_n, per_type_info = eval_classification_pooled(
            model, data, threshold_objective=pooled_obj, mask_attr=mask_attr
        )
        n_str = " + ".join(f"{t} {n:,}" for t, n in per_type_n.items())
        print_section(
            f"A. Classification (POOLED {args.eval_split} N = {n_str})",
            cls_metrics,
        )
        for t, info in per_type_info.items():
            print(f"    · {t:11s} fraud_f1={info['fraud_f1']:.4f}  "
                  f"pred_rate={info['pred_fraud_rate']:.4f}  "
                  f"thr={info['threshold']:.3f}")
        # Predicted-fraud wallet global ids → dual-seed tracing in Metric B.
        # Use the SAME (tuned) wallet threshold as Metric A for consistency.
        logits_by_type, _ = model.all_logits(data)
        if "wallet" in logits_by_type:
            w_off = type_offsets["wallet"]
            w_mask = getattr(data["wallet"], mask_attr)
            w_probs = torch.softmax(logits_by_type["wallet"], dim=-1)[:, 1]
            w_thr = per_type_info.get("wallet", {}).get("threshold", 0.5)
            w_idx = w_mask.nonzero(as_tuple=True)[0].tolist()
            extra_fraud_seeds = [
                w_off + i for i in w_idx if w_probs[i].item() > w_thr
            ]
    else:
        cls_metrics = eval_classification(
            model, data, labels, test_mask, threshold=chosen_threshold
        )
        print_section("A. Classification Metrics", cls_metrics)
    print(f"  ▶ Headline F1-score (fraud class) = {cls_metrics['fraud_f1']:.4f}")

    if args.stability_probe_only:
        phi_stability_probe(
            model, data, causal_graph, args, device,
            type_offsets=type_offsets, target_type=target_type,
            extra_fraud_seeds=extra_fraud_seeds,
        )
        print(f"\n{'─' * 55}\n  Stability probe complete.\n{'─' * 55}\n")
        return

    rct_metrics, stab_metrics = eval_root_cause_and_stability(
        model, data, labels, test_mask, causal_graph, args, device,
        type_offsets=type_offsets, target_type=target_type,
        extra_fraud_seeds=extra_fraud_seeds,
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