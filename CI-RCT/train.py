"""
CI-RCT training entry point.

Two training modes:
  --use_gan false   Phase 1 — backbone + NCM only (good for DBLP/ACM/IMDB sanity check)
  --use_gan true    Full training — backbone + NCM + CausalAdversarialGAN (for Elliptic++)

GAN training follows the WGAN-GP schedule:
  For every Generator update, run n_critic Discriminator updates first.

Loss:
  L_total = L_detection + λ1 · L_adversarial + λ2 · L_stability

Usage:
  # Phase 1 — quick sanity check on DBLP
  python train.py --dataset dblp --epochs 100 --use_gan false

  # Full training on Elliptic++
  python train.py --dataset elliptic --epochs 200 --use_gan true --lambda_adversarial 0.1

Supported datasets: dblp, acm, imdb, elliptic
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import HeteroData

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT
from utils.data_utils import (
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
    default_blocked_edge_types,
    default_rare_edge_types,
)
from utils.metrics import compute_classification_metrics, compute_fraud_f1


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CI-RCT: Causal Intervention-Based Root Cause Tracing"
    )
    parser.add_argument("--dataset", type=str, default="dblp",
                        choices=["dblp", "acm", "imdb", "elliptic", "elliptic++", "crypto", "unsw_nb15", "unsw_mg24"])
    # Elliptic++ detection target: 'transaction' (default, unchanged original
    # behaviour), 'wallet' (clean wallet labels), or 'joint' (one model that
    # classifies both, pooled F1). wallet/joint require --dataset elliptic++.
    parser.add_argument("--variant", type=str, default="transaction",
                        choices=["transaction", "wallet", "joint"])
    parser.add_argument("--lambda_aux_detection", type=float, default=0.3,
                        help="Weight of the auxiliary (wallet) detection loss "
                             "in --variant joint. Use ~1.0 with --symmetric_joint "
                             "to make wallet a co-equal head.")
    parser.add_argument("--symmetric_joint", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Joint head-weighting symmetrisation. False (default) "
                             "keeps the legacy ASYMMETRIC behaviour: a SEPARATE "
                             "0.3-weighted wallet backward AFTER the primary step "
                             "(wallet is a second-class head). True FUSES the wallet "
                             "loss into the SAME backward as the primary loss (one "
                             "optimiser step, gradients combined) so both heads "
                             "co-shape the backbone — pair with --lambda_aux_detection 1.0.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--type_emb_dim", type=int, default=16)
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--ce_threshold", type=float, default=0.1)
    parser.add_argument("--node_limit", type=int, default=500)
    # B1: rare-edge guarantee — see _expand_rare_edge_chain in data_utils.py.
    # Train-time NCM must see the same sparse-edge subgraph as eval, otherwise
    # the bridge / process / device edges never accumulate gradient.
    # Mirrors evaluate.py's --rare_edge_types / --rare_reserve / --rare_max_hops.
    parser.add_argument(
        "--rare_edge_types", type=str, default="",
        help="Comma-separated edge-type strings ('src__to__dst') the BFS "
             "must preserve. Empty string falls back to the per-dataset "
             "default; pass 'none' to disable the rare-edge pass entirely.",
    )
    parser.add_argument(
        "--rare_reserve", type=int, default=100,
        help="Node budget reserved for the rare-edge chain expansion pass.",
    )
    parser.add_argument(
        "--rare_max_hops", type=int, default=5,
        help="Maximum chain depth followed during the rare-edge pass.",
    )
    parser.add_argument(
        "--blocked_edge_types", type=str, default="",
        help="Comma-separated edge-type strings dropped from BOTH the BFS "
             "expansion and the final causal graph. Symmetric to "
             "--rare_edge_types. Empty falls back to "
             "default_blocked_edge_types(dataset); 'none' disables.",
    )
    # Joint loss weights
    parser.add_argument("--lambda_adversarial", type=float, default=0.1,
                        help="λ1: weight of WGAN-GP adversarial loss")
    parser.add_argument("--lambda_stability", type=float, default=0.5,
                        help="λ2: weight of Causal Shapley stability loss")
    parser.add_argument("--lambda_ncm", type=float, default=0.1,
                        help="λ3: weight of NCM supervision (BCE) loss")
    parser.add_argument("--use_reconstruction", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Enable GraphBEAN-style feature+edge reconstruction "
                             "self-supervision over ALL nodes (Step 1). Trains the "
                             "unlabeled majority; recommended for wallet/joint to "
                             "close the gap to SAGE-FIN. OFF keeps transaction "
                             "byte-identical.")
    parser.add_argument("--lambda_recon", type=float, default=1.0,
                        help="λ4: weight of reconstruction loss (only used when "
                             "--use_reconstruction true; SAGE-FIN treats recon as "
                             "a primary signal, so ~1.0 is a sensible start).")
    parser.add_argument("--ncm_edge_balance", type=str, default="none",
                        choices=("none", "uniform", "sqrt", "inverse"),
                        help="Per-edge-type NCM loss balancing (DD-17). "
                             "'sqrt' is recommended for highly imbalanced "
                             "hetero-graphs (e.g. MG24 host→flow has 200× "
                             "more edges than process→host, leaving sparse "
                             "edges' NCM CE≈0.001 at eval time).")
    parser.add_argument("--ncm_baseline", type=str, default="zero",
                        choices=("zero", "type_mean"),
                        help="CE null-intervention baseline. 'zero' (legacy) "
                             "do(h_u=0) is OOD and saturates p_null, making CE "
                             "sign uninterpretable; 'type_mean' do(h_u=E[h_type]) "
                             "recentres CE so sign = promote(+)/suppress(−).")
    # GAN settings
    parser.add_argument("--use_gan", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Enable Module 4 CausalAdversarialGAN")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Discriminator updates per Generator update (WGAN)")
    parser.add_argument("--gp_weight", type=float, default=10.0)
    parser.add_argument("--noise_std", type=float, default=0.05)
    # Model selection / early stopping (imbalance-aware)
    parser.add_argument(
        "--early_stop_metric", type=str, default="macro_f1",
        choices=["macro_f1", "fraud_f1", "weighted_f1"],
        help="Val metric used to pick the best checkpoint. macro_f1 (default, "
             "legacy behaviour) dilutes the minority class; fraud_f1 selects on "
             "the fraud class directly; weighted_f1 = (macro_f1 + fraud_f1) / 2.",
    )
    # Misc
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--subsample_tx", type=int, default=0,
                        help="Max target-type nodes to keep before GPU transfer (0=no limit). "
                             "Stratified: keeps all fraud + random licit. "
                             "Use ~20000 for 16 GB GPU with GAN.")
    # Memory reduction options (for Elliptic++)
    parser.add_argument("--include_addr_addr", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Include wallet→wallet edges (2.87M edges). "
                             "Default False to save GPU memory.")
    parser.add_argument("--labeled_only", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Only load labeled tx nodes + 1-hop neighbors (~1/10 graph size).")
    parser.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Keep all tx but restrict wallets to 1-2 hop neighbors of "
                             "labeled tx, with addr→addr edges within that wallet set.")
    parser.add_argument("--fraud_subgraph_hops", type=int, default=2,
                        help="Number of wallet hops from labeled tx (default 2).")
    parser.add_argument("--max_flows", type=int, default=200_000,
                        help="Max flow records for unsw_nb15 (0 = no limit).")
    # UNSW-MG24 specific options (see DD-1 in unsw_mg24_plan.md)
    parser.add_argument("--mg24_subsample_ddos", type=float, default=1.0,
                        help="Fraction of ddos1 flows to retain for unsw_mg24. "
                             "1.0 = full graph (DD-1 default); 0.1 = 10%% (OOM fallback).")
    parser.add_argument("--mg24_min_host_flows", type=int, default=5,
                        help="External-IP host pruning threshold for unsw_mg24.")
    parser.add_argument("--mg24_prune_external", type=lambda x: x.lower() == "true",
                        default=True,
                        help="Whether to prune external-only IP hosts in unsw_mg24.")
    parser.add_argument("--mg24_split_mode", type=str, default="by_file",
                        choices=("row", "by_file", "hybrid", "by_incident"),
                        help="Train/val/test split strategy for unsw_mg24 "
                             "(DD-8/DD-13). 'row' = row-level random "
                             "(data-leaky); 'by_file' = by-file stratified "
                             "(default, honest cross-session generalisation); "
                             "'hybrid' = benign row-level + malicious "
                             "by-file (production deployment scenario); "
                             "'by_incident' = attack_type aligned across "
                             "flow/audit modalities (DD-13, cuts cross-modal "
                             "label leakage).")
    parser.add_argument("--mg24_host_role", type=str, default="full",
                        choices=("full", "no_mal_count", "zeroed",
                                 "detection_excluded"),
                        help="DD-8 host-feature fairness ablation:\n"
                             "  full               baseline (incl. mal_flow_count)\n"
                             "  no_mal_count       Fix 1: drop label-derived count\n"
                             "  zeroed             Fix 3: zero all host features\n"
                             "  detection_excluded Fix 4: remove host_node from HGT\n"
                             "host_node stays in the graph for RootCauseTracer "
                             "in all modes.")
    parser.add_argument("--mg24_drop_features", type=str, default="",
                        help="DD-8 Fix 5: comma-separated CICFlowMeter "
                             "feature columns to remove from flow_node. "
                             "Example for ablating Active timing fingerprint: "
                             "'Active Std,Active Max,Active Mean'.")
    return parser.parse_args()


# ── Dataset loading ─────────────────────────────────────────────────────────────

def load_dataset(name: str, root: str, **kwargs):
    """Return (HeteroData, target_node_type)."""
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
        # Elliptic++ requires manual download — see README for instructions
        from utils.elliptic_loader import load_elliptic_dataset
        return load_elliptic_dataset(root)
    if name == "elliptic++":
        from utils.elliptic_plus_loader import load_elliptic_plus_dataset
        return load_elliptic_plus_dataset(
            os.path.join(root, "Elliptic++"),
            include_addr_addr=kwargs.get("include_addr_addr", False),
            labeled_only=kwargs.get("labeled_only", False),
            fraud_subgraph=kwargs.get("fraud_subgraph", False),
            fraud_subgraph_hops=kwargs.get("fraud_subgraph_hops", 2),
        )
    if name == "crypto":
        from utils.crypto_loader import load_crypto_dataset
        return load_crypto_dataset(root)
    if name == "unsw_nb15":
        from utils.unsw_loader import load_unsw_dataset
        return load_unsw_dataset(
            os.path.join(root, "unsw_nb15"),
            max_flows=kwargs.get("max_flows", 200_000),
        )
    if name == "unsw_mg24":
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
        split_mode = kwargs.get("mg24_split_mode", "by_file")
        # DD-8 host_role → host_features_mode mapping:
        #   "full"               → features="full"            (baseline)
        #   "no_mal_count"       → features="no_mal_count"    (Fix 1)
        #   "zeroed"             → features="zeroed"          (Fix 3)
        #   "detection_excluded" → features="full" + backbone exclude (Fix 4;
        #                          features value doesn't matter because the
        #                          backbone drops the node type entirely).
        host_role = kwargs.get("mg24_host_role", "full")
        host_features_mode = (
            host_role if host_role in ("full", "no_mal_count", "zeroed")
            else "full"
        )
        drop_raw = kwargs.get("mg24_drop_features", "") or ""
        flow_features_exclude = [
            c.strip() for c in drop_raw.split(",") if c.strip()
        ]
        print(
            f"  Split mode: {split_mode} (DD-8)\n"
            f"  Host role:  {host_role}  "
            f"(features={host_features_mode})"
        )
        if flow_features_exclude:
            print(f"  Dropping flow features (DD-8 Fix 5): "
                  f"{flow_features_exclude}")
        hd = to_pyg_hetero_data(
            mg24, edges,
            seed=kwargs.get("seed", 42),
            split_mode=split_mode,
            host_features_mode=host_features_mode,
            flow_features_exclude=flow_features_exclude or None,
        )
        # DD-3 primary target: flow_node (main detection task; baseline-comparable).
        return hd, "flow_node"
    raise ValueError(f"Unknown dataset: {name!r}")


# ── Training helpers ────────────────────────────────────────────────────────────

def train_step_no_gan(model, data, labels, train_mask, optimizer, causal_graph,
                      target_node, device, type_offsets, target_type,
                      class_weight=None, multi_task_labels=None,
                      ncm_edge_balance="none",
                      aux_labels=None, aux_masks=None, aux_class_weights=None,
                      lambda_aux=0.0):
    """
    Single training step without GAN (Phase 1).
    L_total = L_detection + λ2 · L_stability + λ3 · L_ncm

    Args:
        multi_task_labels: Optional {node_type: label_tensor}. When provided,
            HeteroNCM's supervision uses each edge's destination type's label
            instead of assuming dst is the target type. Required for MG24
            (DD-3) so `host→process`, `device→measurement`, etc. contribute
            real BCE supervision instead of being silently skipped.
    """
    model.train()
    optimizer.zero_grad()

    logits, h_dict = model.forward(data)
    detection_loss = F.cross_entropy(
        logits[train_mask], labels[train_mask], weight=class_weight
    )

    flat_h = model._build_flat_h(h_dict)

    # Compute CE WITH gradients so L_stability backpropagates into the NCM.
    ce_tensors = model.hetero_ncm.forward(flat_h, causal_graph)  # {(src,dst): Tensor}
    # Detached floats used only for Shapley tracking and prev_phi buffer.
    ce_scores = {k: v.item() for k, v in ce_tensors.items()}

    from model.causal_shapley import compute_asymmetric_causal_shapley
    phi_current = compute_asymmetric_causal_shapley(ce_scores, causal_graph, target_node)

    stability_loss = torch.zeros(1, device=device)
    parents = list(causal_graph.parents(target_node))
    n_parents = len(parents)
    if model._prev_phi is not None and phi_current and n_parents > 0:
        common = set(phi_current.keys()) & set(model._prev_phi.keys())
        if common:
            # φ_t = CE(p → target) / n  — keep tensor so NCM receives gradient
            diffs = []
            for p in common:
                ce_t = ce_tensors.get((p, target_node), torch.zeros(1, device=device))
                phi_t = ce_t / n_parents
                phi_prev = torch.tensor(
                    model._prev_phi[p], dtype=torch.float32, device=device
                )
                diffs.append((phi_t - phi_prev) ** 2)
            stability_loss = torch.stack(diffs).mean()

    model._prev_phi = dict(phi_current)

    # L_ncm: supervise NCM to predict the destination node's binary label.
    # For MG24 (DD-3), `multi_task_labels` lets every labelled dst type
    # (flow / process / measurement) provide BCE signal — without this, all
    # edges whose dst is not the primary target type would be silently
    # skipped and the NCM loss would plateau (observed in pilot run).
    ncm_loss = model.hetero_ncm.supervised_ncm_loss(
        flat_h, causal_graph, labels, type_offsets[target_type],
        wallet_labels=data["wallet"].y if hasattr(data["wallet"], "y") else None,
        wallet_type_offset=type_offsets.get("wallet", 0),
        multi_task_labels=multi_task_labels,
        type_offsets=type_offsets if multi_task_labels is not None else None,
        edge_balance=ncm_edge_balance,
    )

    recon_loss = torch.zeros(1, device=device)
    if getattr(model, "use_reconstruction", False):
        recon_loss = model.reconstruction_loss(data, h_dict)

    total_loss = (
        detection_loss
        + model.config.lambda_stability * stability_loss
        + model.config.lambda_ncm * ncm_loss
        + model.config.lambda_recon * recon_loss
    )

    # Symmetric joint: fuse the wallet (aux) detection loss into the same backward.
    aux_loss_val = 0.0
    if aux_labels is not None:
        aux_loss = model.aux_detection_loss(
            data, aux_labels, aux_masks, aux_class_weights
        )
        total_loss = total_loss + lambda_aux * aux_loss
        aux_loss_val = float(aux_loss.detach().item())

    total_loss.backward()
    optimizer.step()

    return (total_loss.item(), detection_loss.item(), stability_loss.item(),
            ncm_loss.item(), aux_loss_val)


def train_step_with_gan(model, data, labels, train_mask, optimizer_backbone,
                        optimizer_generator, causal_graph, fraud_features,
                        topo_order, target_node, step_count, device,
                        type_offsets, target_type, class_weight=None,
                        aux_labels=None, aux_masks=None, aux_class_weights=None,
                        lambda_aux=0.0):
    """
    WGAN-GP training step (Phase 2):
      - Every step: update Discriminator (= backbone)
      - Every n_critic steps: also update Generator

    When ``aux_labels`` is provided (symmetric joint), the auxiliary wallet
    detection loss is FUSED into the discriminator step's total loss before the
    single backward, so the primary and auxiliary heads co-shape the shared
    backbone in one optimiser step (vs the legacy separate 0.3-weighted step).
    """
    n_critic = model.config.n_critic

    # ── Discriminator step ──────────────────────────────────────────────────
    model.train()
    optimizer_backbone.zero_grad()

    target_type_offset = type_offsets[target_type]
    total_loss, detection_loss, adv_loss, stability_loss, ncm_loss, _recon_loss = model.compute_total_loss(
        data=data,
        labels=labels,
        train_mask=train_mask,
        causal_graph=causal_graph,
        fraud_features=fraud_features,
        topo_order=topo_order,
        target_node=target_node,
        is_critic_step=True,
        class_weight=class_weight,
        target_type_offset=target_type_offset,
        wallet_labels=data["wallet"].y if hasattr(data["wallet"], "y") else None,
        wallet_type_offset=type_offsets.get("wallet", 0),
    )
    # Symmetric joint: fuse the wallet (aux) detection loss into the SAME
    # backward so primary + auxiliary co-shape the backbone in one step.
    aux_loss_val = 0.0
    if aux_labels is not None:
        aux_loss = model.aux_detection_loss(
            data, aux_labels, aux_masks, aux_class_weights
        )
        total_loss = total_loss + lambda_aux * aux_loss
        aux_loss_val = float(aux_loss.detach().item())

    # Use full loss (detection + adversarial + NCM + stability [+ aux]) for the
    # discriminator/backbone update.
    total_loss.backward()
    optimizer_backbone.step()

    # ── Generator step (every n_critic steps) ──────────────────────────────
    g_loss_val = 0.0
    if step_count % n_critic == 0:
        optimizer_generator.zero_grad()
        _, _, g_adv_loss, _, _, _ = model.compute_total_loss(
            data=data,
            labels=labels,
            train_mask=train_mask,
            causal_graph=causal_graph,
            fraud_features=fraud_features,
            topo_order=topo_order,
            target_node=target_node,
            is_critic_step=False,
            class_weight=class_weight,
            target_type_offset=target_type_offset,
            wallet_labels=data["wallet"].y if hasattr(data["wallet"], "y") else None,
            wallet_type_offset=type_offsets.get("wallet", 0),
        )
        g_loss = model.config.lambda_adversarial * g_adv_loss
        g_loss.backward()
        optimizer_generator.step()
        g_loss_val = g_loss.item()

    return (total_loss.item(), detection_loss.item(), adv_loss.item(),
            g_loss_val, ncm_loss.item(), aux_loss_val)


@torch.no_grad()
def evaluate_split(model, data, labels, mask):
    model.eval()
    logits, _ = model.forward(data)
    probs = torch.softmax(logits[mask], dim=-1)
    preds = probs.argmax(dim=-1)
    # Binary: pass class-1 probability; multi-class: pass full prob matrix
    if probs.size(1) == 2:
        scores = probs[:, 1]
    else:
        scores = probs
    y_true = labels[mask].cpu()
    preds_cpu = preds.cpu()
    metrics = compute_classification_metrics(preds_cpu, y_true, scores.cpu())
    # Imbalance-aware extras for model selection (binary detection only).
    if probs.size(1) == 2:
        fraud_f1 = compute_fraud_f1(y_true, preds_cpu)
        metrics["fraud_f1"] = fraud_f1
        metrics["weighted_f1"] = 0.5 * metrics["f1"] + 0.5 * fraud_f1
    else:
        metrics["fraud_f1"] = metrics["f1"]
        metrics["weighted_f1"] = metrics["f1"]
    return metrics


# ── Variant helpers (wallet / joint detection) ──────────────────────────────────

def _early_stop_metric_explicit() -> bool:
    """True if the user explicitly passed --early_stop_metric on the CLI."""
    return any(
        a == "--early_stop_metric" or a.startswith("--early_stop_metric=")
        for a in sys.argv
    )


def _inverse_freq_class_weight(labels, mask, device):
    """Inverse-frequency [1.0, neg/pos] weight, or None for a balanced split."""
    pos = labels[mask].sum().float()
    neg = (labels[mask] == 0).sum().float()
    if pos > 0 and neg > 0 and pos != neg:
        return torch.tensor([1.0, (neg / pos).item()], dtype=torch.float32, device=device)
    return None


def _load_variant_dataset(args):
    """Dispatch the loader by --variant (elliptic++ only for wallet/joint)."""
    if args.variant == "transaction":
        return load_dataset(
            args.dataset, args.data_root,
            include_addr_addr=args.include_addr_addr,
            labeled_only=args.labeled_only,
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
    # joint
    from utils.elliptic_plus_joint_loader import load_elliptic_plus_joint_dataset
    return load_elliptic_plus_joint_dataset(
        root, include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )


@torch.no_grad()
def pooled_eval(model, data, mask_attr: str) -> dict:
    """Pooled metrics over ALL classified types' <mask_attr> nodes (joint).

    Each type's predictions come from its OWN head (primary for transaction,
    aux head for wallet); the (y_true, y_pred, score) triples are concatenated
    and a single set of metrics is computed — the joint model's overall F1.
    """
    model.eval()
    logits_by_type, _ = model.all_logits(data)
    y_true, y_pred, scores = [], [], []
    for ntype, logits in logits_by_type.items():
        mask = getattr(data[ntype], mask_attr, None)
        if mask is None or not bool(mask.any()):
            continue
        probs = torch.softmax(logits[mask], dim=-1)
        y_pred.append(probs.argmax(dim=-1).cpu())
        scores.append(probs[:, 1].cpu())
        y_true.append(data[ntype].y[mask].cpu())
    if not y_true:
        return {"f1": 0.0, "fraud_f1": 0.0, "weighted_f1": 0.0, "auc": 0.0}
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    scores = torch.cat(scores)
    m = compute_classification_metrics(y_pred, y_true, scores)
    m["fraud_f1"] = compute_fraud_f1(y_true, y_pred)
    m["weighted_f1"] = 0.5 * m["f1"] + 0.5 * m["fraud_f1"]
    return m


# ── Graph subsampling ───────────────────────────────────────────────────────────

def _subsample_hetero(data, target_type: str, max_target: int, seed: int = 42):
    """
    Subsample the heterogeneous graph so that at most `max_target` nodes of
    `target_type` are kept.  Strategy:
      - Keep ALL fraud (label=1) nodes in the training set.
      - Fill remaining budget with randomly sampled licit nodes.
      - Keep only wallet nodes that are directly connected to selected transactions.
    This preserves the train/val/test masks and label distribution.
    """
    torch.manual_seed(seed)
    y = data[target_type].y
    n_target = data[target_type].num_nodes

    fraud_idx = (y == 1).nonzero(as_tuple=True)[0]
    licit_idx = (y == 0).nonzero(as_tuple=True)[0]

    n_keep_licit = max(0, max_target - len(fraud_idx))
    perm = torch.randperm(len(licit_idx))[:n_keep_licit]
    sampled_tx = torch.cat([fraud_idx, licit_idx[perm]]).sort()[0]

    tx_mask = torch.zeros(n_target, dtype=torch.bool)
    tx_mask[sampled_tx] = True

    subset_dict = {target_type: tx_mask}

    # For every other node type, keep only those connected to selected transactions
    for nt in data.node_types:
        if nt == target_type:
            continue
        keep = torch.zeros(data[nt].num_nodes, dtype=torch.bool)
        for src_t, _, dst_t in data.edge_types:
            ei = data[(src_t, _, dst_t)].edge_index
            if src_t == target_type and dst_t == nt:
                keep[ei[1][tx_mask[ei[0]]]] = True
            elif src_t == nt and dst_t == target_type:
                keep[ei[0][tx_mask[ei[1]]]] = True
        subset_dict[nt] = keep

    kept_tx = int(tx_mask.sum())
    kept_other = {nt: int(subset_dict[nt].sum()) for nt in subset_dict if nt != target_type}
    print(f"  Subsampled: {kept_tx}/{n_target} {target_type}s, "
          + ", ".join(f"{v}/{data[k].num_nodes} {k}s" for k, v in kept_other.items()))

    return data.subgraph(subset_dict)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset} (variant={args.variant})")
    data, target_type = _load_variant_dataset(args)

    # Subsample graph before moving to GPU to avoid OOM on large datasets
    if args.subsample_tx > 0 and data[target_type].num_nodes > args.subsample_tx:
        data = _subsample_hetero(data, target_type, args.subsample_tx, seed=args.seed)

    data = data.to(device)

    labels = data[target_type].y
    train_mask = data[target_type].train_mask
    val_mask   = data[target_type].val_mask
    test_mask  = data[target_type].test_mask

    # --- Config ---
    config = CI_RCT_Config(
        dataset=args.dataset,
        data_root=args.data_root,
        target_node_type=target_type,
        hidden_dim=args.hidden_dim,
        num_hgt_layers=args.num_hgt_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        node_type_emb_dim=args.type_emb_dim,
        ncm_baseline=args.ncm_baseline,
        max_hops=args.max_hops,
        ce_threshold=args.ce_threshold,
        node_limit=args.node_limit,
        lambda_adversarial=args.lambda_adversarial,
        lambda_stability=args.lambda_stability,
        lambda_ncm=args.lambda_ncm,
        lambda_recon=args.lambda_recon,
        n_critic=args.n_critic,
        gp_weight=args.gp_weight,
        noise_std=args.noise_std,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    # --- Causal graph ---
    # Use fraud training nodes as BFS seeds so the subgraph is connected.
    print("Building TypedCausalGraph...")
    type_offsets = compute_type_offsets(data)
    train_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_global_ids = [
        type_offsets[target_type] + i
        for i in train_indices
        if labels[i].item() == 1
    ]
    seed_ids = fraud_global_ids[:20]  # up to 20 fraud seeds for multi-source BFS

    # B1: same rare-edge handling as evaluate.py — explicit > default > none.
    raw_rare = args.rare_edge_types.strip()
    if raw_rare.lower() == "none":
        rare_edge_types = set()
        rare_src = "disabled"
    elif raw_rare:
        rare_edge_types = {t.strip() for t in raw_rare.split(",") if t.strip()}
        rare_src = "explicit"
    else:
        rare_edge_types = default_rare_edge_types(args.dataset)
        rare_src = "dataset-default"
    if rare_edge_types:
        print(f"  [BFS] rare edge types ({len(rare_edge_types)}, "
              f"reserve={args.rare_reserve}, "
              f"max_hops={args.rare_max_hops}, src={rare_src}): "
              f"{sorted(rare_edge_types)}")

    # Symmetric: --blocked_edge_types (replaces the legacy block_addr_to_addr).
    raw_b = args.blocked_edge_types.strip()
    if raw_b.lower() == "none":
        blocked_edge_types = set()
        blocked_src = "disabled"
    elif raw_b:
        blocked_edge_types = {t.strip() for t in raw_b.split(",") if t.strip()}
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
        node_limit=config.node_limit,
        blocked_edge_types=blocked_edge_types if blocked_edge_types else None,
        rare_edge_types=rare_edge_types if rare_edge_types else None,
        rare_reserve=args.rare_reserve,
        rare_max_hops=args.rare_max_hops,
    )
    topo_order = causal_graph.topological_order()

    # Pick one fraud training node that has parents in the causal graph.
    target_node = seed_ids[0] if seed_ids else type_offsets[target_type]
    for gid in seed_ids:
        if gid in causal_graph.set_v and causal_graph.parents(gid):
            target_node = gid
            break
    print(f"  Causal graph: {len(causal_graph.v)} nodes, "
          f"{len(causal_graph.edge_type_map)} directed edges")
    print(f"  [debug] target_node={target_node}, "
          f"in_graph={target_node in causal_graph.set_v}, "
          f"parents={causal_graph.parents(target_node)}")

    # --- in_channels_dict ---
    in_channels_dict = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types)
        if data[nt].x is not None
    }

    # --- num_classes (inferred from labels) ---
    num_classes = int(labels.max().item()) + 1

    # --- Model ---
    node_feature_dim = in_channels_dict.get(target_type)
    # DD-8 Fix 4: when --mg24_host_role=detection_excluded, drop host_node
    # from HGT message passing (graph stays intact for RootCauseTracer).
    backbone_exclude_node_types = []
    if args.dataset == "unsw_mg24" and args.mg24_host_role == "detection_excluded":
        backbone_exclude_node_types = ["host_node"]
        print(f"  Detection graph excludes: {backbone_exclude_node_types} (DD-8 Fix 4)")
    if args.variant == "joint":
        from model.ci_rct_joint import CI_RCT_Joint
        aux_num_classes = {"wallet": int(data["wallet"].y.max().item()) + 1}
        model = CI_RCT_Joint(
            config=config,
            metadata=data.metadata(),
            in_channels_dict=in_channels_dict,
            node_feature_dim=node_feature_dim if args.use_gan else None,
            use_gan=args.use_gan,
            num_classes=num_classes,
            backbone_exclude_node_types=backbone_exclude_node_types,
            aux_node_types=["wallet"],
            aux_num_classes=aux_num_classes,
            use_reconstruction=args.use_reconstruction,
        ).to(device)
        _joint_mode = ("SYMMETRIC (aux fused into primary backward, co-equal head)"
                       if args.symmetric_joint
                       else "asymmetric (separate down-weighted aux step, legacy)")
        print(f"  Joint heads: primary={target_type}  aux=['wallet']  "
              f"(λ_aux={args.lambda_aux_detection})  mode={_joint_mode}")
    else:
        model = CI_RCT(
            config=config,
            metadata=data.metadata(),
            in_channels_dict=in_channels_dict,
            node_feature_dim=node_feature_dim if args.use_gan else None,
            use_gan=args.use_gan,
            num_classes=num_classes,
            backbone_exclude_node_types=backbone_exclude_node_types,
            use_reconstruction=args.use_reconstruction,
        ).to(device)

    # --- Optimisers ---
    _backbone_params = (
        list(model.backbone.parameters()) + list(model.hetero_ncm.parameters())
    )
    if args.variant == "joint":
        _backbone_params += list(model.aux_classifiers.parameters())
    # Reconstruction decoders (Step 1) train on the same backbone optimiser so
    # L_recon actually updates them (and back-propagates into the backbone).
    if getattr(model, "use_reconstruction", False):
        if model.feature_decoders is not None:
            _backbone_params += list(model.feature_decoders.parameters())
        if model.edge_decoder is not None:
            _backbone_params += list(model.edge_decoder.parameters())
        print(f"  Reconstruction self-supervision ON "
              f"(λ_recon={config.lambda_recon}, edge={model._recon_edge_type})")
    optimizer_backbone = optim.Adam(
        _backbone_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_generator = None
    if args.use_gan and model.causal_gan is not None:
        optimizer_generator = optim.Adam(
            model.causal_gan.generator.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # --- Class weight (inverse frequency, handles imbalanced datasets) ---
    pos = labels[train_mask].sum().float()
    neg = (labels[train_mask] == 0).sum().float()
    if pos > 0 and neg > 0 and pos != neg:
        class_weight = torch.tensor(
            [1.0, (neg / pos).item()], dtype=torch.float32, device=device
        )
        print(f"  Class weight: licit=1.00  illicit={neg/pos:.2f}  "
              f"(pos={int(pos)}, neg={int(neg)})")
    else:
        class_weight = None
        print("  Class weight: None (balanced dataset)")

    # --- Joint variant: auxiliary (wallet) supervision tensors ---
    aux_labels = aux_masks = aux_class_weights = None
    if args.variant == "joint":
        aux_labels = {"wallet": data["wallet"].y}
        aux_masks = {"wallet": data["wallet"].train_mask}
        aux_class_weights = {
            "wallet": _inverse_freq_class_weight(
                data["wallet"].y, data["wallet"].train_mask, device
            )
        }

    # --- Fraud features for GAN (use training fraud nodes) ---
    fraud_features = None
    if args.use_gan:
        fraud_mask = (labels == 1) & train_mask
        if fraud_mask.any():
            fraud_features = data[target_type].x[fraud_mask].to(device)
        else:
            # Fallback: use all training nodes
            fraud_features = data[target_type].x[train_mask].to(device)

    # --- Multi-task labels for HeteroNCM supervision (DD-3, MG24-only) ---
    # Every labelled node type with `y` attached contributes BCE supervision
    # for its incoming causal edges. For datasets with only a single labelled
    # type (Elliptic++, NB15, DBLP) this remains None and the legacy
    # single-task / wallet→tx path in supervised_ncm_loss() is used.
    multi_task_labels = None
    if args.dataset == "unsw_mg24":
        multi_task_labels = {
            ntype: data[ntype].y
            for ntype in data.node_types
            if hasattr(data[ntype], "y") and data[ntype].y is not None
        }
        labelled_types = sorted(multi_task_labels.keys())
        print(f"  Multi-task NCM supervision: {labelled_types}")

    # --- Training loop ---
    mode_str = "Phase 2 (GAN)" if args.use_gan else "Phase 1 (No GAN)"
    print(f"\nTraining [{mode_str}] for {args.epochs} epochs on {device} "
          f"(variant={args.variant})...")
    # wallet/joint default to the binary fraud-class F1 (transaction keeps the
    # original macro_f1 default unless the user overrides it).
    if args.variant != "transaction" and not _early_stop_metric_explicit():
        args.early_stop_metric = "fraud_f1"
    sel_key = {"macro_f1": "f1", "fraud_f1": "fraud_f1",
               "weighted_f1": "weighted_f1"}[args.early_stop_metric]
    best_val_score = 0.0
    _ckpt_suffix = "" if args.variant == "transaction" else f"_{args.variant}"
    ckpt_path = os.path.join(
        args.checkpoint_dir, f"ci_rct_{args.dataset}{_ckpt_suffix}_best.pt"
    )
    print(f"  Model selection metric: {args.early_stop_metric} (val['{sel_key}'])")

    model.reset_phi_buffer()  # initialise once before training starts
    for epoch in range(1, args.epochs + 1):

        # Symmetric joint FUSES the wallet aux loss into the primary backward
        # (co-equal head); asymmetric (legacy) keeps it as a separate
        # down-weighted step below.
        _fuse_aux = args.variant == "joint" and args.symmetric_joint
        _aux_kw = (
            dict(aux_labels=aux_labels, aux_masks=aux_masks,
                 aux_class_weights=aux_class_weights,
                 lambda_aux=args.lambda_aux_detection)
            if _fuse_aux else {}
        )

        if not args.use_gan:
            total_loss, det_loss, stab_loss, ncm_loss, aux_val = train_step_no_gan(
                model, data, labels, train_mask, optimizer_backbone,
                causal_graph, target_node, device,
                type_offsets, target_type, class_weight=class_weight,
                multi_task_labels=multi_task_labels,
                ncm_edge_balance=args.ncm_edge_balance,
                **_aux_kw,
            )
            loss_str = (f"Loss {total_loss:.4f} "
                        f"(det={det_loss:.4f}, stab={stab_loss:.2e}, ncm={ncm_loss:.4f})")
        else:
            total_loss, det_loss, adv_loss, g_loss, ncm_loss, aux_val = train_step_with_gan(
                model, data, labels, train_mask,
                optimizer_backbone, optimizer_generator,
                causal_graph, fraud_features, topo_order, target_node,
                step_count=epoch, device=device,
                type_offsets=type_offsets, target_type=target_type,
                class_weight=class_weight,
                **_aux_kw,
            )
            loss_str = (f"Loss {total_loss:.4f} "
                        f"(det={det_loss:.4f}, adv={adv_loss:.4f}, G={g_loss:.4f}, ncm={ncm_loss:.4f})")

        if _fuse_aux:
            loss_str += f" aux={aux_val:.4f}(fused)"
        elif args.variant == "joint":
            # Legacy ASYMMETRIC: separate wallet (auxiliary) backward/step on the
            # shared backbone, AFTER the primary update.
            optimizer_backbone.zero_grad()
            aux_loss = model.aux_detection_loss(
                data, aux_labels, aux_masks, aux_class_weights
            )
            (args.lambda_aux_detection * aux_loss).backward()
            optimizer_backbone.step()
            loss_str += f" aux={float(aux_loss.detach().item()):.4f}(separate)"

        if epoch % args.eval_every == 0:
            val_metrics = (
                pooled_eval(model, data, "val_mask") if args.variant == "joint"
                else evaluate_split(model, data, labels, val_mask)
            )
            print(
                f"Epoch {epoch:03d} | {loss_str} | "
                f"Val macroF1={val_metrics['f1']:.4f}  fraudF1={val_metrics['fraud_f1']:.4f}  "
                f"AUC={val_metrics['auc']:.4f}"
            )
            if val_metrics[sel_key] > best_val_score:
                best_val_score = val_metrics[sel_key]
                model.save_checkpoint(ckpt_path)
                print(f"  ↑ Best checkpoint saved (val {args.early_stop_metric}"
                      f"={best_val_score:.4f}) → {ckpt_path}")

    # --- Final test ---
    print("\nLoading best checkpoint for test evaluation...")
    model.load_checkpoint(ckpt_path)
    test_metrics = (
        pooled_eval(model, data, "test_mask") if args.variant == "joint"
        else evaluate_split(model, data, labels, test_mask)
    )
    print(f"Test  fraudF1={test_metrics['fraud_f1']:.4f}  "
          f"(macroF1={test_metrics['f1']:.4f}  AUC={test_metrics['auc']:.4f})")


if __name__ == "__main__":
    main()
