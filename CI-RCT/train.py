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

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import HeteroData

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT
from utils.data_utils import build_typed_causal_graph_from_hetero, compute_type_offsets
from utils.metrics import compute_classification_metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CI-RCT: Causal Intervention-Based Root Cause Tracing"
    )
    parser.add_argument("--dataset", type=str, default="dblp",
                        choices=["dblp", "acm", "imdb", "elliptic", "elliptic++", "crypto"])
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
    # Joint loss weights
    parser.add_argument("--lambda_adversarial", type=float, default=0.1,
                        help="λ1: weight of WGAN-GP adversarial loss")
    parser.add_argument("--lambda_stability", type=float, default=0.5,
                        help="λ2: weight of Causal Shapley stability loss")
    # GAN settings
    parser.add_argument("--use_gan", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Enable Module 4 CausalAdversarialGAN")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Discriminator updates per Generator update (WGAN)")
    parser.add_argument("--gp_weight", type=float, default=10.0)
    parser.add_argument("--noise_std", type=float, default=0.05)
    # Misc
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
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
    raise ValueError(f"Unknown dataset: {name!r}")


# ── Training helpers ────────────────────────────────────────────────────────────

def train_step_no_gan(model, data, labels, train_mask, optimizer, causal_graph,
                      target_node, device, class_weight=None):
    """
    Single training step without GAN (Phase 1).
    L_total = L_detection + λ2 · L_stability
    """
    model.train()
    optimizer.zero_grad()

    logits, h_dict = model.forward(data)
    detection_loss = F.cross_entropy(
        logits[train_mask], labels[train_mask], weight=class_weight
    )

    flat_h = model._build_flat_h(h_dict)
    ce_scores = model.hetero_ncm.detached_causal_effects(flat_h, causal_graph)

    from model.causal_shapley import compute_asymmetric_causal_shapley
    phi_current = compute_asymmetric_causal_shapley(ce_scores, causal_graph, target_node)

    stability_loss = torch.zeros(1, device=device)
    if model._prev_phi is not None and phi_current:
        common = set(phi_current.keys()) & set(model._prev_phi.keys())
        if common:
            diffs = [(phi_current[p] - model._prev_phi[p]) ** 2 for p in common]
            stability_loss = torch.tensor(
                sum(diffs) / len(diffs), dtype=torch.float32,
                device=device, requires_grad=False,
            )

    model._prev_phi = dict(phi_current)

    total_loss = detection_loss + model.config.lambda_stability * stability_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), detection_loss.item(), stability_loss.item()


def train_step_with_gan(model, data, labels, train_mask, optimizer_backbone,
                        optimizer_generator, causal_graph, fraud_features,
                        topo_order, target_node, step_count, device,
                        class_weight=None):
    """
    WGAN-GP training step (Phase 2):
      - Every step: update Discriminator (= backbone)
      - Every n_critic steps: also update Generator
    """
    n_critic = model.config.n_critic

    # ── Discriminator step ──────────────────────────────────────────────────
    model.train()
    optimizer_backbone.zero_grad()

    total_loss, detection_loss, adv_loss, stability_loss = model.compute_total_loss(
        data=data,
        labels=labels,
        train_mask=train_mask,
        causal_graph=causal_graph,
        fraud_features=fraud_features,
        topo_order=topo_order,
        target_node=target_node,
        is_critic_step=True,
        class_weight=class_weight,
    )
    # Use only detection + adversarial for discriminator update
    d_loss = detection_loss + model.config.lambda_adversarial * adv_loss
    d_loss.backward()
    optimizer_backbone.step()

    # ── Generator step (every n_critic steps) ──────────────────────────────
    g_loss_val = 0.0
    if step_count % n_critic == 0:
        optimizer_generator.zero_grad()
        _, _, g_adv_loss, _ = model.compute_total_loss(
            data=data,
            labels=labels,
            train_mask=train_mask,
            causal_graph=causal_graph,
            fraud_features=fraud_features,
            topo_order=topo_order,
            target_node=target_node,
            is_critic_step=False,
            class_weight=class_weight,
        )
        g_loss = model.config.lambda_adversarial * g_adv_loss
        g_loss.backward()
        optimizer_generator.step()
        g_loss_val = g_loss.item()

    return total_loss.item(), detection_loss.item(), adv_loss.item(), g_loss_val


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
    return compute_classification_metrics(
        preds.cpu(), labels[mask].cpu(), scores.cpu()
    )


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    data, target_type = load_dataset(
        args.dataset, args.data_root,
        include_addr_addr=args.include_addr_addr,
        labeled_only=args.labeled_only,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )
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
        max_hops=args.max_hops,
        ce_threshold=args.ce_threshold,
        node_limit=args.node_limit,
        lambda_adversarial=args.lambda_adversarial,
        lambda_stability=args.lambda_stability,
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

    causal_graph = build_typed_causal_graph_from_hetero(
        data,
        seed_node_ids=seed_ids if seed_ids else None,
        node_limit=config.node_limit,
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
    model = CI_RCT(
        config=config,
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        node_feature_dim=node_feature_dim if args.use_gan else None,
        use_gan=args.use_gan,
        num_classes=num_classes,
    ).to(device)

    # --- Optimisers ---
    optimizer_backbone = optim.Adam(
        list(model.backbone.parameters()) + list(model.hetero_ncm.parameters()),
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

    # --- Fraud features for GAN (use training fraud nodes) ---
    fraud_features = None
    if args.use_gan:
        fraud_mask = (labels == 1) & train_mask
        if fraud_mask.any():
            fraud_features = data[target_type].x[fraud_mask].to(device)
        else:
            # Fallback: use all training nodes
            fraud_features = data[target_type].x[train_mask].to(device)

    # --- Training loop ---
    mode_str = "Phase 2 (GAN)" if args.use_gan else "Phase 1 (No GAN)"
    print(f"\nTraining [{mode_str}] for {args.epochs} epochs on {device}...")
    best_val_f1 = 0.0
    ckpt_path = os.path.join(args.checkpoint_dir, f"ci_rct_{args.dataset}_best.pt")

    model.reset_phi_buffer()  # initialise once before training starts
    for epoch in range(1, args.epochs + 1):

        if not args.use_gan:
            total_loss, det_loss, stab_loss = train_step_no_gan(
                model, data, labels, train_mask, optimizer_backbone,
                causal_graph, target_node, device, class_weight=class_weight
            )
            loss_str = (f"Loss {total_loss:.4f} "
                        f"(det={det_loss:.4f}, stab={stab_loss:.2e})")
        else:
            total_loss, det_loss, adv_loss, g_loss = train_step_with_gan(
                model, data, labels, train_mask,
                optimizer_backbone, optimizer_generator,
                causal_graph, fraud_features, topo_order, target_node,
                step_count=epoch, device=device, class_weight=class_weight
            )
            loss_str = (f"Loss {total_loss:.4f} "
                        f"(det={det_loss:.4f}, adv={adv_loss:.4f}, G={g_loss:.4f})")

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_split(model, data, labels, val_mask)
            print(
                f"Epoch {epoch:03d} | {loss_str} | "
                f"Val F1={val_metrics['f1']:.4f}  AUC={val_metrics['auc']:.4f}"
            )
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                model.save_checkpoint(ckpt_path)
                print(f"  ↑ Best checkpoint saved → {ckpt_path}")

    # --- Final test ---
    print("\nLoading best checkpoint for test evaluation...")
    model.load_checkpoint(ckpt_path)
    test_metrics = evaluate_split(model, data, labels, test_mask)
    print(f"Test  F1={test_metrics['f1']:.4f}  AUC={test_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
