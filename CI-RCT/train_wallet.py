"""
CI-RCT *wallet-target* training entry point (Elliptic++ only).

A thin sibling of ``train.py`` for the SAGE-FIN per-type comparison: it trains
the **unchanged** CI-RCT model with ``target_node_type="wallet"`` so we get a
wallet-only detection F1 alongside the existing transaction-only F1.

Nothing under ``model/`` or in ``train.py`` is modified — this file *imports*
``train.py``'s training steps and reuses them verbatim; only the orchestration
(dataset loading with clean wallet labels, target type, checkpoint name) differs.

Usage:
  python train_wallet.py --dataset elliptic++ --epochs 200 --use_gan true \
    --fraud_subgraph true --fraud_subgraph_hops 2 --seed 42 \
    --early_stop_metric fraud_f1 --checkpoint_dir checkpoints/wallet

The checkpoint is written as ``ci_rct_<dataset>_wallet_best.pt`` so it never
collides with the transaction model's ``ci_rct_<dataset>_best.pt``.

IMPORTANT: pass the SAME --fraud_subgraph / --include_addr_addr /
--fraud_subgraph_hops / --seed flags to ``evaluate_wallet.py`` — the wallet set
and its test mask are only reproducible under identical flags.
"""
import os
import random

import numpy as np
import torch
import torch.optim as optim

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT
from utils.data_utils import (
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
    default_blocked_edge_types,
    default_rare_edge_types,
)
from utils.elliptic_plus_wallet_loader import load_elliptic_plus_wallet_dataset

# Reuse the real training logic from train.py (left untouched).
from train import (
    _subsample_hetero,
    evaluate_split,
    parse_args,
    train_step_no_gan,
    train_step_with_gan,
)


def _resolve_edge_type_set(raw: str, dataset: str, kind: str):
    """Mirror train.py's rare/blocked edge-type resolution (explicit > default > none)."""
    raw = (raw or "").strip()
    if raw.lower() == "none":
        return set(), "disabled"
    if raw:
        return {t.strip() for t in raw.split(",") if t.strip()}, "explicit"
    default = (default_rare_edge_types(dataset) if kind == "rare"
              else default_blocked_edge_types(dataset))
    return default, "dataset-default"


def main() -> None:
    args = parse_args()
    if args.dataset != "elliptic++":
        raise ValueError(
            "train_wallet.py only supports --dataset elliptic++ "
            f"(got {args.dataset!r}); the wallet target is Elliptic++-specific."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading dataset: elliptic++ (wallet target)")
    data, target_type = load_elliptic_plus_wallet_dataset(
        os.path.join(args.data_root, "Elliptic++"),
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )

    # Subsample before GPU transfer (keeps all fraud + random licit wallets).
    if args.subsample_tx > 0 and data[target_type].num_nodes > args.subsample_tx:
        data = _subsample_hetero(data, target_type, args.subsample_tx, seed=args.seed)

    data = data.to(device)

    labels = data[target_type].y
    train_mask = data[target_type].train_mask
    val_mask = data[target_type].val_mask
    test_mask = data[target_type].test_mask

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
        lambda_ncm=args.lambda_ncm,
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

    # ── Causal graph (seeded from fraud wallet training nodes) ──────────────
    print("Building TypedCausalGraph...")
    type_offsets = compute_type_offsets(data)
    train_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    fraud_global_ids = [
        type_offsets[target_type] + i
        for i in train_indices
        if labels[i].item() == 1
    ]
    seed_ids = fraud_global_ids[:20]

    rare_edge_types, rare_src = _resolve_edge_type_set(
        args.rare_edge_types, args.dataset, "rare")
    if rare_edge_types:
        print(f"  [BFS] rare edge types ({len(rare_edge_types)}, "
              f"reserve={args.rare_reserve}, max_hops={args.rare_max_hops}, "
              f"src={rare_src}): {sorted(rare_edge_types)}")
    blocked_edge_types, blocked_src = _resolve_edge_type_set(
        args.blocked_edge_types, args.dataset, "blocked")
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

    target_node = seed_ids[0] if seed_ids else type_offsets[target_type]
    for gid in seed_ids:
        if gid in causal_graph.set_v and causal_graph.parents(gid):
            target_node = gid
            break
    print(f"  Causal graph: {len(causal_graph.v)} nodes, "
          f"{len(causal_graph.edge_type_map)} directed edges")

    # ── Model ───────────────────────────────────────────────────────────────
    in_channels_dict = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types)
        if data[nt].x is not None
    }
    num_classes = int(labels.max().item()) + 1
    node_feature_dim = in_channels_dict.get(target_type)
    model = CI_RCT(
        config=config,
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        node_feature_dim=node_feature_dim if args.use_gan else None,
        use_gan=args.use_gan,
        num_classes=num_classes,
    ).to(device)

    # ── Optimisers ────────────────────────────────────────────────────────────
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

    # ── Class weight (inverse frequency over wallet labels) ───────────────────
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
        print("  Class weight: None (balanced / degenerate split)")

    # ── Fraud features for GAN (training fraud wallets) ──────────────────────
    fraud_features = None
    if args.use_gan:
        fraud_mask = (labels == 1) & train_mask
        if fraud_mask.any():
            fraud_features = data[target_type].x[fraud_mask].to(device)
        else:
            fraud_features = data[target_type].x[train_mask].to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    mode_str = "Phase 2 (GAN)" if args.use_gan else "Phase 1 (No GAN)"
    print(f"\nTraining [{mode_str}] for {args.epochs} epochs on {device} "
          f"(target={target_type})...")
    sel_key = {"macro_f1": "f1", "fraud_f1": "fraud_f1",
               "weighted_f1": "weighted_f1"}[args.early_stop_metric]
    best_val_score = 0.0
    ckpt_path = os.path.join(
        args.checkpoint_dir, f"ci_rct_{args.dataset}_wallet_best.pt"
    )
    print(f"  Model selection metric: {args.early_stop_metric} (val['{sel_key}'])")

    model.reset_phi_buffer()
    for epoch in range(1, args.epochs + 1):
        if not args.use_gan:
            total_loss, det_loss, stab_loss, ncm_loss = train_step_no_gan(
                model, data, labels, train_mask, optimizer_backbone,
                causal_graph, target_node, device,
                type_offsets, target_type, class_weight=class_weight,
                multi_task_labels=None,
                ncm_edge_balance=args.ncm_edge_balance,
            )
            loss_str = (f"Loss {total_loss:.4f} "
                        f"(det={det_loss:.4f}, stab={stab_loss:.2e}, ncm={ncm_loss:.4f})")
        else:
            total_loss, det_loss, adv_loss, g_loss, ncm_loss = train_step_with_gan(
                model, data, labels, train_mask,
                optimizer_backbone, optimizer_generator,
                causal_graph, fraud_features, topo_order, target_node,
                step_count=epoch, device=device,
                type_offsets=type_offsets, target_type=target_type,
                class_weight=class_weight,
            )
            loss_str = (f"Loss {total_loss:.4f} "
                        f"(det={det_loss:.4f}, adv={adv_loss:.4f}, "
                        f"G={g_loss:.4f}, ncm={ncm_loss:.4f})")

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_split(model, data, labels, val_mask)
            print(
                f"Epoch {epoch:03d} | {loss_str} | "
                f"Val macroF1={val_metrics['f1']:.4f}  "
                f"fraudF1={val_metrics['fraud_f1']:.4f}  "
                f"AUC={val_metrics['auc']:.4f}"
            )
            if val_metrics[sel_key] > best_val_score:
                best_val_score = val_metrics[sel_key]
                model.save_checkpoint(ckpt_path)
                print(f"  ↑ Best checkpoint saved (val {args.early_stop_metric}"
                      f"={best_val_score:.4f}) → {ckpt_path}")

    # ── Final test ────────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    model.load_checkpoint(ckpt_path)
    test_metrics = evaluate_split(model, data, labels, test_mask)
    print(f"Test (wallet)  macroF1={test_metrics['f1']:.4f}  "
          f"fraudF1={test_metrics['fraud_f1']:.4f}  AUC={test_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
