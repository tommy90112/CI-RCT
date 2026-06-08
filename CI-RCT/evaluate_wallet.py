"""
CI-RCT *wallet-target* evaluation entry point (Elliptic++ only).

Reports the wallet-only classification metrics (Metric A: F1-macro, fraud-F1,
AUC, recall) needed for the SAGE-FIN per-type comparison.  Root-cause /
explanation / stability metrics (B/C/D) are transaction-causal-chain specific
and intentionally out of scope here.

Reuses ``evaluate.py``'s ``eval_classification`` / ``print_section`` verbatim
(both target-agnostic — they classify whatever ``target_node_type`` the loaded
model was built with).  Nothing under ``model/`` or in ``evaluate.py`` is
modified.

Usage:
  python evaluate_wallet.py --dataset elliptic++ \
    --fraud_subgraph true --fraud_subgraph_hops 2 --seed 42 \
    --checkpoint checkpoints/wallet/ci_rct_elliptic++_wallet_best.pt

IMPORTANT: pass the SAME --fraud_subgraph / --include_addr_addr /
--fraud_subgraph_hops / --seed flags used for train_wallet.py, otherwise the
rebuilt wallet set and its test mask will not match the trained model.
"""
import os

import torch

from configs.config import CI_RCT_Config
from model.ci_rct import CI_RCT
from utils.elliptic_plus_wallet_loader import load_elliptic_plus_wallet_dataset
from utils.threshold_utils import sweep_best_threshold

# Reuse evaluate.py's argument parser + metric helpers (left untouched).
from evaluate import eval_classification, parse_args, print_section


def _make_arch_get(ckpt_arch):
    """Prefer the checkpoint's stored architecture; fall back to the CLI value."""
    def arch_get(key, cli_value):
        if ckpt_arch and ckpt_arch.get(key) is not None:
            stored = ckpt_arch[key]
            if stored != cli_value:
                print(f"  [arch] {key}: checkpoint={stored} (overrides CLI={cli_value})")
            return stored
        return cli_value
    return arch_get


def _resolve_threshold(args, model, data, labels):
    """Manual cut > val-sweep > argmax (None). Mirrors evaluate.py."""
    if args.threshold >= 0.0:
        print(f"\n[threshold] using manual cut = {args.threshold:.3f}")
        return args.threshold
    if args.threshold_tuning != "val":
        return None
    val_mask = getattr(data["wallet"], "val_mask", None)
    if val_mask is None:
        print("\n[threshold] val tuning requested but no wallet val_mask; using argmax.")
        return None
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward(data)
    val_scores = torch.softmax(logits[val_mask], dim=-1)[:, 1].cpu().numpy()
    val_true = labels[val_mask].cpu().numpy()
    thr, val_obj = sweep_best_threshold(
        val_scores, val_true, objective=args.threshold_objective
    )
    print(f"\n[threshold] val-tuned cut = {thr:.3f} "
          f"({args.threshold_objective}={val_obj:.4f} on wallet val)")
    return thr


def main() -> None:
    args = parse_args()
    if args.dataset != "elliptic++":
        raise ValueError(
            "evaluate_wallet.py only supports --dataset elliptic++ "
            f"(got {args.dataset!r})."
        )
    device = torch.device(args.device)

    # Restore architecture from the checkpoint (v2 format) so the rebuilt model
    # — including target_node_type="wallet" — matches the trained weights.
    ckpt_arch = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_arch = CI_RCT.read_arch_metadata(args.checkpoint, device=args.device)
    if ckpt_arch is None and args.checkpoint:
        print("  [arch] checkpoint has no embedded architecture (legacy format) "
              "— using CLI flags; ensure they match the training recipe.")
    arch_get = _make_arch_get(ckpt_arch)

    target_type = arch_get("target_node_type", "wallet")
    if target_type != "wallet":
        print(f"  [warn] checkpoint target_node_type={target_type!r}, not 'wallet'. "
              "This entry point expects a wallet-target model.")

    print("Loading dataset: elliptic++ (wallet target)")
    data, target_type = load_elliptic_plus_wallet_dataset(
        os.path.join(args.data_root, "Elliptic++"),
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )
    data = data.to(device)
    labels = data["wallet"].y
    test_mask = data["wallet"].test_mask

    config = CI_RCT_Config(
        dataset=args.dataset,
        target_node_type="wallet",
        max_hops=args.max_hops,
        ce_threshold=args.ce_threshold,
        top_k_paths=args.top_k,
        hidden_dim=arch_get("hidden_dim", args.hidden_dim),
        num_hgt_layers=arch_get("num_hgt_layers", args.num_hgt_layers),
        num_heads=arch_get("num_heads", args.num_heads),
        dropout=arch_get("dropout", args.dropout),
        node_type_emb_dim=arch_get("node_type_emb_dim", args.type_emb_dim),
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
        backbone_exclude_node_types=arch_get("backbone_exclude_node_types", []),
    ).to(device)

    if args.checkpoint:
        # Warm up lazy HGTConv per-relation weights before strict=False load
        # (same trap guarded against in evaluate.py).
        model.eval()
        with torch.no_grad():
            model.forward(data)
        model.load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — evaluating randomly initialised model (baseline).")

    threshold = _resolve_threshold(args, model, data, labels)
    n_test = int(test_mask.sum())
    metrics = eval_classification(model, data, labels, test_mask, threshold=threshold)
    print_section(f"A. Classification Metrics — wallet (test N={n_test:,})", metrics)
    print(f"\n{'─' * 55}\n  Wallet evaluation complete.\n{'─' * 55}\n")


if __name__ == "__main__":
    main()
