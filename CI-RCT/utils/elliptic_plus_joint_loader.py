"""
Elliptic++ *joint* loader for CI-RCT — transaction primary + wallet auxiliary.

Produces ONE HeteroData where BOTH node types carry clean binary labels and
train/val/test masks, so a single joint model (model.ci_rct_joint.CI_RCT_Joint)
can be trained to classify transactions AND wallets together:

    data["transaction"].{y, train_mask, val_mask, test_mask}  ← base loader
    data["wallet"].{y, train_mask, val_mask, test_mask}       ← clean overlay

The transaction side comes straight from the unchanged base loader
(``load_elliptic_plus_dataset``).  The wallet side reuses
``elliptic_plus_wallet_loader._build_clean_wallet_labels`` — the same per-node
(class1->1, class2->0, class3-excluded) logic the wallet-only pipeline uses,
which is row-order aligned with the base loader and duplicate-address safe.

Returns ``target_node_type="transaction"`` (the primary head); wallet is the
auxiliary head.  Nothing under ``model/`` or in the base loaders is modified.

IMPORTANT: train and eval must pass identical ``include_addr_addr`` /
``fraud_subgraph`` / ``fraud_subgraph_hops`` flags, or the wallet set / masks
will not reproduce.
"""
from __future__ import annotations

import os
from typing import Tuple

from torch_geometric.data import HeteroData

from utils.elliptic_plus_loader import _stratified_masks, load_elliptic_plus_dataset
from utils.elliptic_plus_wallet_loader import _build_clean_wallet_labels


def load_elliptic_plus_joint_dataset(
    data_root: str,
    include_addr_addr: bool = False,
    fraud_subgraph: bool = False,
    fraud_subgraph_hops: int = 2,
) -> Tuple[HeteroData, str]:
    """
    Build the Elliptic++ graph with transaction (primary) + wallet (auxiliary)
    both supervised.  Returns ``(data, "transaction")``.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Elliptic++ data_root does not exist: {data_root}")

    # 1. Transaction primary (y + masks) from the base loader. Wallet nodes are
    #    collapsed to one-per-address (canonical universe, leak-free wallet split);
    #    the transaction primary head is unaffected by the wallet dedup.
    data, target_type = load_elliptic_plus_dataset(
        data_root,
        include_addr_addr=include_addr_addr,
        fraud_subgraph=fraud_subgraph,
        fraud_subgraph_hops=fraud_subgraph_hops,
        wallet_per_address=True,
    )

    # 2. Clean wallet labels + stratified masks (same per-address dedup →
    #    row-order aligned with the base loader's data["wallet"]).
    from pathlib import Path

    all_wallet_ids, y_cls, labeled_idx = _build_clean_wallet_labels(
        Path(data_root),
        include_addr_addr=include_addr_addr,
        fraud_subgraph=fraud_subgraph,
        fraud_subgraph_hops=fraud_subgraph_hops,
        wallet_per_address=True,
    )
    n_wallets = int(data["wallet"].num_nodes)
    if len(all_wallet_ids) != n_wallets:
        raise RuntimeError(
            f"Reconstructed wallet node count ({len(all_wallet_ids):,}) does not "
            f"match data['wallet'].num_nodes ({n_wallets:,}). Check that "
            "include_addr_addr / fraud_subgraph / fraud_subgraph_hops are "
            "identical to the base loader call."
        )
    train_mask, val_mask, test_mask = _stratified_masks(y_cls, labeled_idx, n_wallets)

    # 3. Overlay wallet supervision (the base loader's polluted y is replaced by
    #    the clean one; the wallet split is added).
    data["wallet"].y = y_cls
    data["wallet"].train_mask = train_mask
    data["wallet"].val_mask = val_mask
    data["wallet"].test_mask = test_mask

    n_illicit = int(y_cls.eq(1).sum())
    print(
        f"  Joint wallet aux: illicit={n_illicit:,} / labeled={len(labeled_idx):,} "
        f"/ {n_wallets:,} wallets; split train={int(train_mask.sum()):,} "
        f"val={int(val_mask.sum()):,} test={int(test_mask.sum()):,}"
    )
    return data, target_type
