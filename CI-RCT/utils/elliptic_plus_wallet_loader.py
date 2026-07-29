"""
Elliptic++ *wallet-target* loader for CI-RCT.

This is a thin wrapper around the unchanged transaction-target loader
(``utils.elliptic_plus_loader.load_elliptic_plus_dataset``).  It builds the
exact same heterogeneous graph, then overlays **clean wallet labels** plus
stratified train/val/test masks and returns ``target_node_type="wallet"`` so
that the *same* CI-RCT pipeline (model/, train steps, eval) can be driven to
classify wallets — giving a wallet-only F1 for the SAGE-FIN per-type comparison.

Why a separate loader (instead of reusing ``data["wallet"].y``)?
    The base loader marks every non-illicit wallet — *including the ~557K
    class-3 "unknown" wallets* — as label 0, and gives wallets no masks.
    Training a classifier on that conflates "unknown" with "licit": an
    unusable, label-polluted target.  Here we instead produce:

        class 1 (illicit) -> y = 1   (kept in masks)
        class 2 (licit)   -> y = 0   (kept in masks)
        class 3 (unknown) -> y = 0   (placeholder; excluded from ALL masks)

    then a stratified 70/15/15 split over labeled (class 1 or 2) wallets only.

The base loader and everything under ``model/`` are left completely untouched;
this module only *reads* their helpers.

The per-node wallet ordering is reproduced by replaying the base loader's own
wallet construction — its ``_fraud_subgraph_wallets`` plus the identical
``connected`` / ``wallets_filt`` filtering — so the labels/masks line up 1-to-1
with ``data["wallet"]`` under identical flags.  **Train and eval must therefore
pass the same ``include_addr_addr`` / ``fraud_subgraph`` /
``fraud_subgraph_hops`` flags.**

Caveat (wallet-timestep nodes): Elliptic++'s ``wallets_features.csv`` has one
row per (wallet, timestep), so the same address appears as multiple nodes.  Like
the base loader, this loader treats every such row as its own node and labels it
by the wallet's class.  Different-timestep rows of one wallet can therefore land
in different splits — a wallet-level leakage that the base transaction pipeline
shares.  Acceptable for a like-for-like baseline comparison; revisit (dedup to
one node per address) if SAGE-FIN uses a wallet-level split.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from utils.elliptic_plus_loader import (
    _fraud_subgraph_wallets,
    _stratified_masks,
    load_elliptic_plus_dataset,
)
from utils.lfpn_utils import CLASS_ILLICIT, CLASS_LICIT, CLASS_UNKNOWN

_REQUIRED_CSVS = (
    "wallets_classes.csv",
    "wallets_features.csv",
    "txs_features.csv",
    "txs_classes.csv",
    "AddrTx_edgelist.csv",
    "TxAddr_edgelist.csv",
    "AddrAddr_edgelist.csv",
)


def load_elliptic_plus_wallet_dataset(
    data_root: str,
    include_addr_addr: bool = False,
    fraud_subgraph: bool = False,
    fraud_subgraph_hops: int = 2,
) -> Tuple[HeteroData, str]:
    """
    Build the Elliptic++ graph with **wallet** as the supervised target.

    Args:
        data_root:            directory containing the Elliptic++ CSV files
                              (same contract as ``load_elliptic_plus_dataset``).
        include_addr_addr:    include wallet→wallet edges (must match training).
        fraud_subgraph:       restrict wallets to BFS neighbours of labeled tx.
        fraud_subgraph_hops:  wallet hops from labeled tx for ``fraud_subgraph``.

    Returns:
        (data, "wallet") — ``data["wallet"]`` now carries clean binary ``y`` and
        ``train_mask`` / ``val_mask`` / ``test_mask`` over labeled wallets.
    """
    root = Path(data_root)
    _validate_data_root(root)

    # 1. Build the full graph via the unchanged base loader.
    data, _ = load_elliptic_plus_dataset(
        str(root),
        include_addr_addr=include_addr_addr,
        fraud_subgraph=fraud_subgraph,
        fraud_subgraph_hops=fraud_subgraph_hops,
    )

    # 2. Reproduce the base loader's per-node wallet ordering + clean labels.
    all_wallet_ids, y_cls, labeled_idx = _build_clean_wallet_labels(
        root,
        include_addr_addr=include_addr_addr,
        fraud_subgraph=fraud_subgraph,
        fraud_subgraph_hops=fraud_subgraph_hops,
    )

    n_wallets = int(data["wallet"].num_nodes)
    if len(all_wallet_ids) != n_wallets:
        raise RuntimeError(
            f"Reconstructed wallet node count ({len(all_wallet_ids):,}) does not "
            f"match data['wallet'].num_nodes ({n_wallets:,}). The base loader's "
            "wallet filtering and this loader's replay have drifted out of sync "
            "— wallet labels/masks would be misaligned. Check include_addr_addr "
            "/ fraud_subgraph / fraud_subgraph_hops flags are identical."
        )

    # 3. Stratified 70/15/15 over labeled wallets only (seed fixed at 42,
    #    same as the transaction split → reproducible across train/eval).
    train_mask, val_mask, test_mask = _stratified_masks(
        y_cls, labeled_idx, n_wallets
    )

    # 4. Overlay as the PRIMARY supervised fields for target_type="wallet".
    data["wallet"].y = y_cls
    data["wallet"].train_mask = train_mask
    data["wallet"].val_mask = val_mask
    data["wallet"].test_mask = test_mask

    _print_summary(y_cls, labeled_idx, train_mask, val_mask, test_mask)
    return data, "wallet"


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _validate_data_root(root: Path) -> None:
    """Fail fast with a clear message if any required CSV is missing."""
    if not root.exists():
        raise FileNotFoundError(f"Elliptic++ data_root does not exist: {root}")
    missing = [name for name in _REQUIRED_CSVS if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required Elliptic++ CSV(s) under {root}: {missing}"
        )


def _build_clean_wallet_labels(
    root: Path,
    include_addr_addr: bool,
    fraud_subgraph: bool,
    fraud_subgraph_hops: int,
) -> Tuple[List, torch.Tensor, List[int]]:
    """
    Return ``(all_wallet_ids, y_cls, labeled_idx)``:

        all_wallet_ids : per-node wallet address, in the SAME row order the base
                         loader assigns to ``data["wallet"]`` (length = n_wallets;
                         the same address may repeat across timestep rows).
        y_cls          : LongTensor[n_wallets] — 1 for illicit, 0 otherwise.
        labeled_idx    : node indices with a real label (class 1 or 2);
                         class-3 (unknown) nodes are excluded.

    The ``connected`` / ``wallets_filt`` filtering replays
    ``elliptic_plus_loader.load_elliptic_plus_dataset`` verbatim (reusing its own
    ``_fraud_subgraph_wallets``) so the ordering is guaranteed identical — no
    dedup, no ``astype`` coercion that could drift from the base loader.
    """
    wallets_cls = pd.read_csv(root / "wallets_classes.csv")
    wallets_cls.columns = [c.strip() for c in wallets_cls.columns]

    # First column of wallets_features.csv is the wallet address (in file order).
    wallets = pd.read_csv(root / "wallets_features.csv", usecols=[0])
    wallets.columns = ["address"]
    txs_feat = pd.read_csv(root / "txs_features.csv", usecols=[0])
    txs_feat.columns = ["txId"]
    txs_cls_df = pd.read_csv(root / "txs_classes.csv")
    txs_cls_df.columns = [c.strip() for c in txs_cls_df.columns]
    addr_tx = pd.read_csv(root / "AddrTx_edgelist.csv")
    addr_tx.columns = [c.strip() for c in addr_tx.columns]
    tx_addr = pd.read_csv(root / "TxAddr_edgelist.csv")
    tx_addr.columns = [c.strip() for c in tx_addr.columns]
    addr_addr = pd.read_csv(root / "AddrAddr_edgelist.csv")
    addr_addr.columns = [c.strip() for c in addr_addr.columns]

    tx_to_idx = {tid: i for i, tid in enumerate(txs_feat["txId"].tolist())}

    # ── connected-wallet set: identical computation to the base loader ────────
    if fraud_subgraph:
        connected = _fraud_subgraph_wallets(
            txs_cls_df, tx_to_idx, addr_tx, tx_addr, addr_addr,
            hops=fraud_subgraph_hops,
        )
    else:
        connected = (
            set(addr_tx["input_address"].dropna())
            | set(tx_addr["output_address"].dropna())
        )
        if include_addr_addr:
            connected |= (
                set(addr_addr["input_address"].dropna())
                | set(addr_addr["output_address"].dropna())
            )

    wallets_filt = wallets[wallets["address"].isin(connected)].reset_index(drop=True)
    all_wallet_ids = wallets_filt["address"].tolist()

    # Address → class (native dtype, matching the base loader's cls_map lookup).
    cls_map = dict(zip(wallets_cls["address"], wallets_cls["class"]))

    n_wallets = len(all_wallet_ids)
    y_cls = torch.zeros(n_wallets, dtype=torch.long)
    labeled_idx: List[int] = []
    for node_idx, addr in enumerate(all_wallet_ids):
        cls = cls_map.get(addr, CLASS_UNKNOWN)
        if cls == CLASS_ILLICIT:
            y_cls[node_idx] = 1
        if cls in (CLASS_ILLICIT, CLASS_LICIT):
            labeled_idx.append(node_idx)

    return all_wallet_ids, y_cls, labeled_idx


def _print_summary(
    y_cls: torch.Tensor,
    labeled_idx: List[int],
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> None:
    n_illicit = int(y_cls.eq(1).sum())
    n_labeled = len(labeled_idx)
    n_licit = n_labeled - n_illicit
    print(
        f"  Wallet target labels: illicit={n_illicit:,}  licit={n_licit:,}  "
        f"(labeled={n_labeled:,} / {y_cls.numel():,} wallets; "
        f"unknown excluded from masks)"
    )
    print(
        f"  Wallet split: train={int(train_mask.sum()):,}  "
        f"val={int(val_mask.sum()):,}  test={int(test_mask.sum()):,}"
    )
