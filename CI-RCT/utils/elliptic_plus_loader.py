"""
Elliptic++ Heterogeneous Graph Loader for CI-RCT.

Builds a PyG HeteroData from the Elliptic++ Bitcoin dataset (KDD 2023):
    txs_features.csv              — transaction node features (184 cols)
    txs_classes.csv               — transaction labels (1=illicit, 2=licit, 3=unknown)
    txs_edgelist.csv              — transaction → transaction edges
    AddrTx_edgelist.csv           — wallet → transaction edges (input)
    TxAddr_edgelist.csv           — transaction → wallet edges (output)
    AddrAddr_edgelist.csv         — wallet → wallet edges
    wallets_features.csv          — wallet node features (54 cols)

Graph schema
────────────
Node types
    transaction : ~203K nodes;  110-dim local feature vector
                  (93 Local_features + 17 stats; 72 Aggregate_features dropped)
    wallet      : connected wallets only (~900K);  54-dim feature vector

Edge types (all directed)
    (wallet,      sends,      transaction)  — AddrTx  (wallet is tx input)
    (transaction, pays,       wallet)       — TxAddr  (tx output to wallet)
    (transaction, flows_to,   transaction)  — txs_edgelist
    (wallet,      connects,   wallet)       — AddrAddr

Labels
    transaction.y = 1  if class == 1 (illicit / fraud)
    transaction.y = 0  if class == 2 (licit  / normal)
    class == 3 (unknown) → all masks False, excluded from supervised loss

Train / val / test masks
    Stratified 70/15/15 split on labeled transaction nodes only.
    Wallet nodes have no masks (no ground-truth labels used for training).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ── Constants ──────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# Column index ranges in txs_features.csv (0-based, after header)
# Col 0: txId  |  Col 1: Time step
# Cols  2-94  : Local_feature_1 .. Local_feature_93  (93 features, KEEP)
# Cols 95-166 : Aggregate_feature_1 .. Aggregate_feature_72  (72 features, DROP)
# Cols 167-183: in_txs_degree .. out_BTC_total  (17 features, KEEP)
_LOCAL_FEAT_COLS = list(range(2,  95))   # 93 local semantic features
_STATS_FEAT_COLS = list(range(167, 184)) # 17 transaction-level stats
_TX_FEAT_COLS    = _LOCAL_FEAT_COLS + _STATS_FEAT_COLS  # 110 total


# ── Public API ─────────────────────────────────────────────────────────────────

def load_elliptic_plus_dataset(
    data_root: str,
    include_addr_addr: bool = False,
    labeled_only: bool = False,
    fraud_subgraph: bool = False,
    fraud_subgraph_hops: int = 2,
) -> Tuple[HeteroData, str]:
    """
    Build and return (HeteroData, target_node_type).

    Args:
        data_root:            directory containing all Elliptic++ CSV files
        include_addr_addr:    whether to include wallet→wallet edges.
                              Default False to save GPU memory (removes 2.87M edges).
        labeled_only:         if True, keep only labeled tx nodes + their 1-hop
                              tx/wallet neighbors (~1/10 of full graph).
        fraud_subgraph:       if True, keep all tx nodes but restrict wallet nodes
                              to those within `fraud_subgraph_hops` hops of labeled
                              tx, and include addr→addr edges only within that
                              wallet subset. Dramatically reduces wallet count
                              from ~900K to tens of thousands while preserving
                              addr→addr structural signal near fraud nodes.
        fraud_subgraph_hops:  number of wallet hops from labeled tx (default 2).

    Returns:
        data:             PyG HeteroData ready for CI-RCT training
        target_node_type: "transaction"
    """
    root = Path(data_root)

    # ── 1. Load raw CSVs ──────────────────────────────────────────────────────
    print("  Loading CSVs …")
    txs_feat  = pd.read_csv(root / "txs_features.csv")
    txs_cls   = pd.read_csv(root / "txs_classes.csv")
    txs_edges = pd.read_csv(root / "txs_edgelist.csv")
    addr_tx   = pd.read_csv(root / "AddrTx_edgelist.csv")
    tx_addr   = pd.read_csv(root / "TxAddr_edgelist.csv")
    addr_addr = pd.read_csv(root / "AddrAddr_edgelist.csv")
    wallets   = pd.read_csv(root / "wallets_features.csv")

    # ── 2. Transaction node index ─────────────────────────────────────────────
    all_tx_ids  = txs_feat["txId"].tolist()
    tx_to_idx   = {tid: i for i, tid in enumerate(all_tx_ids)}
    n_tx        = len(all_tx_ids)

    # ── 3. Transaction features (110-dim) ────────────────────────────────────
    print("  Building transaction features …")
    tx_feat = _build_tx_features(txs_feat)   # [n_tx, 110]

    # ── 4. Transaction labels & masks ─────────────────────────────────────────
    print("  Building labels …")
    labels, train_mask, val_mask, test_mask = _build_tx_labels_and_masks(
        txs_cls, tx_to_idx, n_tx
    )

    # ── 4b. (Optional) restrict to labeled tx + 1-hop neighbors ─────────────
    if labeled_only:
        print("  [labeled_only] Filtering to labeled tx nodes + 1-hop neighbors …")
        cls_map = txs_cls.set_index("txId")["class"].to_dict()
        labeled_tx_ids = {tid for tid, cls in cls_map.items() if cls in (1, 2)}

        # 1-hop tx neighbors via tx→tx edges
        neighbor_tx_ids: set = set()
        for _, row in txs_edges.iterrows():
            if row["txId1"] in labeled_tx_ids or row["txId2"] in labeled_tx_ids:
                neighbor_tx_ids.add(row["txId1"])
                neighbor_tx_ids.add(row["txId2"])
        keep_tx_ids = labeled_tx_ids | neighbor_tx_ids

        # Keep only rows whose tx is in keep_tx_ids
        keep_mask = txs_feat["txId"].isin(keep_tx_ids)
        txs_feat  = txs_feat[keep_mask].reset_index(drop=True)
        txs_edges = txs_edges[
            txs_edges["txId1"].isin(keep_tx_ids) & txs_edges["txId2"].isin(keep_tx_ids)
        ].reset_index(drop=True)
        addr_tx   = addr_tx[addr_tx["txId"].isin(keep_tx_ids)].reset_index(drop=True)
        tx_addr   = tx_addr[tx_addr["txId"].isin(keep_tx_ids)].reset_index(drop=True)

        # Rebuild tx index from filtered features
        all_tx_ids = txs_feat["txId"].tolist()
        tx_to_idx  = {tid: i for i, tid in enumerate(all_tx_ids)}
        n_tx       = len(all_tx_ids)

        # Rebuild features and labels for filtered nodes
        tx_feat = _build_tx_features(txs_feat)
        labels, train_mask, val_mask, test_mask = _build_tx_labels_and_masks(
            txs_cls, tx_to_idx, n_tx
        )
        print(f"  [labeled_only] Kept {n_tx:,} transactions "
              f"(labeled={len(labeled_tx_ids):,}, 1-hop={len(neighbor_tx_ids - labeled_tx_ids):,})")

    # ── 5. Wallet node index (connected wallets only) ────────────────────────
    print("  Filtering connected wallets …")

    if fraud_subgraph:
        # Build wallet set anchored to labeled tx, then BFS over addr→addr
        connected = _fraud_subgraph_wallets(
            txs_cls, tx_to_idx, addr_tx, tx_addr, addr_addr,
            hops=fraud_subgraph_hops,
        )
        # addr→addr is always included in fraud_subgraph mode (that's the point)
        _include_aa = True
    else:
        connected = (
            set(addr_tx["input_address"].dropna())
            | set(tx_addr["output_address"].dropna())
        )
        _include_aa = include_addr_addr
        if _include_aa:
            connected |= (
                set(addr_addr["input_address"].dropna())
                | set(addr_addr["output_address"].dropna())
            )
    wallets_filt  = wallets[wallets["address"].isin(connected)].reset_index(drop=True)
    all_wallet_ids = wallets_filt["address"].tolist()
    wallet_to_idx  = {addr: i for i, addr in enumerate(all_wallet_ids)}
    n_wallets      = len(all_wallet_ids)

    # ── 6. Wallet features (54-dim) ───────────────────────────────────────────
    print("  Building wallet features …")
    wallet_feat = _build_wallet_features(wallets_filt)   # [n_wallets, 54]

    # ── 7. Build edges ────────────────────────────────────────────────────────
    print("  Building edges …")
    ei_wt, ei_tw, ei_tt, ei_ww = _build_edges(
        addr_tx, tx_addr, txs_edges,
        addr_addr if _include_aa else None,
        tx_to_idx, wallet_to_idx,
    )

    # ── 8. Assemble HeteroData ────────────────────────────────────────────────
    data = HeteroData()

    data["transaction"].x          = tx_feat
    data["transaction"].y          = labels
    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask   = val_mask
    data["transaction"].test_mask  = test_mask

    data["wallet"].x = wallet_feat

    data["wallet",       "sends",    "transaction"].edge_index = ei_wt
    data["transaction",  "pays",     "wallet"      ].edge_index = ei_tw
    data["transaction",  "flows_to", "transaction" ].edge_index = ei_tt
    data["wallet",       "connects", "wallet"       ].edge_index = ei_ww

    # ── 9. Summary ────────────────────────────────────────────────────────────
    n_illicit = int(labels.eq(1).sum())
    n_licit   = int(labels.eq(0).sum())
    n_unknown = n_tx - n_illicit - n_licit
    print(f"  Transactions: {n_tx:,}  "
          f"(illicit={n_illicit:,}, licit={n_licit:,}, unknown={n_unknown:,})")
    print(f"  Wallets (connected): {n_wallets:,}")
    print(f"  Edges:  wallet→tx={ei_wt.size(1):,}  "
          f"tx→wallet={ei_tw.size(1):,}  "
          f"tx→tx={ei_tt.size(1):,}  "
          f"wallet→wallet={ei_ww.size(1):,}")

    return data, "transaction"


# ── Feature builders ───────────────────────────────────────────────────────────

def _build_tx_features(txs_feat: pd.DataFrame) -> torch.Tensor:
    """
    Extract 110 local transaction features (93 local + 17 stats),
    drop the 72 Aggregate_features to prevent GNN message-passing leakage.

    Standardises: zero-mean, unit-variance (fit on full dataset).
    """
    arr = txs_feat.iloc[:, _TX_FEAT_COLS].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0
    arr = (arr - mean) / std

    return torch.tensor(arr, dtype=torch.float32)


def _build_wallet_features(wallets: pd.DataFrame) -> torch.Tensor:
    """
    All 54 wallet behavioural features (cols 2 onwards), standardised.
    """
    feat_cols = [c for c in wallets.columns if c not in ("address", "Time step")]
    arr = wallets[feat_cols].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0
    arr = (arr - mean) / std

    return torch.tensor(arr, dtype=torch.float32)


# ── Label & mask builder ───────────────────────────────────────────────────────

def _build_tx_labels_and_masks(
    txs_cls:   pd.DataFrame,
    tx_to_idx: dict,
    n_tx:      int,
) -> tuple:
    """
    Map class labels to binary targets.
        class 1 (illicit)  → y = 1
        class 2 (licit)    → y = 0
        class 3 (unknown)  → y = 0  (placeholder; excluded by masks)

    Stratified 70/15/15 split on labeled nodes only.
    Unknown nodes get all-False masks.
    """
    labels = torch.zeros(n_tx, dtype=torch.long)

    cls_map = txs_cls.set_index("txId")["class"].to_dict()
    for tid, idx in tx_to_idx.items():
        cls = cls_map.get(tid, 3)
        if cls == 1:
            labels[idx] = 1

    # Labeled indices (class 1 or 2 only)
    labeled_idx = [
        tx_to_idx[tid]
        for tid, cls in cls_map.items()
        if cls in (1, 2) and tid in tx_to_idx
    ]

    train_mask, val_mask, test_mask = _stratified_masks(
        labels, labeled_idx, n_tx
    )
    return labels, train_mask, val_mask, test_mask


def _stratified_masks(
    labels:      torch.Tensor,
    labeled_idx: list,
    n_tx:        int,
) -> tuple:
    """Stratified 70/15/15 split across illicit and licit nodes."""
    rng = np.random.default_rng(42)

    train_mask = torch.zeros(n_tx, dtype=torch.bool)
    val_mask   = torch.zeros(n_tx, dtype=torch.bool)
    test_mask  = torch.zeros(n_tx, dtype=torch.bool)

    labeled_arr = np.array(labeled_idx)

    for cls in [0, 1]:
        idx = labeled_arr[labels[labeled_arr].numpy() == cls]
        rng.shuffle(idx)
        n     = len(idx)
        n_tr  = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_mask[idx[:n_tr]]           = True
        val_mask  [idx[n_tr:n_tr+n_val]] = True
        test_mask [idx[n_tr+n_val:]]     = True

    return train_mask, val_mask, test_mask


# ── Fraud subgraph wallet set builder ──────────────────────────────────────────

def _fraud_subgraph_wallets(
    txs_cls:   "pd.DataFrame",
    tx_to_idx: dict,
    addr_tx:   "pd.DataFrame",
    tx_addr:   "pd.DataFrame",
    addr_addr: "pd.DataFrame",
    hops:      int = 2,
) -> set:
    """
    BFS outward from labeled (fraud + licit) tx nodes to collect wallet IDs.

    Hop 1: wallets that directly send to / receive from labeled tx
           (via AddrTx input_address and TxAddr output_address)
    Hop 2+: wallets reachable via addr→addr edges from hop-1 wallets

    Returns:
        set of wallet address strings to keep
    """
    # Labeled tx IDs
    cls_map = txs_cls.set_index("txId")["class"].to_dict()
    labeled_tx = {tid for tid, cls in cls_map.items() if cls in (1, 2) and tid in tx_to_idx}

    # Hop 1: wallets connected directly to labeled tx
    seed_wallets: set = set()
    at = addr_tx.dropna(subset=["input_address", "txId"])
    seed_wallets |= set(at.loc[at["txId"].isin(labeled_tx), "input_address"])
    ta = tx_addr.dropna(subset=["txId", "output_address"])
    seed_wallets |= set(ta.loc[ta["txId"].isin(labeled_tx), "output_address"])

    if hops <= 1 or addr_addr is None or addr_addr.empty:
        print(f"  [fraud_subgraph] hop-1 wallets: {len(seed_wallets):,}")
        return seed_wallets

    # Precompute adjacency from addr_addr for fast BFS
    aa = addr_addr.dropna(subset=["input_address", "output_address"])
    adj: dict = {}
    for src, dst in zip(aa["input_address"], aa["output_address"]):
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)

    # BFS for remaining hops
    frontier = set(seed_wallets)
    visited  = set(seed_wallets)
    for hop in range(2, hops + 1):
        next_frontier: set = set()
        for w in frontier:
            for nb in adj.get(w, ()):
                if nb not in visited:
                    next_frontier.add(nb)
        visited |= next_frontier
        frontier = next_frontier
        print(f"  [fraud_subgraph] hop-{hop} wallets added: {len(next_frontier):,}  "
              f"(total so far: {len(visited):,})")
        if not frontier:
            break

    print(f"  [fraud_subgraph] Total wallets in subgraph: {len(visited):,} "
          f"(vs full ~900K)")
    return visited


# ── Edge builder ───────────────────────────────────────────────────────────────

def _build_edges(
    addr_tx:       pd.DataFrame,
    tx_addr:       pd.DataFrame,
    txs_edges:     pd.DataFrame,
    addr_addr:     pd.DataFrame | None,
    tx_to_idx:     dict,
    wallet_to_idx: dict,
) -> tuple:
    """
    Returns four edge_index tensors:
        (wallet→tx, tx→wallet, tx→tx, wallet→wallet)
    All edges whose endpoints are not in the node index are silently dropped.
    """
    # wallet → transaction  (addr is tx input / sender)
    wt = addr_tx.dropna(subset=["input_address", "txId"])
    wt = wt[wt["input_address"].isin(wallet_to_idx) & wt["txId"].isin(tx_to_idx)]
    src_wt = torch.tensor([wallet_to_idx[a] for a in wt["input_address"]], dtype=torch.long)
    dst_wt = torch.tensor([tx_to_idx[t]     for t in wt["txId"]],          dtype=torch.long)
    ei_wt  = torch.stack([src_wt, dst_wt], dim=0)

    # transaction → wallet  (tx output to receiving addr)
    tw = tx_addr.dropna(subset=["txId", "output_address"])
    tw = tw[tw["txId"].isin(tx_to_idx) & tw["output_address"].isin(wallet_to_idx)]
    src_tw = torch.tensor([tx_to_idx[t]     for t in tw["txId"]],            dtype=torch.long)
    dst_tw = torch.tensor([wallet_to_idx[a] for a in tw["output_address"]], dtype=torch.long)
    ei_tw  = torch.stack([src_tw, dst_tw], dim=0)

    # transaction → transaction
    tt = txs_edges.dropna(subset=["txId1", "txId2"])
    tt = tt[tt["txId1"].isin(tx_to_idx) & tt["txId2"].isin(tx_to_idx)]
    src_tt = torch.tensor([tx_to_idx[t] for t in tt["txId1"]], dtype=torch.long)
    dst_tt = torch.tensor([tx_to_idx[t] for t in tt["txId2"]], dtype=torch.long)
    ei_tt  = torch.stack([src_tt, dst_tt], dim=0)

    # wallet → wallet (optional, disabled for local CPU testing)
    if addr_addr is not None:
        ww = addr_addr.dropna(subset=["input_address", "output_address"])
        ww = ww[ww["input_address"].isin(wallet_to_idx) & ww["output_address"].isin(wallet_to_idx)]
        src_ww = torch.tensor([wallet_to_idx[a] for a in ww["input_address"]],  dtype=torch.long)
        dst_ww = torch.tensor([wallet_to_idx[a] for a in ww["output_address"]], dtype=torch.long)
        ei_ww  = torch.stack([src_ww, dst_ww], dim=0)
    else:
        ei_ww = torch.zeros((2, 0), dtype=torch.long)

    return ei_wt, ei_tw, ei_tt, ei_ww
