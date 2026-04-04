"""
CryptoExchange Heterogeneous Graph Loader for CI-RCT.

Builds a PyG HeteroData from raw transaction CSVs of a crypto exchange:
    twd_transfer_train.csv      — TWD fiat deposits / withdrawals
    crypto_transfer_train.csv   — on-chain & internal crypto transfers
    usdt_twd_trading_train.csv  — order-book USDT/TWD trades
    usdt_swap_train.csv         — one-click buy/sell orders

Graph schema
────────────
Node types
    user   : 51 k users;  83-dim behavioural feature vector
    wallet : 85 k external wallet hashes;  4-dim degree/amount features

Edge types  (all directed, from crypto_transfer_train)
    (user,   sends,    wallet) — external withdrawal  (kind=1, sub_kind=0)
    (wallet, funds,    user)   — external deposit      (kind=0, sub_kind=0)
    (user,   transfers,user)   — internal transfer     (sub_kind=1)

Labels
    user.y = 1  if user_id appears in blacklist_analysis.csv or white_to_black.csv
    user.y = 0  otherwise

Train / val / test masks
    Stratified split (70 / 15 / 15) applied to ALL labeled users
    (both fraud AND normal nodes receive masks so the model trains on the full graph)
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ── Constants ──────────────────────────────────────────────────────────────────

SCALE = 1e-8          # monetary amounts stored ×1e8 in the DB
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15    # remainder becomes test


# ── Public API ─────────────────────────────────────────────────────────────────

def load_crypto_dataset(data_root: str) -> Tuple[HeteroData, str]:
    """
    Build and return (HeteroData, target_node_type).

    Args:
        data_root: directory that contains all six CSV files

    Returns:
        data:             PyG HeteroData ready for CI-RCT training
        target_node_type: "user"
    """
    root = Path(data_root)

    print("  Loading transaction CSVs …")
    twd      = pd.read_csv(root / "twd_transfer_train.csv")
    crypto   = pd.read_csv(root / "crypto_transfer_train.csv")
    trading  = pd.read_csv(root / "usdt_twd_trading_train.csv")
    swap     = pd.read_csv(root / "usdt_swap_train.csv")
    blacklist = pd.read_csv(root / "blacklist_analysis.csv")
    w2b       = pd.read_csv(root / "white_to_black.csv")

    # ── 1. Collect all user IDs ───────────────────────────────────────────────
    all_user_ids = sorted(set(
        twd["user_id"].tolist() +
        crypto["user_id"].tolist() +
        trading["user_id"].tolist() +
        swap["user_id"].tolist()
    ))
    user_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}
    n_users = len(all_user_ids)

    # ── 2. Collect all wallet hashes ─────────────────────────────────────────
    ext = crypto[crypto["sub_kind"] == 0]
    wallet_hashes = sorted(set(
        ext["from_wallet_hash"].dropna().tolist() +
        ext["to_wallet_hash"].dropna().tolist()
    ))
    wallet_to_idx = {w: i for i, w in enumerate(wallet_hashes)}
    n_wallets = len(wallet_hashes)

    # ── 3. User features ─────────────────────────────────────────────────────
    print("  Building user features …")
    user_feat = _build_user_features(
        twd, crypto, trading, swap, all_user_ids
    )                                                # [n_users, feat_dim]

    # ── 4. Wallet features ───────────────────────────────────────────────────
    wallet_feat = _build_wallet_features(
        ext, wallet_to_idx, n_wallets
    )                                                # [n_wallets, 4]

    # ── 5. Labels ─────────────────────────────────────────────────────────────
    fraud_ids = set(blacklist["user_id"].astype(int).tolist()) | \
                set(w2b["user_id"].astype(int).tolist())
    labels = torch.zeros(n_users, dtype=torch.long)
    for uid, idx in user_to_idx.items():
        if uid in fraud_ids:
            labels[idx] = 1

    # ── 6. Edges ──────────────────────────────────────────────────────────────
    print("  Building edges …")
    ei_user_wallet, ei_wallet_user, ei_user_user = _build_edges(
        crypto, user_to_idx, wallet_to_idx
    )

    # ── 7. Train / val / test masks ──────────────────────────────────────────
    train_mask, val_mask, test_mask = _stratified_masks(labels, n_users)

    # ── 8. Assemble HeteroData ───────────────────────────────────────────────
    data = HeteroData()

    data["user"].x          = user_feat
    data["user"].y          = labels
    data["user"].train_mask = train_mask
    data["user"].val_mask   = val_mask
    data["user"].test_mask  = test_mask

    data["wallet"].x = wallet_feat

    data["user",   "sends",     "wallet"].edge_index = ei_user_wallet
    data["wallet", "funds",     "user"  ].edge_index = ei_wallet_user
    data["user",   "transfers", "user"  ].edge_index = ei_user_user

    n_fraud  = int(labels.sum())
    n_normal = n_users - n_fraud
    print(f"  Users:   {n_users:,}  (fraud={n_fraud:,}, normal={n_normal:,})")
    print(f"  Wallets: {n_wallets:,}")
    print(f"  Edges:   user→wallet={ei_user_wallet.size(1):,}  "
          f"wallet→user={ei_wallet_user.size(1):,}  "
          f"user→user={ei_user_user.size(1):,}")

    return data, "user"


# ── Feature builders ───────────────────────────────────────────────────────────

def _build_user_features(
    twd: pd.DataFrame,
    crypto: pd.DataFrame,
    trading: pd.DataFrame,
    swap: pd.DataFrame,
    all_user_ids: list,
) -> torch.Tensor:
    """
    Aggregate per-user statistics from all four transaction tables.
    Returns a normalised float tensor of shape [n_users, feat_dim].
    """
    uid_series = pd.Series(all_user_ids, name="user_id")
    base = uid_series.to_frame()

    # --- TWD features ---
    twd_dep = twd[twd["kind"] == 0].groupby("user_id")["ori_samount"].agg(
        twd_dep_count="count", twd_dep_sum="sum", twd_dep_mean="mean"
    ) * SCALE
    twd_wit = twd[twd["kind"] == 1].groupby("user_id")["ori_samount"].agg(
        twd_wit_count="count", twd_wit_sum="sum", twd_wit_mean="mean"
    ) * SCALE
    # count cols should NOT be scaled
    twd_dep["twd_dep_count"] /= SCALE
    twd_wit["twd_wit_count"] /= SCALE

    # --- Crypto features ---
    cr_dep = crypto[(crypto["kind"] == 0) & (crypto["sub_kind"] == 0)]
    cr_wit = crypto[(crypto["kind"] == 1) & (crypto["sub_kind"] == 0)]
    cr_int = crypto[crypto["sub_kind"] == 1]

    cr_dep_agg = cr_dep.groupby("user_id")["ori_samount"].agg(
        cr_dep_count="count", cr_dep_sum="sum", cr_dep_mean="mean"
    )
    cr_dep_agg[["cr_dep_sum", "cr_dep_mean"]] *= SCALE

    cr_wit_agg = cr_wit.groupby("user_id")["ori_samount"].agg(
        cr_wit_count="count", cr_wit_sum="sum", cr_wit_mean="mean"
    )
    cr_wit_agg[["cr_wit_sum", "cr_wit_mean"]] *= SCALE

    cr_cur_div = crypto.groupby("user_id")["currency"].nunique().rename("cr_currency_div")
    cr_proto_div = crypto[crypto["sub_kind"] == 0].groupby("user_id")["protocol"].nunique().rename("cr_protocol_div")
    cr_int_count = cr_int.groupby("user_id").size().rename("cr_internal_count")
    cr_wallet_div = cr_wit.groupby("user_id")["to_wallet_hash"].nunique().rename("cr_wallet_div")

    # --- Trading features ---
    tr_agg = trading.groupby("user_id").agg(
        tr_count=("trade_samount", "count"),
        tr_sum=("trade_samount", "sum"),
        tr_mean=("trade_samount", "mean"),
        tr_buy_ratio=("is_buy", "mean"),
        tr_market_ratio=("is_market", "mean"),
    )
    tr_agg[["tr_sum", "tr_mean"]] *= SCALE

    # --- Swap features ---
    sw_agg = swap.groupby("user_id").agg(
        sw_count=("twd_samount", "count"),
        sw_twd_sum=("twd_samount", "sum"),
        sw_twd_mean=("twd_samount", "mean"),
    )
    sw_agg[["sw_twd_sum", "sw_twd_mean"]] *= SCALE

    # --- Merge all into base ---
    frames = [
        twd_dep, twd_wit,
        cr_dep_agg, cr_wit_agg,
        cr_cur_div, cr_proto_div, cr_int_count, cr_wallet_div,
        tr_agg, sw_agg,
    ]
    feat = base.set_index("user_id")
    for df in frames:
        feat = feat.join(df, how="left")

    feat = feat.fillna(0.0).astype(np.float32)

    # Standardise: zero-mean, unit-variance per column (fit on full dataset)
    arr = feat.values
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0          # avoid division by zero for constant columns
    arr = (arr - mean) / std

    return torch.tensor(arr, dtype=torch.float32)


def _build_wallet_features(
    ext: pd.DataFrame,
    wallet_to_idx: dict,
    n_wallets: int,
) -> torch.Tensor:
    """
    4-dim wallet features:
        [out_degree, in_degree, log1p(total_sent_TWD), log1p(total_received_TWD)]
    """
    feat = np.zeros((n_wallets, 4), dtype=np.float32)

    # user → wallet  (withdrawal: to_wallet_hash)
    wit = ext[ext["kind"] == 1][["to_wallet_hash", "ori_samount", "twd_srate"]].dropna()
    wit = wit[wit["to_wallet_hash"].isin(wallet_to_idx)]
    for _, row in wit.iterrows():
        i = wallet_to_idx[row["to_wallet_hash"]]
        feat[i, 0] += 1                                          # out_degree
        twd_val = float(row["ori_samount"]) * float(row["twd_srate"]) * SCALE * SCALE
        feat[i, 2] += twd_val                                    # total sent

    # wallet → user  (deposit: from_wallet_hash)
    dep = ext[ext["kind"] == 0][["from_wallet_hash", "ori_samount", "twd_srate"]].dropna()
    dep = dep[dep["from_wallet_hash"].isin(wallet_to_idx)]
    for _, row in dep.iterrows():
        i = wallet_to_idx[row["from_wallet_hash"]]
        feat[i, 1] += 1                                          # in_degree
        twd_val = float(row["ori_samount"]) * float(row["twd_srate"]) * SCALE * SCALE
        feat[i, 3] += twd_val                                    # total received

    feat[:, 2] = np.log1p(feat[:, 2])
    feat[:, 3] = np.log1p(feat[:, 3])

    return torch.tensor(feat, dtype=torch.float32)


# ── Edge builder ───────────────────────────────────────────────────────────────

def _build_edges(
    crypto: pd.DataFrame,
    user_to_idx: dict,
    wallet_to_idx: dict,
) -> tuple:
    """
    Returns three edge_index tensors:
        (user→wallet, wallet→user, user→user)
    """
    ext = crypto[crypto["sub_kind"] == 0]

    # user → wallet  (external withdrawal)
    wit = ext[(ext["kind"] == 1)][["user_id", "to_wallet_hash"]].dropna()
    wit = wit[wit["user_id"].isin(user_to_idx) & wit["to_wallet_hash"].isin(wallet_to_idx)]
    src_uw = torch.tensor([user_to_idx[u] for u in wit["user_id"]], dtype=torch.long)
    dst_uw = torch.tensor([wallet_to_idx[w] for w in wit["to_wallet_hash"]], dtype=torch.long)
    ei_user_wallet = torch.stack([src_uw, dst_uw], dim=0)

    # wallet → user  (external deposit)
    dep = ext[(ext["kind"] == 0)][["from_wallet_hash", "user_id"]].dropna()
    dep = dep[dep["from_wallet_hash"].isin(wallet_to_idx) & dep["user_id"].isin(user_to_idx)]
    src_wu = torch.tensor([wallet_to_idx[w] for w in dep["from_wallet_hash"]], dtype=torch.long)
    dst_wu = torch.tensor([user_to_idx[u] for u in dep["user_id"]], dtype=torch.long)
    ei_wallet_user = torch.stack([src_wu, dst_wu], dim=0)

    # user → user  (internal transfer)
    internal = crypto[crypto["sub_kind"] == 1][["user_id", "relation_user_id"]].dropna()
    internal = internal.copy()
    internal["relation_user_id"] = internal["relation_user_id"].astype(float).astype(int)
    internal = internal[
        internal["user_id"].isin(user_to_idx) &
        internal["relation_user_id"].isin(user_to_idx)
    ]
    src_uu = torch.tensor([user_to_idx[u] for u in internal["user_id"]], dtype=torch.long)
    dst_uu = torch.tensor([user_to_idx[u] for u in internal["relation_user_id"]], dtype=torch.long)
    ei_user_user = torch.stack([src_uu, dst_uu], dim=0)

    return ei_user_wallet, ei_wallet_user, ei_user_user


# ── Mask builder ───────────────────────────────────────────────────────────────

def _stratified_masks(
    labels: torch.Tensor,
    n_users: int,
) -> tuple:
    """
    Stratified 70/15/15 split across fraud AND normal users.
    Both classes are represented in every split.
    """
    rng = np.random.default_rng(42)

    train_mask = torch.zeros(n_users, dtype=torch.bool)
    val_mask   = torch.zeros(n_users, dtype=torch.bool)
    test_mask  = torch.zeros(n_users, dtype=torch.bool)

    for cls in [0, 1]:
        idx = np.where(labels.numpy() == cls)[0]
        rng.shuffle(idx)
        n      = len(idx)
        n_tr   = int(n * TRAIN_RATIO)
        n_val  = int(n * VAL_RATIO)

        train_mask[idx[:n_tr]]          = True
        val_mask  [idx[n_tr:n_tr+n_val]] = True
        test_mask [idx[n_tr+n_val:]]    = True

    return train_mask, val_mask, test_mask
