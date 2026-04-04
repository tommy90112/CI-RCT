"""
AWS Event Dataset Loader for CI-RCT.

Loads the crypto-exchange heterogeneous graph from CSV files:
  - data/gnn_node_list.csv   : nodes (user / wallet) with labels
  - data/gnn_edge_list.csv   : directed edges with edge_type
  - data/white_to_black.csv  : rich features for high-risk users
  - data/blacklist_analysis.csv : rich features for blacklisted users

Returns a PyG HeteroData object ready for CI-RCT training.

Node features
  - user  : 84-dim (risk_score + 83 behaviour features from feature tables;
             users without an entry get risk_score padded with zeros)
  - wallet: 1-dim  (constant 1.0  — wallets carry no additional attributes)

Splits: 60 / 20 / 20  stratified on user labels.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData


# ── Feature columns (excluding user_id) ──────────────────────────────────────

_FEAT_COLS = [
    "risk_score", "kyc_speed_sec", "account_age_days", "is_high_risk_career",
    "is_high_risk_income", "career_income_risk", "career_freq", "is_app_user",
    "reg_hour", "reg_is_night", "reg_is_weekend", "reg_to_kyc1_days",
    "twd_dep_count", "twd_dep_sum", "twd_dep_mean", "twd_dep_std",
    "twd_dep_max", "twd_wit_count", "twd_wit_std", "twd_wit_max",
    "twd_net_flow", "twd_withdraw_ratio", "twd_smurf_flag", "twd_wit_ip_ratio",
    "crypto_dep_count", "crypto_dep_mean", "crypto_dep_max", "crypto_wit_count",
    "crypto_wit_sum", "crypto_wit_mean", "crypto_wit_max",
    "crypto_currency_diversity", "crypto_protocol_diversity",
    "crypto_wallet_hash_nunique", "crypto_internal_count",
    "crypto_internal_peer_count", "crypto_wit_ip_ratio",
    "trading_mean", "trading_max", "trading_buy_ratio",
    "trading_market_order_ratio", "swap_count", "swap_sum",
    "ip_unique_count", "ip_night_ratio", "ip_max_shared", "fund_stay_sec",
    "pagerank_score", "connected_component_size", "total_tx_count",
    "first_to_last_tx_days", "weekend_tx_ratio", "dep_to_first_wit_hours",
    "twd_to_crypto_out_ratio", "tx_amount_cv", "rapid_kyc_then_trade",
    "crypto_out_in_ratio", "same_day_in_out_count", "tx_interval_mean",
    "tx_interval_std", "tx_interval_min", "tx_interval_median",
    "amount_p90_p10_ratio", "active_days", "if_score", "hbos_score",
    "lof_score",
    "gnn_emb_0", "gnn_emb_1", "gnn_emb_2", "gnn_emb_3", "gnn_emb_4",
    "gnn_emb_5", "gnn_emb_6", "gnn_emb_7", "gnn_emb_8", "gnn_emb_9",
    "gnn_emb_10", "gnn_emb_11", "gnn_emb_12", "gnn_emb_13",
    "gnn_emb_14", "gnn_emb_15",
]

_USER_FEAT_DIM = len(_FEAT_COLS)   # 83
_WALLET_FEAT_DIM = 1


def load_aws_dataset(data_root: str = "data") -> Tuple[HeteroData, str]:
    """
    Load the AWS crypto-exchange dataset.

    Args:
        data_root: directory containing the CSV files.

    Returns:
        (HeteroData, target_node_type)  where target_node_type == "user"
    """
    node_path  = os.path.join(data_root, "gnn_node_list.csv")
    edge_path  = os.path.join(data_root, "gnn_edge_list.csv")
    w2b_path   = os.path.join(data_root, "white_to_black.csv")
    bl_path    = os.path.join(data_root, "blacklist_analysis.csv")

    nodes = pd.read_csv(node_path)
    edges = pd.read_csv(edge_path, low_memory=False)

    # ── Build feature lookup {user_id (int): feature_array} ──────────────────
    feat_df = pd.concat(
        [pd.read_csv(w2b_path), pd.read_csv(bl_path)],
        ignore_index=True,
    ).drop_duplicates(subset=["user_id"])

    # Fill missing columns with 0
    for col in _FEAT_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    feat_df = feat_df[["user_id"] + _FEAT_COLS].copy()
    feat_df[_FEAT_COLS] = feat_df[_FEAT_COLS].fillna(0.0)
    feat_lookup = {
        int(row["user_id"]): row[_FEAT_COLS].values.astype(np.float32)
        for _, row in feat_df.iterrows()
    }

    # ── Separate node types ───────────────────────────────────────────────────
    user_nodes   = nodes[nodes["node_type"] == "user"].reset_index(drop=True)
    wallet_nodes = nodes[nodes["node_type"] == "wallet"].reset_index(drop=True)

    # Extract numeric part: "user_56" → 56
    user_ids_raw = user_nodes["node_id"].str.split("_").str[1].astype(int).tolist()

    # Build user feature matrix
    user_feats = np.zeros((len(user_nodes), _USER_FEAT_DIM), dtype=np.float32)
    for local_idx, uid in enumerate(user_ids_raw):
        if uid in feat_lookup:
            user_feats[local_idx] = feat_lookup[uid]
        else:
            # Fallback: use risk_score from node list, rest zeros
            user_feats[local_idx, 0] = float(user_nodes.loc[local_idx, "risk_score"])

    # Wallet features: constant 1.0
    wallet_feats = np.ones((len(wallet_nodes), _WALLET_FEAT_DIM), dtype=np.float32)

    # ── Node ID → local index maps ────────────────────────────────────────────
    user_id_map   = {nid: i for i, nid in enumerate(user_nodes["node_id"])}
    wallet_id_map = {nid: i for i, nid in enumerate(wallet_nodes["node_id"])}

    # ── Build HeteroData ──────────────────────────────────────────────────────
    data = HeteroData()

    data["user"].x   = torch.tensor(user_feats,   dtype=torch.float)
    data["wallet"].x = torch.tensor(wallet_feats, dtype=torch.float)

    # Labels: user nodes only (0 = normal, 1 = fraud)
    data["user"].y = torch.tensor(
        user_nodes["label"].fillna(0).astype(int).values, dtype=torch.long
    )

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_type_map = {
        "user_sends_wallet":    ("user",   "sends",    "wallet"),
        "wallet_funds_user":    ("wallet", "funds",    "user"),
        "user_transfers_user":  ("user",   "transfers","user"),
    }

    for raw_etype, (src_type, rel, dst_type) in edge_type_map.items():
        subset = edges[edges["edge_type"] == raw_etype]
        if subset.empty:
            continue

        src_map = user_id_map   if src_type == "user" else wallet_id_map
        dst_map = wallet_id_map if dst_type == "wallet" else user_id_map

        src_list, dst_list = [], []
        for _, row in subset.iterrows():
            s = str(row["source"])
            t = str(row["target"])
            if s in src_map and t in dst_map:
                src_list.append(src_map[s])
                dst_list.append(dst_map[t])

        if src_list:
            data[src_type, rel, dst_type].edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )

    # ── Train / Val / Test splits (stratified on user labels) ─────────────────
    labels = data["user"].y.numpy()
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=labels[temp_idx], random_state=42
    )

    n = len(labels)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data["user"].train_mask = train_mask
    data["user"].val_mask   = val_mask
    data["user"].test_mask  = test_mask

    return data, "user"
