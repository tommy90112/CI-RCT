"""
UNSW-NB15 Heterogeneous Graph Loader for CI-RCT.

Builds a PyG HeteroData from the UNSW-NB15 network intrusion dataset:
    UNSW-NB15_1.csv ~ UNSW-NB15_4.csv  — network flow records

Graph schema
────────────
Node types
    flow_node : each network flow record; 47-dim feature vector
                (original 49 features minus srcip / dstip identifiers)
    ip_node   : unique IP addresses (src + dst merged);
                statistical aggregate features (8-dim)
    port_node : unique destination port numbers; 2-dim feature (port type)

Edge types (all directed)
    (ip_node,   initiates, flow_node)  — src_ip → flow
    (flow_node, targets,   ip_node)    — flow → dst_ip
    (flow_node, uses,      port_node)  — flow → dst_port

Labels
    flow_node.y = 1  if Label == 1 (attack)
    flow_node.y = 0  if Label == 0 (normal)

Train / val / test masks
    Stratified 70/15/15 split on flow_node.
    ip_node and port_node have no labels.

Scale note
    Full dataset has ~2.5M flows. Use max_flows to subsample for development.
    Stratified sampling keeps all attack flows + random normal flows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# ── Constants ──────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# Columns to drop when building flow features (identifiers, not features)
_DROP_COLS = ["srcip", "dstip", "sport", "dsport", "stime", "ltime",
              "attack_cat", "Label"]

# Categorical columns that need encoding
_CATEGORICAL_COLS = ["proto", "service", "state"]

# Timestamp column for Granger causality
TIMESTAMP_COL = "stime"

# Default CSV filenames (try both hyphen and underscore variants)
_CSV_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

# Column names (UNSW-NB15 CSVs have no header row)
_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def load_unsw_dataset(
    data_root: str,
    max_flows: int = 200_000,
    seed: int = 42,
) -> Tuple[HeteroData, str]:
    """
    Build and return (HeteroData, target_node_type).

    Args:
        data_root:  directory containing UNSW-NB15_*.csv files
        max_flows:  max number of flow records to keep (0 = no limit).
                    Stratified: keeps all attack flows + random normal flows.
        seed:       random seed for sampling

    Returns:
        (HeteroData, 'flow_node')
    """
    root = Path(data_root)

    print("Loading UNSW-NB15 dataset...")
    df = _load_csvs(root)
    print(f"  Raw records: {len(df):,}  "
          f"(attack={df['Label'].sum():,}, normal={(df['Label']==0).sum():,})")

    if max_flows > 0 and len(df) > max_flows:
        df = _stratified_sample(df, max_flows, seed)
        print(f"  After sampling: {len(df):,}  "
              f"(attack={df['Label'].sum():,}, normal={(df['Label']==0).sum():,})")

    df = df.reset_index(drop=True)

    print("  Building flow features...")
    flow_x, flow_y = _build_flow_features(df)

    print("  Building IP nodes...")
    ip_x, src_ip_idx, dst_ip_idx = _build_ip_features(df)

    print("  Building port nodes...")
    port_x, dst_port_idx = _build_port_features(df)

    print("  Building edges...")
    edge_ip_flow   = _build_edge_ip_initiates_flow(src_ip_idx)
    edge_flow_ip   = _build_edge_flow_targets_ip(dst_ip_idx)
    edge_flow_port = _build_edge_flow_uses_port(dst_port_idx)

    print("  Building masks...")
    train_mask, val_mask, test_mask = _build_masks(flow_y, seed)

    data = HeteroData()

    # ── Nodes ──
    data["flow_node"].x         = flow_x
    data["flow_node"].y         = flow_y
    data["flow_node"].train_mask = train_mask
    data["flow_node"].val_mask   = val_mask
    data["flow_node"].test_mask  = test_mask

    data["ip_node"].x   = ip_x
    data["port_node"].x = port_x

    # ── Edges ──
    data["ip_node",   "initiates", "flow_node"].edge_index = edge_ip_flow
    data["flow_node", "targets",   "ip_node"  ].edge_index = edge_flow_ip
    data["flow_node", "uses",      "port_node"].edge_index = edge_flow_port

    n_flow  = flow_x.size(0)
    n_ip    = ip_x.size(0)
    n_port  = port_x.size(0)
    n_edges = (edge_ip_flow.size(1) + edge_flow_ip.size(1)
               + edge_flow_port.size(1))

    print(f"  flow_node: {n_flow:,}  ip_node: {n_ip:,}  port_node: {n_port:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Attack flows: {flow_y.sum().item():,} / {n_flow:,} "
          f"({100*flow_y.float().mean().item():.1f}%)")

    # Store original df on data for Granger computation (detached from graph)
    data._df = df  # type: ignore[attr-defined]

    return data, "flow_node"


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_csvs(root: Path) -> pd.DataFrame:
    """Load one or more UNSW-NB15 CSV files."""
    frames: List[pd.DataFrame] = []

    # Try numbered files first
    for fname in _CSV_FILES:
        fpath = root / fname
        if fpath.exists():
            df = pd.read_csv(fpath, header=None, names=_COLUMNS,
                             low_memory=False, encoding="utf-8-sig")
            frames.append(df)

    # Fall back to a single combined file
    if not frames:
        for pattern in ["UNSW_NB15_training-set.csv",
                        "UNSW-NB15.csv", "unsw_nb15.csv"]:
            fpath = root / pattern
            if fpath.exists():
                df = pd.read_csv(fpath, low_memory=False, encoding="utf-8-sig")
                df.columns = [c.strip().lower() for c in df.columns]
                frames.append(df)
                break

    if not frames:
        raise FileNotFoundError(
            f"No UNSW-NB15 CSV files found in {root}. "
            "Expected UNSW-NB15_1.csv … UNSW-NB15_4.csv "
            "or UNSW-NB15_training-set.csv."
        )

    df = pd.concat(frames, ignore_index=True)

    # Coerce Label to int
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    df["Label"] = (df["Label"] > 0).astype(int)  # binary

    # Coerce stime to float
    df["stime"] = pd.to_numeric(df["stime"], errors="coerce").fillna(0.0)

    # Fix dsport: may contain hex strings (e.g. '0xc0a8') or '-'
    df["dsport"] = df["dsport"].apply(_parse_port_value)

    # Fix srcip: strip any leading BOM or whitespace
    df["srcip"] = df["srcip"].astype(str).str.strip()
    df["dstip"] = df["dstip"].astype(str).str.strip()

    return df


def _parse_port_value(val) -> int:
    """Convert port value to int, handling hex strings and '-'."""
    s = str(val).strip()
    if s in ("", "-", "nan"):
        return 0
    if s.lower().startswith("0x"):
        try:
            return int(s, 16) % 65536
        except ValueError:
            return 0
    try:
        return int(float(s)) % 65536
    except ValueError:
        return 0


def _stratified_sample(df: pd.DataFrame, max_flows: int, seed: int) -> pd.DataFrame:
    """Keep all attack flows + random sample of normal flows."""
    rng = np.random.default_rng(seed)

    attack_df = df[df["Label"] == 1]
    normal_df = df[df["Label"] == 0]

    n_normal_keep = max(0, max_flows - len(attack_df))
    if n_normal_keep < len(normal_df):
        idx = rng.choice(len(normal_df), size=n_normal_keep, replace=False)
        normal_df = normal_df.iloc[idx]

    return pd.concat([attack_df, normal_df], ignore_index=True)


def _build_flow_features(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build flow_node feature matrix and label vector."""
    feat_df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns],
                      errors="ignore")

    # Encode categorical columns
    for col in _CATEGORICAL_COLS:
        if col in feat_df.columns:
            feat_df[col] = pd.Categorical(feat_df[col]).codes.astype(float)

    # Fill NaN and convert
    feat_df = feat_df.fillna(0.0)
    x = torch.tensor(feat_df.values, dtype=torch.float32)

    # Normalise each feature column to [0, 1]
    col_min = x.min(dim=0).values
    col_max = x.max(dim=0).values
    denom   = (col_max - col_min).clamp(min=1e-8)
    x = (x - col_min) / denom

    y = torch.tensor(df["Label"].values, dtype=torch.long)
    return x, y


def _build_ip_features(
    df: pd.DataFrame,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Build ip_node feature matrix and per-flow IP index lists.

    Features per IP (8-dim):
        log_total_flows, attack_ratio, avg_sbytes, avg_dbytes,
        avg_dur, unique_dst_ports, avg_spkts, avg_dpkts
    """
    all_ips = pd.concat([df["srcip"], df["dstip"]]).unique().tolist()
    ip_to_idx: Dict[str, int] = {ip: i for i, ip in enumerate(all_ips)}

    n_ip = len(all_ips)
    feats = np.zeros((n_ip, 8), dtype=np.float32)

    for ip, grp in df.groupby("srcip"):
        idx = ip_to_idx[ip]
        feats[idx, 0] = np.log1p(len(grp))
        feats[idx, 1] = grp["Label"].mean()
        feats[idx, 2] = np.log1p(grp["sbytes"].clip(0).mean())
        feats[idx, 3] = np.log1p(grp["dbytes"].clip(0).mean())
        feats[idx, 4] = np.log1p(grp["dur"].clip(0).mean())
        feats[idx, 5] = np.log1p(grp["dsport"].nunique())
        feats[idx, 6] = np.log1p(grp["spkts"].clip(0).mean())
        feats[idx, 7] = np.log1p(grp["dpkts"].clip(0).mean())

    src_ip_idx = [ip_to_idx[ip] for ip in df["srcip"]]
    dst_ip_idx = [ip_to_idx[ip] for ip in df["dstip"]]

    return torch.tensor(feats, dtype=torch.float32), src_ip_idx, dst_ip_idx


def _build_port_features(
    df: pd.DataFrame,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Build port_node feature matrix and per-flow port index list.

    Features per port (2-dim):
        normalised_port_number, port_type (0=well-known, 1=registered, 2=dynamic)
    """
    ports = pd.to_numeric(df["dsport"], errors="coerce").fillna(0).astype(int)
    unique_ports = sorted(ports.unique().tolist())
    port_to_idx: Dict[int, int] = {p: i for i, p in enumerate(unique_ports)}

    def _port_type(p: int) -> int:
        if p < 1024:
            return 0   # well-known
        if p < 49152:
            return 1   # registered
        return 2       # dynamic / ephemeral

    n_port = len(unique_ports)
    feats  = np.zeros((n_port, 2), dtype=np.float32)
    for p, idx in port_to_idx.items():
        feats[idx, 0] = p / 65535.0
        feats[idx, 1] = _port_type(p) / 2.0

    dst_port_idx = [port_to_idx[int(p)] for p in ports]

    return torch.tensor(feats, dtype=torch.float32), dst_port_idx


def _build_edge_ip_initiates_flow(src_ip_idx: List[int]) -> torch.Tensor:
    """ip_node -[initiates]→ flow_node"""
    flow_idx = list(range(len(src_ip_idx)))
    return torch.tensor([src_ip_idx, flow_idx], dtype=torch.long)


def _build_edge_flow_targets_ip(dst_ip_idx: List[int]) -> torch.Tensor:
    """flow_node -[targets]→ ip_node"""
    flow_idx = list(range(len(dst_ip_idx)))
    return torch.tensor([flow_idx, dst_ip_idx], dtype=torch.long)


def _build_edge_flow_uses_port(dst_port_idx: List[int]) -> torch.Tensor:
    """flow_node -[uses]→ port_node"""
    flow_idx = list(range(len(dst_port_idx)))
    return torch.tensor([flow_idx, dst_port_idx], dtype=torch.long)


def _build_masks(
    y: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified 70/15/15 split."""
    rng   = np.random.default_rng(seed)
    n     = len(y)
    idx   = np.arange(n)

    attack_idx = idx[y.numpy() == 1]
    normal_idx = idx[y.numpy() == 0]

    def _split(arr: np.ndarray):
        rng.shuffle(arr)
        n_train = int(len(arr) * TRAIN_RATIO)
        n_val   = int(len(arr) * VAL_RATIO)
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    a_tr, a_va, a_te = _split(attack_idx)
    n_tr, n_va, n_te = _split(normal_idx)

    def _mask(indices):
        m = torch.zeros(n, dtype=torch.bool)
        m[indices] = True
        return m

    train_mask = _mask(np.concatenate([a_tr, n_tr]))
    val_mask   = _mask(np.concatenate([a_va, n_va]))
    test_mask  = _mask(np.concatenate([a_te, n_te]))

    return train_mask, val_mask, test_mask
