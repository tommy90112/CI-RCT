"""
Granger Causality Utilities for CI-RCT Metric C Evaluation on Elliptic++.

Core idea
─────────
Elliptic++ transactions have integer time steps (1–49).
For each wallet, we build a 49-dim time series where each value is the
total transaction count initiated by that wallet at each time step
(all transactions, not just illicit ones, to ensure a rich enough
time series for statistical testing).

Granger causality: wallet A Granger-causes wallet B if past illicit
activity from A significantly helps predict future illicit activity from B,
beyond B's own history.

Pipeline
────────
1. Load txs_features.csv  → txId → time_step mapping
2. Load txs_classes.csv   → txId → is_illicit flag
3. Load AddrTx_edgelist.csv → wallet → [txId] (wallet initiates tx)
4. Build per-wallet illicit-activity time series (49 time steps)
5. Filter wallets with sufficient non-zero time steps
6. Run pairwise Granger tests (capped at max_wallet_pairs)
7. Return {illicit_tx_global_id: set_of_causal_wallet_global_ids}

Usage
─────
    from utils.elliptic_granger_utils import compute_elliptic_granger_ground_truth

    gt = compute_elliptic_granger_ground_truth(
        data_root="data/elliptic++",
        tx_global_offset=type_offsets["transaction"],
        wallet_global_offset=type_offsets["wallet"],
    )
    # gt: {illicit_tx_global_id: set_of_causal_wallet_global_ids}
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_elliptic_granger_ground_truth(
    data_root: str,
    tx_global_offset: int,
    wallet_global_offset: int,
    max_lag: int = 3,
    p_threshold: float = 0.05,
    min_observations: int = 5,
    max_wallet_pairs: int = 5000,
    verbose: bool = True,
) -> Dict[int, Set[int]]:
    """
    Compute Granger-causal ground-truth for Metric C on Elliptic++.

    Args:
        data_root:            Path to the directory containing Elliptic++ CSVs.
        tx_global_offset:     Global node ID offset for transaction nodes.
        wallet_global_offset: Global node ID offset for wallet nodes.
        max_lag:              Maximum lag for Granger test (default 3).
        p_threshold:          p-value threshold for Granger test (default 0.05).
        min_observations:     Minimum non-zero time steps for a wallet to qualify.
        max_wallet_pairs:     Cap on wallet pairs to test (performance limit).
        verbose:              Print progress messages.

    Returns:
        {illicit_tx_global_id: set_of_causal_wallet_global_ids}
        Maps each illicit transaction to the set of wallet global IDs whose
        past illicit activity Granger-causes the tx's source wallet.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        raise ImportError(
            "statsmodels is required. Install with: pip install statsmodels"
        )

    root = Path(data_root)

    if verbose:
        print("[Elliptic Granger] Loading CSVs …")

    txs_feat  = pd.read_csv(root / "txs_features.csv", usecols=[0, 1])
    txs_feat.columns = ["txId", "time_step"]

    txs_cls = pd.read_csv(root / "txs_classes.csv")
    txs_cls.columns = [c.strip() for c in txs_cls.columns]

    addr_tx = pd.read_csv(root / "AddrTx_edgelist.csv")
    addr_tx.columns = [c.strip() for c in addr_tx.columns]

    wallets = pd.read_csv(root / "wallets_features.csv", usecols=[0])
    wallets.columns = ["address"]

    tx_addr = pd.read_csv(root / "TxAddr_edgelist.csv")
    tx_addr.columns = [c.strip() for c in tx_addr.columns]

    # ── Rebuild wallet ordering (must match loader) ──────────────────────────
    connected = (
        set(addr_tx.iloc[:, 0].dropna().astype(str))
        | set(tx_addr.iloc[:, 1].dropna().astype(str))
    )
    wallets_filt = wallets[wallets["address"].astype(str).isin(connected)].reset_index(drop=True)
    wallet_to_idx: Dict[str, int] = {
        str(addr): i for i, addr in enumerate(wallets_filt["address"].tolist())
    }

    # ── Rebuild tx ordering (must match loader) ──────────────────────────────
    tx_to_idx: Dict = {tid: i for i, tid in enumerate(txs_feat["txId"].tolist())}

    # ── Build illicit flag per txId ──────────────────────────────────────────
    cls_map: Dict = txs_cls.set_index("txId")["class"].to_dict()
    illicit_tx: Set = {tid for tid, cls in cls_map.items() if cls == 1}

    # ── Build per-wallet illicit time series (time steps 1–49) ──────────────
    if verbose:
        print("[Elliptic Granger] Building wallet time series …")

    time_step_map: Dict = txs_feat.set_index("txId")["time_step"].to_dict()

    # addr_tx columns: input_address (wallet), txId
    addr_col   = addr_tx.columns[0]
    txid_col   = addr_tx.columns[1]

    n_steps = 49
    wallet_series: Dict[str, np.ndarray] = {}

    for _, row in addr_tx.iterrows():
        wallet = str(row[addr_col])
        tx_id  = row[txid_col]
        if wallet not in wallet_to_idx:
            continue
        ts = int(time_step_map.get(tx_id, 0))
        if ts < 1 or ts > n_steps:
            continue
        if wallet not in wallet_series:
            wallet_series[wallet] = np.zeros(n_steps, dtype=np.float32)
        wallet_series[wallet][ts - 1] += 1.0

    # Filter wallets with sufficient non-zero time steps
    qualified: Dict[str, np.ndarray] = {
        w: s for w, s in wallet_series.items()
        if int((s > 0).sum()) >= min_observations
    }

    if verbose:
        print(f"[Elliptic Granger] {len(qualified)} wallets with ≥{min_observations} "
              f"non-zero time steps (out of {len(wallet_series)} active wallets).")

    if len(qualified) < 2:
        if verbose:
            print("[Elliptic Granger] Too few wallets — skipping Granger tests.")
        return {}

    # ── Run pairwise Granger tests ───────────────────────────────────────────
    if verbose:
        print(f"[Elliptic Granger] Running Granger tests "
              f"(max_pairs={max_wallet_pairs}) …")

    granger_edges = _run_granger_tests(
        qualified, grangercausalitytests,
        max_lag, p_threshold, max_wallet_pairs, verbose,
    )

    if verbose:
        print(f"[Elliptic Granger] Found {len(granger_edges)} causal wallet→wallet edges.")

    if not granger_edges:
        return {}

    # ── Build ground-truth mapping ───────────────────────────────────────────
    gt = _build_ground_truth(
        granger_edges, addr_tx, illicit_tx, time_step_map,
        tx_to_idx, wallet_to_idx,
        tx_global_offset, wallet_global_offset,
    )

    if verbose:
        print(f"[Elliptic Granger] Ground-truth entries: {len(gt)} illicit transactions.")

    return gt


# ── Private helpers ────────────────────────────────────────────────────────────

def _run_granger_tests(
    wallet_series: Dict[str, np.ndarray],
    grangercausalitytests,
    max_lag: int,
    p_threshold: float,
    max_wallet_pairs: int,
    verbose: bool,
) -> List[Tuple[str, str]]:
    """
    Run pairwise Granger tests among qualified wallets.

    Returns list of (cause_wallet, effect_wallet) pairs that pass the test.
    """
    wallets = list(wallet_series.keys())
    n       = len(wallets)
    edges: List[Tuple[str, str]] = []
    tested  = 0
    skipped = 0

    # All ordered pairs (i ≠ j)
    candidates = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    rng = np.random.default_rng(42)
    rng.shuffle(candidates)  # type: ignore[arg-type]
    candidates = candidates[:max_wallet_pairs]

    for i, j in candidates:
        cause_w  = wallets[i]
        effect_w = wallets[j]

        x = wallet_series[cause_w]
        y = wallet_series[effect_w]

        # Skip constant series
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            skipped += 1
            continue

        data_matrix = np.column_stack([y, x])

        try:
            result  = grangercausalitytests(data_matrix, maxlag=max_lag, verbose=False)
            p_value = result[1][0]["ssr_ftest"][1]
            if p_value < p_threshold:
                edges.append((cause_w, effect_w))
        except Exception:
            skipped += 1
            continue

        tested += 1

    if verbose:
        print(f"[Elliptic Granger]   Tested={tested}, Skipped={skipped}, "
              f"Causal pairs={len(edges)}")

    return edges


def _build_ground_truth(
    granger_edges: List[Tuple[str, str]],
    addr_tx: pd.DataFrame,
    illicit_tx: Set,
    time_step_map: Dict,
    tx_to_idx: Dict,
    wallet_to_idx: Dict[str, int],
    tx_global_offset: int,
    wallet_global_offset: int,
) -> Dict[int, Set[int]]:
    """
    Map Granger-causal wallet pairs → {illicit_tx_global_id: causal_wallet_global_ids}.

    For each illicit tx T initiated by effect_wallet B,
    the causal set = all cause_wallet A that Granger-cause B.
    """
    # effect_wallet → set of cause_wallets
    effect_to_causes: Dict[str, Set[str]] = {}
    for cause_w, effect_w in granger_edges:
        effect_to_causes.setdefault(effect_w, set()).add(cause_w)

    addr_col = addr_tx.columns[0]
    txid_col = addr_tx.columns[1]

    # wallet → illicit txIds it initiated
    wallet_to_illicit_txs: Dict[str, List] = {}
    for _, row in addr_tx.iterrows():
        wallet = str(row[addr_col])
        tx_id  = row[txid_col]
        if wallet not in effect_to_causes:
            continue
        if tx_id not in illicit_tx:
            continue
        wallet_to_illicit_txs.setdefault(wallet, []).append(tx_id)

    gt: Dict[int, Set[int]] = {}

    for effect_w, tx_ids in wallet_to_illicit_txs.items():
        cause_wallets = effect_to_causes[effect_w]
        causal_global_ids: Set[int] = set()
        for cw in cause_wallets:
            local_idx = wallet_to_idx.get(cw)
            if local_idx is not None:
                causal_global_ids.add(wallet_global_offset + local_idx)

        if not causal_global_ids:
            continue

        for tx_id in tx_ids:
            local_tx_idx = tx_to_idx.get(tx_id)
            if local_tx_idx is not None:
                tx_global_id = tx_global_offset + local_tx_idx
                gt[tx_global_id] = causal_global_ids

    return gt
