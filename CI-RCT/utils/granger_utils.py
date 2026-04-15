"""
Granger Causality Utilities for CI-RCT Metric C Evaluation.

Computes Granger-causal ground-truth from temporal network flow data
(UNSW-NB15) to enable Metric C (Explanation Quality) evaluation.

Core idea
─────────
Granger causality: X Granger-causes Y if past values of X significantly
help predict future values of Y, beyond Y's own past.

Applied to UNSW-NB15:
  1. Divide time into equal-width windows.
  2. For each IP, count attack flows per window → time series.
  3. For each pair (ip_i, ip_j), run Granger test.
  4. If ip_i Granger-causes ip_j (p < threshold), add causal edge.
  5. For each malicious target ip_j, collect its Granger-causal sources
     as the ground-truth causal set for Metric C.

Usage
─────
    from utils.granger_utils import compute_granger_ground_truth

    gt_causal_nodes = compute_granger_ground_truth(
        df=data._df,
        ip_global_offset=type_offsets["ip_node"],
        flow_global_offset=type_offsets["flow_node"],
        window_size=60,     # seconds per time window
        max_lag=3,
        p_threshold=0.05,
        min_observations=10,
    )
    # gt_causal_nodes: {target_flow_global_id: set_of_causal_ip_global_ids}
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_granger_ground_truth(
    df: pd.DataFrame,
    ip_global_offset: int,
    flow_global_offset: int,
    window_size: float = 60.0,
    max_lag: int = 3,
    p_threshold: float = 0.05,
    min_observations: int = 10,
    max_ip_pairs: int = 5000,
    verbose: bool = True,
) -> Dict[int, Set[int]]:
    """
    Compute Granger-causal ground-truth for Metric C.

    Args:
        df:                 Raw UNSW-NB15 DataFrame (from data._df)
        ip_global_offset:   Global node ID offset for ip_node type
        flow_global_offset: Global node ID offset for flow_node type
        window_size:        Time window width in seconds
        max_lag:            Maximum lag for Granger test
        p_threshold:        p-value threshold (default 0.05)
        min_observations:   Minimum non-zero windows for an IP to be included
        max_ip_pairs:       Cap on IP pairs to test (performance limit)
        verbose:            Print progress

    Returns:
        {target_flow_global_id: set_of_causal_ip_global_ids}
        Maps each attack flow to the set of IP global IDs that
        Granger-cause its source IP's attack activity.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        raise ImportError(
            "statsmodels is required for Granger causality. "
            "Install it with: pip install statsmodels"
        )

    if verbose:
        print("[Granger] Building IP time series...")

    ip_series, ip_to_local = _build_ip_time_series(
        df, window_size, min_observations
    )

    if verbose:
        print(f"[Granger] {len(ip_series)} IPs with sufficient observations.")

    if verbose:
        print(f"[Granger] Running Granger tests (max_pairs={max_ip_pairs})...")

    granger_edges = _run_granger_tests(
        ip_series, grangercausalitytests,
        max_lag, p_threshold, max_ip_pairs, verbose
    )

    if verbose:
        print(f"[Granger] Found {len(granger_edges)} causal IP→IP edges.")

    gt = _build_ground_truth(
        df, granger_edges, ip_to_local,
        ip_global_offset, flow_global_offset
    )

    if verbose:
        print(f"[Granger] Ground-truth entries: {len(gt)} attack flows.")

    return gt


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_ip_time_series(
    df: pd.DataFrame,
    window_size: float,
    min_observations: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Build per-IP attack activity time series.

    Returns:
        ip_series:   {ip_str: array of attack flow counts per window}
        ip_to_local: {ip_str: local ip_node index}
    """
    stime = pd.to_numeric(df["stime"], errors="coerce").fillna(0.0)
    t_min = stime.min()
    t_max = stime.max()

    if t_max <= t_min:
        return {}, {}

    n_windows = max(1, int(np.ceil((t_max - t_min) / window_size)))
    window_idx = ((stime - t_min) / window_size).astype(int).clip(0, n_windows - 1)

    df_work = df.copy()
    df_work["_window"] = window_idx.values
    df_work["_stime"]  = stime.values

    # Build ip → local index from the union of srcip and dstip
    all_ips = pd.concat([df["srcip"], df["dstip"]]).unique().tolist()
    ip_to_local: Dict[str, int] = {ip: i for i, ip in enumerate(all_ips)}

    ip_series: Dict[str, np.ndarray] = {}

    for ip, grp in df_work.groupby("srcip"):
        series = np.zeros(n_windows, dtype=np.float32)
        agg    = grp.groupby("_window")["Label"].sum()
        series[agg.index.values] = agg.values.astype(np.float32)

        # Only keep IPs with enough non-zero windows
        if int((series > 0).sum()) >= min_observations:
            ip_series[str(ip)] = series

    return ip_series, ip_to_local


def _run_granger_tests(
    ip_series: Dict[str, np.ndarray],
    grangercausalitytests,
    max_lag: int,
    p_threshold: float,
    max_ip_pairs: int,
    verbose: bool,
) -> List[Tuple[str, str]]:
    """
    Run pairwise Granger tests among candidate IPs.

    Only test pairs where both IPs have attack activity (series mean > 0).
    Returns list of (cause_ip, effect_ip) pairs that pass the test.
    """
    ips     = list(ip_series.keys())
    n       = len(ips)
    edges   = []
    tested  = 0
    skipped = 0

    # Candidate pairs: both IPs have attack activity
    candidates = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and ip_series[ips[i]].mean() > 0 and ip_series[ips[j]].mean() > 0
    ]

    # Shuffle and cap
    rng = np.random.default_rng(42)
    rng.shuffle(candidates)  # type: ignore[arg-type]
    candidates = candidates[:max_ip_pairs]

    for i, j in candidates:
        ip_cause  = ips[i]
        ip_effect = ips[j]

        x = ip_series[ip_cause]
        y = ip_series[ip_effect]

        data_matrix = np.column_stack([y, x])

        # Skip if either series is constant
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            skipped += 1
            continue

        try:
            result  = grangercausalitytests(data_matrix, maxlag=max_lag, verbose=False)
            # Use lag=1 F-test p-value as primary signal
            p_value = result[1][0]["ssr_ftest"][1]
            if p_value < p_threshold:
                edges.append((ip_cause, ip_effect))
        except Exception:
            skipped += 1
            continue

        tested += 1

    if verbose:
        print(f"[Granger]   Tested={tested}, Skipped={skipped}, "
              f"Causal pairs={len(edges)}")

    return edges


def _build_ground_truth(
    df: pd.DataFrame,
    granger_edges: List[Tuple[str, str]],
    ip_to_local: Dict[str, int],
    ip_global_offset: int,
    flow_global_offset: int,
) -> Dict[int, Set[int]]:
    """
    Map Granger-causal IP pairs → {attack_flow_global_id: causal_ip_global_ids}.

    For each attack flow f originating from ip_j (effect IP),
    the causal set = all ip_i that Granger-cause ip_j.
    """
    # Build effect_ip → set of cause_ips
    effect_to_causes: Dict[str, Set[str]] = {}
    for cause_ip, effect_ip in granger_edges:
        effect_to_causes.setdefault(effect_ip, set()).add(cause_ip)

    if not effect_to_causes:
        return {}

    gt: Dict[int, Set[int]] = {}

    for flow_local_idx, row in enumerate(df.itertuples(index=False)):
        if row.Label != 1:
            continue  # only attack flows

        src_ip = str(row.srcip)
        if src_ip not in effect_to_causes:
            continue

        flow_global_id = flow_global_offset + flow_local_idx
        causal_ips     = effect_to_causes[src_ip]

        causal_global_ids: Set[int] = set()
        for cip in causal_ips:
            local_idx = ip_to_local.get(cip)
            if local_idx is not None:
                causal_global_ids.add(ip_global_offset + local_idx)

        if causal_global_ids:
            gt[flow_global_id] = causal_global_ids

    return gt
