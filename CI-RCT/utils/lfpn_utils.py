"""
LFPN (Labeled Fraud Propagation Neighborhood) Utilities
───────────────────────────────────────────────────────
Metric C ground-truth builder for Elliptic++, tailored to CI-RCT's
backward root-cause tracing semantics.

Why not Granger causality?
──────────────────────────
Elliptic++ transaction timestamps are integer time steps in [1, 49] and
most wallets appear in only a handful of time steps — far too sparse
for autoregressive Granger tests to have non-trivial statistical power.
In practice the Granger-based builder yielded only ~10 causal edges
covering ~25 wallets and 32 transactions, and Metric C collapsed to
EA = ER = 0 because the tracer's paths almost never intersected those
25 wallets.

Why LFPN?
─────────
LFPN discards time-series entirely and defines the ground-truth purely
from two sources that are robust for this dataset:

  (1) Direct structural attribution
      W_direct(T) = { w : AddrTx(w, T) }     — who initiated T

  (2) Labeled fraud propagation (multi-hop)
      LFPN_k(T)  = { w : w is a labeled illicit wallet (wallets_classes
                         class == 1)
                         AND dist_{addr→addr}(w, W_direct(T)) ≤ k }

  GT_strict(T)   = W_direct(T)                        — conservative
  GT_extended(T) = W_direct(T) ∪ LFPN_k(T)            — multi-hop

Both definitions are aligned with CI-RCT's tracer which follows
wallet↔tx backward edges: GT_strict evaluates 1-hop attribution
quality, GT_extended evaluates multi-hop tracing quality.

References
──────────
  • Elmougy & Liu, "Demystifying Fraudulent Transactions and Illicit
    Nodes in the Bitcoin Network" (KDD 2023).
    Labels in wallets_classes.csv follow the same schema as the
    underlying Elliptic dataset: 1 = illicit, 2 = licit, 3 = unknown.
  • Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting
    with Graph Convolutional Networks for Financial Forensics",
    KDD 2019 AMLW — origin of the 1/2/3 class schema.

Public API
──────────
  compute_lfpn_ground_truth(
      data_root, tx_global_offset, wallet_global_offset,
      mode="strict" | "extended",
      k_hops=2,
      include_addr_addr=False,
      fraud_subgraph=False,
      fraud_subgraph_hops=2,
  ) -> Dict[int, Set[int]]
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ── Label schema (inherited from Elliptic, Weber et al. 2019 / KDD) ──────────

CLASS_ILLICIT = 1
CLASS_LICIT   = 2
CLASS_UNKNOWN = 3


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def compute_lfpn_ground_truth(
    data_root: str,
    tx_global_offset: int,
    wallet_global_offset: int,
    mode: str = "extended",
    k_hops: int = 2,
    include_addr_addr: bool = False,
    fraud_subgraph: bool = False,
    fraud_subgraph_hops: int = 2,
    verbose: bool = True,
) -> Dict[int, Set[int]]:
    """
    Build Metric C ground-truth for Elliptic++.

    Parameters
    ----------
    data_root
        Path to the Elliptic++ directory (contains txs_classes.csv,
        wallets_classes.csv, AddrTx_edgelist.csv, AddrAddr_edgelist.csv,
        txs_features.csv, wallets_features.csv, TxAddr_edgelist.csv).

    tx_global_offset, wallet_global_offset
        Global-ID offsets used by build_typed_causal_graph_from_hetero.
        Must match the offsets produced by compute_type_offsets(data) for
        the *same* loading flags (include_addr_addr, fraud_subgraph, ...).

    mode
        "strict"   : GT(T) = W_direct(T).
        "extended" : GT(T) = W_direct(T) ∪ LFPN_{k_hops}(T).

    k_hops
        BFS depth over AddrAddr edges for extended mode. Ignored in strict.

    include_addr_addr, fraud_subgraph, fraud_subgraph_hops
        Loader flags. Must match the flags passed to
        load_elliptic_plus_dataset so the wallet_to_idx mapping built
        here is identical to the one the loader produced — otherwise
        global IDs drift and GT points at the wrong wallets.

    verbose
        Print progress / diagnostics.

    Returns
    -------
    gt : Dict[int, Set[int]]
        { illicit_tx_global_id : { wallet_global_id, ... } }.
        Only contains entries for illicit tx that have at least one
        resolvable wallet in the GT set; tx with no direct initiators
        present in the loader's wallet subset are skipped.
    """
    if mode not in ("strict", "extended"):
        raise ValueError(f"mode must be 'strict' or 'extended', got {mode!r}")
    if k_hops < 1:
        raise ValueError(f"k_hops must be >= 1, got {k_hops}")

    root = Path(data_root)

    if verbose:
        print(f"[LFPN] mode={mode}"
              f"{f', k_hops={k_hops}' if mode == 'extended' else ''}"
              f", include_addr_addr={include_addr_addr}"
              f", fraud_subgraph={fraud_subgraph}")
        print("[LFPN] Loading CSVs ...")

    # ── 1. Load all CSVs that feed either the index reconstruction or
    #      the GT logic itself.
    txs_feat  = pd.read_csv(root / "txs_features.csv", usecols=[0])
    txs_feat.columns = ["txId"]

    txs_cls   = pd.read_csv(root / "txs_classes.csv")
    txs_cls.columns = [c.strip() for c in txs_cls.columns]

    wallets   = pd.read_csv(root / "wallets_features.csv", usecols=[0])
    wallets.columns = ["address"]

    wallets_cls = pd.read_csv(root / "wallets_classes.csv")
    wallets_cls.columns = [c.strip() for c in wallets_cls.columns]

    addr_tx   = pd.read_csv(root / "AddrTx_edgelist.csv")
    addr_tx.columns = [c.strip() for c in addr_tx.columns]

    tx_addr   = pd.read_csv(root / "TxAddr_edgelist.csv")
    tx_addr.columns = [c.strip() for c in tx_addr.columns]

    addr_addr = pd.read_csv(root / "AddrAddr_edgelist.csv")
    addr_addr.columns = [c.strip() for c in addr_addr.columns]

    # ── 2. Rebuild tx_to_idx exactly as the loader does: ordering follows
    #      rows in txs_features.csv.  (Unknown tx stay in the index because
    #      the loader keeps them too — only labels/masks differ.)
    tx_to_idx: Dict = {tid: i for i, tid in enumerate(txs_feat["txId"].tolist())}

    # ── 3. Rebuild wallet_to_idx exactly as the loader does, honouring the
    #      same three CLI flags.
    wallet_to_idx = _rebuild_wallet_to_idx(
        wallets=wallets,
        wallets_cls=wallets_cls,
        txs_cls=txs_cls,
        tx_to_idx=tx_to_idx,
        addr_tx=addr_tx,
        tx_addr=tx_addr,
        addr_addr=addr_addr,
        include_addr_addr=include_addr_addr,
        fraud_subgraph=fraud_subgraph,
        fraud_subgraph_hops=fraud_subgraph_hops,
        verbose=verbose,
    )
    if verbose:
        print(f"[LFPN] wallet_to_idx built: {len(wallet_to_idx):,} wallets")

    # ── 4. Extract illicit tx IDs.
    cls_map_tx: Dict = txs_cls.set_index("txId")["class"].to_dict()
    illicit_tx_ids: Set = {
        tid for tid, cls in cls_map_tx.items() if cls == CLASS_ILLICIT
    }
    if verbose:
        print(f"[LFPN] Illicit tx (class==1): {len(illicit_tx_ids):,}")

    # ── 5. Extract labeled illicit wallet addresses.
    cls_map_w: Dict = wallets_cls.set_index("address")["class"].to_dict()
    illicit_wallets: Set[str] = {
        str(addr) for addr, cls in cls_map_w.items() if cls == CLASS_ILLICIT
    }
    if verbose:
        n_in_idx = sum(1 for a in illicit_wallets if a in wallet_to_idx)
        print(f"[LFPN] Illicit wallets (class==1): {len(illicit_wallets):,} "
              f"(of which {n_in_idx:,} are in the loader's wallet subset)")

    # ── 6. Build { illicit_tx_id -> set of direct-initiator wallet addrs }.
    addr_tx_illicit = addr_tx[addr_tx["txId"].isin(illicit_tx_ids)]
    w_direct: Dict = {}
    for addr, tid in zip(addr_tx_illicit["input_address"],
                         addr_tx_illicit["txId"]):
        if pd.isna(addr) or pd.isna(tid):
            continue
        w_direct.setdefault(tid, set()).add(str(addr))
    if verbose:
        n_with_direct = len(w_direct)
        print(f"[LFPN] Illicit tx with ≥1 direct initiator: "
              f"{n_with_direct:,} / {len(illicit_tx_ids):,}")

    # ── 7. Strict mode: just convert W_direct to global IDs and return.
    if mode == "strict":
        gt = _finalize_gt(
            w_direct_map=w_direct,
            extra_map=None,
            tx_to_idx=tx_to_idx,
            wallet_to_idx=wallet_to_idx,
            tx_global_offset=tx_global_offset,
            wallet_global_offset=wallet_global_offset,
        )
        _report_gt_stats(gt, "strict", verbose)
        return gt

    # ── 8. Extended mode: compute LFPN_k(T) via AddrAddr BFS.
    if verbose:
        print(f"[LFPN] Building AddrAddr adjacency "
              f"({len(addr_addr):,} edges) ...")
    aa_adj = _build_aa_adjacency(addr_addr)

    # Cache: for each wallet w encountered in any W_direct, compute its
    # k-hop labeled-illicit neighbours once.
    seed_wallets: Set[str] = set()
    for addrs in w_direct.values():
        seed_wallets.update(addrs)
    if verbose:
        print(f"[LFPN] Seed wallets for BFS: {len(seed_wallets):,}")

    if verbose:
        print(f"[LFPN] Running per-wallet BFS (k={k_hops}) ...")
    illicit_neighbour_cache: Dict[str, Set[str]] = {}
    for i, w in enumerate(seed_wallets, start=1):
        illicit_neighbour_cache[w] = _bfs_labeled_illicit(
            seed=w,
            adj=aa_adj,
            illicit_wallets=illicit_wallets,
            k_hops=k_hops,
        )
        if verbose and i % 5000 == 0:
            print(f"[LFPN]   BFS progress: {i:,}/{len(seed_wallets):,}")

    # Assemble LFPN per tx.
    lfpn_map: Dict = {}
    for tid, direct_addrs in w_direct.items():
        union: Set[str] = set()
        for w in direct_addrs:
            union |= illicit_neighbour_cache.get(w, set())
        # Remove direct initiators from LFPN to avoid double-counting —
        # they're already in W_direct and will be unioned in _finalize_gt.
        union -= direct_addrs
        if union:
            lfpn_map[tid] = union

    if verbose:
        total_lfpn_wallets = sum(len(s) for s in lfpn_map.values())
        print(f"[LFPN] Tx with ≥1 LFPN wallet: {len(lfpn_map):,}  "
              f"(total LFPN wallet slots: {total_lfpn_wallets:,})")

    gt = _finalize_gt(
        w_direct_map=w_direct,
        extra_map=lfpn_map,
        tx_to_idx=tx_to_idx,
        wallet_to_idx=wallet_to_idx,
        tx_global_offset=tx_global_offset,
        wallet_global_offset=wallet_global_offset,
    )
    _report_gt_stats(gt, f"extended (k={k_hops})", verbose)
    return gt


# ═════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═════════════════════════════════════════════════════════════════════════════

def _rebuild_wallet_to_idx(
    wallets:            pd.DataFrame,
    wallets_cls:        pd.DataFrame,
    txs_cls:            pd.DataFrame,
    tx_to_idx:          Dict,
    addr_tx:            pd.DataFrame,
    tx_addr:            pd.DataFrame,
    addr_addr:          pd.DataFrame,
    include_addr_addr:  bool,
    fraud_subgraph:     bool,
    fraud_subgraph_hops:int,
    verbose:            bool,
) -> Dict[str, int]:
    """
    Replicates utils.elliptic_plus_loader's wallet-filtering logic
    verbatim.

    IMPORTANT: this must stay in sync with
    elliptic_plus_loader.load_elliptic_plus_dataset — if the loader's
    filtering rules change, the global IDs returned by this GT builder
    will silently drift.
    """
    if fraud_subgraph:
        connected = _fraud_subgraph_wallets(
            txs_cls=txs_cls,
            tx_to_idx=tx_to_idx,
            addr_tx=addr_tx,
            tx_addr=tx_addr,
            addr_addr=addr_addr,
            hops=fraud_subgraph_hops,
            verbose=verbose,
        )
    else:
        connected = (
            set(addr_tx["input_address"].dropna().astype(str))
            | set(tx_addr["output_address"].dropna().astype(str))
        )
        if include_addr_addr:
            connected |= (
                set(addr_addr["input_address"].dropna().astype(str))
                | set(addr_addr["output_address"].dropna().astype(str))
            )

    wallets_filt = wallets[wallets["address"].astype(str).isin(connected)] \
                      .reset_index(drop=True)
    return {str(addr): i for i, addr in enumerate(wallets_filt["address"])}


def _fraud_subgraph_wallets(
    txs_cls:   pd.DataFrame,
    tx_to_idx: Dict,
    addr_tx:   pd.DataFrame,
    tx_addr:   pd.DataFrame,
    addr_addr: pd.DataFrame,
    hops:      int,
    verbose:   bool,
) -> Set[str]:
    """
    Mirrors utils.elliptic_plus_loader._fraud_subgraph_wallets.
    Duplicated here to avoid importing a private symbol cross-module.
    """
    cls_map = txs_cls.set_index("txId")["class"].to_dict()
    labeled_tx = {
        tid for tid, cls in cls_map.items()
        if cls in (CLASS_ILLICIT, CLASS_LICIT) and tid in tx_to_idx
    }

    at = addr_tx.dropna(subset=["input_address", "txId"])
    seed_wallets: Set[str] = set(
        at.loc[at["txId"].isin(labeled_tx), "input_address"].astype(str)
    )
    ta = tx_addr.dropna(subset=["txId", "output_address"])
    seed_wallets |= set(
        ta.loc[ta["txId"].isin(labeled_tx), "output_address"].astype(str)
    )

    if hops <= 1 or addr_addr is None or addr_addr.empty:
        if verbose:
            print(f"[LFPN]   fraud_subgraph hop-1 wallets: {len(seed_wallets):,}")
        return seed_wallets

    aa = addr_addr.dropna(subset=["input_address", "output_address"])
    adj: Dict[str, Set[str]] = {}
    for src, dst in zip(aa["input_address"].astype(str),
                        aa["output_address"].astype(str)):
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)

    frontier = set(seed_wallets)
    visited  = set(seed_wallets)
    for hop in range(2, hops + 1):
        next_frontier: Set[str] = set()
        for w in frontier:
            for nb in adj.get(w, ()):
                if nb not in visited:
                    next_frontier.add(nb)
        visited |= next_frontier
        frontier = next_frontier
        if verbose:
            print(f"[LFPN]   fraud_subgraph hop-{hop}: +{len(next_frontier):,}  "
                  f"(total {len(visited):,})")
        if not frontier:
            break
    return visited


def _build_aa_adjacency(addr_addr: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build an undirected adjacency from the AddrAddr edge list.

    We treat the wallet-wallet graph as undirected for BFS distance —
    LFPN is about proximity in the wallet-linkage graph, not about
    money-flow direction (that lives on the tx-wallet bipartite side).
    """
    aa = addr_addr.dropna(subset=["input_address", "output_address"])
    adj: Dict[str, Set[str]] = {}
    for src, dst in zip(aa["input_address"].astype(str),
                        aa["output_address"].astype(str)):
        if src == dst:
            continue
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    return adj


def _bfs_labeled_illicit(
    seed:            str,
    adj:             Dict[str, Set[str]],
    illicit_wallets: Set[str],
    k_hops:          int,
) -> Set[str]:
    """
    BFS outward from `seed` up to `k_hops`, return the set of labeled
    illicit wallets reached (excluding the seed itself).
    """
    if seed not in adj:
        return set()

    visited: Set[str] = {seed}
    frontier: List[str] = [seed]
    illicit_hits: Set[str] = set()

    for _ in range(k_hops):
        next_frontier: List[str] = []
        for node in frontier:
            for nb in adj.get(node, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                next_frontier.append(nb)
                if nb in illicit_wallets:
                    illicit_hits.add(nb)
        if not next_frontier:
            break
        frontier = next_frontier

    return illicit_hits


def _finalize_gt(
    w_direct_map:        Dict,
    extra_map:           Optional[Dict],
    tx_to_idx:           Dict,
    wallet_to_idx:       Dict[str, int],
    tx_global_offset:    int,
    wallet_global_offset:int,
) -> Dict[int, Set[int]]:
    """
    Convert {tx_id: set of wallet addresses} to the global-ID schema that
    build_typed_causal_graph_from_hetero uses.  Wallets not present in
    the loader's wallet subset are silently dropped; tx whose resulting
    GT set is empty are dropped too.
    """
    gt: Dict[int, Set[int]] = {}

    for tid, direct_addrs in w_direct_map.items():
        tx_local = tx_to_idx.get(tid)
        if tx_local is None:
            continue
        tx_global = tx_global_offset + tx_local

        gt_addrs: Set[str] = set(direct_addrs)
        if extra_map is not None:
            gt_addrs |= extra_map.get(tid, set())

        gt_globals: Set[int] = {
            wallet_global_offset + wallet_to_idx[a]
            for a in gt_addrs
            if a in wallet_to_idx
        }

        if gt_globals:
            gt[tx_global] = gt_globals

    return gt


def _report_gt_stats(
    gt:      Dict[int, Set[int]],
    label:   str,
    verbose: bool,
) -> None:
    if not verbose:
        return
    if not gt:
        print(f"[LFPN] GT ({label}): 0 tx covered — nothing to evaluate.")
        return
    sizes = [len(s) for s in gt.values()]
    n = len(sizes)
    print(f"[LFPN] GT ({label}): {n:,} illicit tx covered.")
    print(f"[LFPN]   |GT|   min={min(sizes)}  "
          f"median={sorted(sizes)[n // 2]}  "
          f"mean={sum(sizes) / n:.2f}  max={max(sizes)}")