"""
elliptic_identity.py — decode causal-graph global IDs back to the REAL
Elliptic++ identities (transaction txId / wallet Bitcoin address).

The causal graph's global node IDs are internal indices; this module rebuilds
the loader's exact node ordering so any traced chain can be shown with its real
entities. Shared by evaluate.py (--dump_chains) and scripts/export_crime_chains.py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def build_reverse_maps(
    data_root: str,
    include_addr_addr: bool = True,
    fraud_subgraph: bool = False,
    fraud_subgraph_hops: int = 2,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Return ({tx_local_idx -> txId}, {wallet_local_idx -> address}) matching the
    loader's node ordering exactly.

    Transaction order == row order in txs_features.csv. Wallet order is
    reproduced with lfpn_utils._rebuild_wallet_to_idx under the SAME loader
    flags, then inverted — the same helper evaluate.py uses for illicit wallets.
    """
    from utils.lfpn_utils import _rebuild_wallet_to_idx

    root = Path(os.path.join(data_root, "Elliptic++"))

    txid_list = pd.read_csv(root / "txs_features.csv", usecols=["txId"])["txId"].tolist()
    idx_to_txid = {i: str(t) for i, t in enumerate(txid_list)}
    tx_to_idx = {t: i for i, t in enumerate(txid_list)}

    wallets = pd.read_csv(root / "wallets_features.csv", usecols=[0])
    wallets.columns = ["address"]
    wallets_cls = pd.read_csv(root / "wallets_classes.csv")
    wallets_cls.columns = [c.strip() for c in wallets_cls.columns]
    txs_cls = pd.read_csv(root / "txs_classes.csv")
    txs_cls.columns = [c.strip() for c in txs_cls.columns]
    addr_tx = pd.read_csv(root / "AddrTx_edgelist.csv")
    addr_tx.columns = [c.strip() for c in addr_tx.columns]
    tx_addr = pd.read_csv(root / "TxAddr_edgelist.csv")
    tx_addr.columns = [c.strip() for c in tx_addr.columns]
    addr_addr = pd.read_csv(root / "AddrAddr_edgelist.csv")
    addr_addr.columns = [c.strip() for c in addr_addr.columns]

    wallet_to_idx = _rebuild_wallet_to_idx(
        wallets=wallets, wallets_cls=wallets_cls, txs_cls=txs_cls,
        tx_to_idx=tx_to_idx, addr_tx=addr_tx, tx_addr=tx_addr, addr_addr=addr_addr,
        include_addr_addr=include_addr_addr, fraud_subgraph=fraud_subgraph,
        fraud_subgraph_hops=fraud_subgraph_hops, verbose=False,
    )
    idx_to_addr = {idx: addr for addr, idx in wallet_to_idx.items()}
    return idx_to_txid, idx_to_addr


def decode_node(
    gid: int,
    type_offsets: Dict[str, int],
    causal_graph,
    data,
    idx_to_txid: Dict[int, str],
    idx_to_addr: Dict[int, str],
) -> dict:
    """Decode one global id into a real-identity record for the viewer."""
    ntype = causal_graph.node_type.get(gid, "unknown")
    local = gid - type_offsets.get(ntype, 0)
    if ntype == "transaction":
        real_id = idx_to_txid.get(local, f"tx#{local}")
    elif ntype == "wallet":
        real_id = idx_to_addr.get(local, f"wallet#{local}")
    else:
        real_id = f"{ntype}#{local}"
    y = getattr(data[ntype], "y", None)
    fraud = bool(y is not None and 0 <= local < y.size(0) and int(y[local]) == 1)
    t = causal_graph.timestamps.get(gid)
    return {
        "global": int(gid),
        "type": ntype,
        "real_id": str(real_id),
        "time": (int(t) if t is not None else None),
        "fraud": fraud,
    }


def chain_to_record(
    chain: list,
    causal_effects: Dict[Tuple, float],
    is_true_positive: bool,
    type_offsets: Dict[str, int],
    causal_graph,
    data,
    idx_to_txid: Dict[int, str],
    idx_to_addr: Dict[int, str],
) -> dict:
    """
    Build one viewer chain record from a traced chain ([target, ..., root]).

    Each non-target node carries `ce` = the causal effect of that (more
    upstream) node on the previous, more-downstream one — the edge the tracer
    followed.
    """
    nodes = []
    for pos, gid in enumerate(chain):
        rec = decode_node(gid, type_offsets, causal_graph, data,
                          idx_to_txid, idx_to_addr)
        rec["pos"] = pos
        if pos == 0:
            rec["is_target"] = True
        else:
            rec["ce"] = float(causal_effects.get((gid, chain[pos - 1]), 0.0))
        nodes.append(rec)
    return {
        "target_txid": nodes[0]["real_id"],
        "depth": len(chain) - 1,
        "root_type": nodes[-1]["type"],
        "root_real_id": nodes[-1]["real_id"],
        "root_is_fraud": nodes[-1]["fraud"],
        "is_true_positive": bool(is_true_positive),
        "nodes": nodes,
    }
