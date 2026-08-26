#!/usr/bin/env python3
"""Export the 1-hop neighbourhood of every crime-chain node from the FULL
Elliptic++ graph (not just the chain union), for the explainability viewer.

The frontend's "顯示一階鄰居" feature otherwise only sees neighbours that happen
to be shared across the 2000 traced chains. To show a chain's *real* surrounding
context we must consult the original edge lists. Those files are large
(AddrAddr ≈ 200 MB), so we stream each once and keep only edges that touch a
chain node, capping neighbours per node to keep the output small.

Edge tables (data/Elliptic++/):
  AddrTx_edgelist.csv   input_address(wallet) -> txId(transaction)
  TxAddr_edgelist.csv   txId(transaction)     -> output_address(wallet)
  txs_edgelist.csv      txId1(transaction)    -> txId2(transaction)
  AddrAddr_edgelist.csv input_address(wallet) -> output_address(wallet)

Output (results/chain_neighbors.json):
  {
    "meta": {"cap": 30, "n_chain_nodes": ..., "generated_from": "..."},
    "neighbors": {
      "<chain_node_real_id>": {
        "degree": <full neighbour count in the graph>,
        "nodes": [{"real_id": "...", "type": "transaction|wallet"}, ...]  # <= cap
      }, ...
    }
  }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

TRANSACTION = "transaction"
WALLET = "wallet"


def load_chain_nodes(chains_csv: str) -> set[str]:
    """Distinct real_ids appearing on any chain (column chain_real_ids)."""
    nodes: set[str] = set()
    with open(chains_csv, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        try:
            col = header.index("chain_real_ids")
        except ValueError:
            sys.exit("crime_chains.csv 缺少 chain_real_ids 欄位")
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= col:
                continue
            for rid in parts[col].split("|"):
                rid = rid.strip()
                if rid:
                    nodes.add(rid)
    return nodes


def stream_edges(path: str, type_a: str, type_b: str, chain_nodes: set[str],
                 degree: dict[str, int], seen: dict[str, set[str]],
                 kept: dict[str, list[dict[str, str]]], cap: int) -> None:
    """One streaming pass: for each edge (a,b), if an endpoint is a chain node,
    record the other endpoint as its neighbour (deduped, capped)."""
    if not os.path.exists(path):
        print(f"  (略過,找不到 {path})")
        return
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        fh.readline()  # header
        for line in fh:
            i = line.find(",")
            if i < 0:
                continue
            a = line[:i].strip()
            b = line[i + 1:].rstrip("\n").strip()
            if not a or not b:
                continue
            # a is a chain node -> b is its neighbour
            if a in chain_nodes and b != a:
                degree[a] += 1
                s = seen[a]
                if b not in s:
                    s.add(b)
                    if len(kept[a]) < cap:
                        kept[a].append({"real_id": b, "type": type_b})
            # b is a chain node -> a is its neighbour
            if b in chain_nodes and a != b:
                degree[b] += 1
                s = seen[b]
                if a not in s:
                    s.add(a)
                    if len(kept[b]) < cap:
                        kept[b].append({"real_id": a, "type": type_a})
            n += 1
    print(f"  掃描 {os.path.basename(path)}: {n:,} 邊")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", default="results/crime_chains.csv")
    ap.add_argument("--data_root", default="data/Elliptic++")
    ap.add_argument("--out", default="results/chain_neighbors.json")
    ap.add_argument("--cap", type=int, default=30,
                    help="每個鏈節點最多保留的鄰居數(完整度數仍會記在 degree)")
    args = ap.parse_args()

    print(f"讀取鏈節點: {args.chains}")
    chain_nodes = load_chain_nodes(args.chains)
    print(f"  相異鏈節點 real_id 數: {len(chain_nodes):,}")

    degree: dict[str, int] = defaultdict(int)
    seen: dict[str, set[str]] = defaultdict(set)
    kept: dict[str, list[dict[str, str]]] = defaultdict(list)

    files = [
        ("AddrTx_edgelist.csv", WALLET, TRANSACTION),
        ("TxAddr_edgelist.csv", TRANSACTION, WALLET),
        ("txs_edgelist.csv", TRANSACTION, TRANSACTION),
        ("AddrAddr_edgelist.csv", WALLET, WALLET),
    ]
    print("掃描邊表(完整 Elliptic++ 圖):")
    for name, ta, tb in files:
        stream_edges(os.path.join(args.data_root, name), ta, tb,
                     chain_nodes, degree, seen, kept, args.cap)

    neighbors = {
        rid: {"degree": degree[rid], "nodes": kept[rid]}
        for rid in chain_nodes
        if degree.get(rid, 0) > 0
    }

    out = {
        "meta": {
            "cap": args.cap,
            "n_chain_nodes": len(chain_nodes),
            "n_with_neighbors": len(neighbors),
            "generated_from": os.path.basename(args.chains),
        },
        "neighbors": neighbors,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)

    total_deg = sum(degree.values())
    print(f"\n寫出 {args.out}")
    print(f"  有鄰居的鏈節點: {len(neighbors):,} / {len(chain_nodes):,}")
    print(f"  總度數(完整圖): {total_deg:,}")
    print(f"  檔案大小: {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
