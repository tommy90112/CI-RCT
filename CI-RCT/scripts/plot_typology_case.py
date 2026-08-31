"""
plot_typology_case.py — 論文 4.3 節風格的單鏈追溯圖（typology 案例用）。

沿用 plot_fraud_chain_svg.py 的視覺編碼（方形=交易、圓形=錢包、
紅/綠/藍=illicit/licit/unknown、target 粗黑框、蛇形主鏈、CE 邊標註、
真實邊方向取自原始邊表），但改為參數化 target、僅畫主鏈，並可在交易節點
下方加註金額（剝離鏈案例需要）。不修改任何既有程式碼。

用法
    cd CI-RCT
    python scripts/plot_typology_case.py \
        --target 210646674 --out figures/fig_case5_peeling_210646674.svg
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

RED, GREEN, BLUE = "#e21313", "#178a17", "#1f4ee0"
CLS_COLOR = {1: RED, 2: GREEN, 3: BLUE}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thesis-style typology case figure")
    p.add_argument("--target", required=True)
    p.add_argument("--chains", default="viz/crime_chains.json")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out", required=True, help="輸出路徑（.svg；另存同名 .png）")
    p.add_argument("--amount_col", default="out_BTC_total")
    p.add_argument("--show_amounts", type=lambda x: x.lower() == "true",
                   default=True)
    p.add_argument("--title", default=None)
    return p.parse_args()


def load_classes(root: str) -> tuple:
    tx_cls = {r[0]: int(r[1]) for r in
              csv.reader(open(os.path.join(root, "txs_classes.csv")))
              if r[0] != "txId"}
    w_cls = {r[0]: int(r[1]) for r in
             csv.reader(open(os.path.join(root, "wallets_classes.csv")))
             if r[0] != "address"}
    return tx_cls, w_cls


def resolve_directions(root: str, G: nx.Graph) -> None:
    """以原始邊表還原每條主鏈邊的真實方向（wallet→tx / tx→wallet / tx→tx）。"""
    relevant = set(G.nodes())
    directed = set()
    for fname in ("AddrTx_edgelist.csv", "TxAddr_edgelist.csv",
                  "txs_edgelist.csv", "AddrAddr_edgelist.csv"):
        with open(os.path.join(root, fname)) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) >= 2 and row[0] in relevant and row[1] in relevant:
                    directed.add((row[0], row[1]))
    for u, v in G.edges():
        if (u, v) in directed:
            G[u][v]["dir"] = (u, v)
        elif (v, u) in directed:
            G[u][v]["dir"] = (v, u)
        else:  # fallback：因果上游（pos 較大）→ 下游
            pu, pv = G.nodes[u]["pos_idx"], G.nodes[v]["pos_idx"]
            G[u][v]["dir"] = (u, v) if pu > pv else (v, u)


def main() -> None:
    args = parse_args()
    root = os.path.join(args.data_root, "Elliptic++")
    chains = json.load(open(args.chains))["chains"]
    try:
        chain = next(c for c in chains if c["target_txid"] == args.target)
    except StopIteration:
        sys.exit(f"[plot] target {args.target} 不在 {args.chains}")
    tx_cls, w_cls = load_classes(root)
    amounts = {}
    if args.show_amounts:
        t = pd.read_csv(os.path.join(root, "txs_features.csv"),
                        usecols=["txId", args.amount_col])
        t["txId"] = t["txId"].astype(str)
        amounts = t.set_index("txId")[args.amount_col].to_dict()

    G = nx.Graph()
    chain_ids = [n["real_id"] for n in chain["nodes"]]
    for n in chain["nodes"]:
        rid = n["real_id"]
        cls = (tx_cls if n["type"] == "transaction" else w_cls).get(rid, 3)
        G.add_node(rid, type=n["type"], cls=cls,
                   is_target=bool(n.get("is_target", False)),
                   pos_idx=n["pos"], ce=n.get("ce"))
    for a, b in zip(chain_ids, chain_ids[1:]):
        G.add_edge(a, b, main=True)
    resolve_directions(root, G)

    # 蛇形主鏈佈局（root 在左、target 在右 → 以 pos 反轉排列）
    display = list(reversed(chain_ids))
    pos = {}
    for i, rid in enumerate(display):
        pos[rid] = (i * 2.6, 1.4 * math.sin(i * 0.9))

    n = len(display)
    fig, ax = plt.subplots(figsize=(max(13, n * 1.55), 8))
    ax.axis("off")

    nodelist = list(G)
    size_map = {m: (2550 if G.nodes[m]["is_target"] else 1500) for m in G}
    nx.draw_networkx_edges(
        G, pos, edgelist=[G[u][v]["dir"] for u, v in G.edges()],
        edge_color="#111111", width=2.6, arrows=True, arrowstyle="-|>",
        arrowsize=20, nodelist=nodelist,
        node_size=[size_map[m] for m in nodelist],
        min_source_margin=5, min_target_margin=5, ax=ax)

    def draw_group(nodes, shape):
        if not nodes:
            return
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_shape=shape,
            node_color=[CLS_COLOR[G.nodes[m]["cls"]] for m in nodes],
            node_size=[1500 * (1.7 if G.nodes[m]["is_target"] else 1.0)
                       for m in nodes],
            edgecolors=["black" if G.nodes[m]["is_target"] else "#333333"
                        for m in nodes],
            linewidths=[3.2 if G.nodes[m]["is_target"] else 0.6 for m in nodes],
            ax=ax)

    draw_group([m for m in G if G.nodes[m]["type"] == "transaction"], "s")
    draw_group([m for m in G if G.nodes[m]["type"] == "wallet"], "o")

    # 匿名序號標籤（與論文圖一致：transaction k / address k）
    labels, tk, ak = {}, 0, 0
    for m in display:
        if G.nodes[m]["type"] == "transaction":
            if G.nodes[m]["is_target"]:
                labels[m] = "Target\ntransaction"
            else:
                tk += 1
                labels[m] = f"transaction {tk}"
        elif G.nodes[m]["is_target"]:
            labels[m] = "Target\naddress"
        else:
            ak += 1
            labels[m] = f"address {ak}"
    label_pos = {m: (pos[m][0],
                     pos[m][1] + (0.66 if G.nodes[m]["is_target"] else 0.42))
                 for m in labels}
    nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=8,
                            font_weight="bold", font_color="#111111", ax=ax)
    print("[plot] 標籤對照：")
    for m in display:
        print(f"  {labels[m].replace(chr(10), ' ')} = {m}")

    # 金額標註（交易節點下方）
    if args.show_amounts:
        for m in display:
            if G.nodes[m]["type"] != "transaction":
                continue
            v = amounts.get(m)
            if v is None or pd.isna(v):
                continue
            ax.text(pos[m][0], pos[m][1] - 0.5, f"{v:.4g} BTC",
                    ha="center", va="top", fontsize=8, color="#5a3d00",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#d8c9a3", alpha=0.9))

    # CE 邊標註（CE 存於上游節點：CE(up → down)）
    edge_labels = {}
    for i in range(len(chain_ids) - 1):
        down, up = chain_ids[i], chain_ids[i + 1]
        if G.nodes[up]["ce"] is not None:
            edge_labels[(down, up)] = f"CE={G.nodes[up]['ce']:.2f}"
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8.5, font_color="#a80000",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d0d0d0",
                  alpha=0.9),
        rotate=False, ax=ax)

    tgt, rt = chain_ids[0], chain_ids[-1]
    tgt_kind = ("illicit wallet" if G.nodes[tgt]["type"] == "wallet"
                else "known illicit")
    ax.annotate(f"Detected fraud\n({tgt_kind} — tracing start)",
                xy=pos[tgt], xytext=(-4, -74), textcoords="offset points",
                ha="center", fontsize=9.5, fontweight="bold", color="#a80000",
                arrowprops=dict(arrowstyle="-|>", color="#a80000", lw=1.6))
    ax.annotate("Traced source\n(root cause)",
                xy=pos[rt], xytext=(0, -72), textcoords="offset points",
                ha="center", fontsize=9.5, fontweight="bold", color="#0a6b0a",
                arrowprops=dict(arrowstyle="-|>", color="#0a6b0a", lw=1.6))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=RED, edgecolor="#333333", label="Illicit"),
        Patch(facecolor=GREEN, edgecolor="#333333", label="Licit"),
        Patch(facecolor=BLUE, edgecolor="#333333", label="Unknown"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#cfcfcf",
               markeredgecolor="#333333", markersize=13, linewidth=0,
               label="Transaction"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#cfcfcf",
               markeredgecolor="#333333", markersize=13, linewidth=0,
               label="Address"),
    ], loc="upper left", bbox_to_anchor=(1.002, 1.0), fontsize=11,
        frameon=True, borderpad=0.8, labelspacing=0.7)

    title = args.title or (f"Peeling-chain case — causal fraud chain "
                           f"(depth {chain['depth']}, joint variant)")
    ax.set_title(title, fontsize=14, pad=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, format="svg", bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {args.out} + {png} | nodes {G.number_of_nodes()}")


if __name__ == "__main__":
    main()
