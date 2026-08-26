"""
Render one real fraud chain (joint variant) as a 2D node-link SVG,
styled like the reference explanation figures.

  - target tx 226669358, depth 7, follow-the-money tx<->wallet path
  - shape: transaction=square, wallet=circle
  - colour (3-way Elliptic++ class): illicit=red, licit=green, unknown=blue
  - target node: thick black outline
  - main chain: bold black edges; neighbours: thin grey edges radiating out
"""
import csv
import json
import math

import matplotlib.pyplot as plt
import networkx as nx

TARGET = "226669358"
CAP = 5  # max neighbours drawn per decorated chain node
DECORATE = {0, 1, 4, 6, 7}  # chain positions to hang neighbours off
BLANK = False       # True → nodes + legend only (no edges, no id labels)
CHAIN_ONLY = True   # True → drop neighbour nodes, show only the traced chain
SHOW_CE = True      # True → label each chain edge with its CE score + annotate

RED, GREEN, BLUE = "#e21313", "#178a17", "#1f4ee0"
CLS_COLOR = {1: RED, 2: GREEN, 3: BLUE}
CLS_NAME = {1: "Illicit", 2: "Licit", 3: "Unknown"}

# ── load 3-way ground-truth classes ────────────────────────────────────────
tx_cls = {r[0]: int(r[1]) for r in csv.reader(open("data/Elliptic++/txs_classes.csv")) if r[0] != "txId"}
w_cls = {r[0]: int(r[1]) for r in csv.reader(open("data/Elliptic++/wallets_classes.csv")) if r[0] != "address"}


def klass(real_id, ntype):
    return (tx_cls if ntype == "transaction" else w_cls).get(real_id, 3)


# ── load chain + neighbours ─────────────────────────────────────────────────
chains = json.load(open("viz/crime_chains.json"))["chains"]
chain = next(c for c in chains if c["target_txid"] == TARGET)
neighbours = json.load(open("results/chain_neighbors.json"))["neighbors"]

G = nx.Graph()
chain_ids = [n["real_id"] for n in chain["nodes"]]
chain_set = set(chain_ids)

# chain nodes (ce = CE(this node -> its downstream node toward the target))
for n in chain["nodes"]:
    rid = n["real_id"]
    G.add_node(rid, type=n["type"], cls=klass(rid, n["type"]),
               is_target=n.get("is_target", False), chain=True, pos_idx=n["pos"],
               ce=n.get("ce"))
for a, b in zip(chain_ids, chain_ids[1:]):
    G.add_edge(a, b, main=True)

# neighbour nodes (context, hung off selected chain nodes)
for n in chain["nodes"]:
    if CHAIN_ONLY or n["pos"] not in DECORATE:
        continue
    entry = neighbours.get(n["real_id"])
    if not entry:
        continue
    added = 0
    for nb in entry["nodes"]:
        rid = nb["real_id"]
        if rid in chain_set or G.has_node(rid):
            continue
        G.add_node(rid, type=nb["type"], cls=klass(rid, nb["type"]),
                   is_target=False, chain=False, pos_idx=None, parent=n["real_id"])
        G.add_edge(n["real_id"], rid, main=False)
        added += 1
        if added >= CAP:
            break

# ── resolve REAL edge direction from raw edgelists (all edges are directed) ─
relevant = set(G.nodes())


def scan_directed(path):
    """Return {(src, dst)} rows whose both endpoints are in the graph."""
    out = set()
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) >= 2 and row[0] in relevant and row[1] in relevant:
                out.add((row[0], row[1]))
    return out


directed = set()
directed |= scan_directed("data/Elliptic++/AddrTx_edgelist.csv")  # wallet -> tx
directed |= scan_directed("data/Elliptic++/TxAddr_edgelist.csv")  # tx -> wallet
directed |= scan_directed("data/Elliptic++/txs_edgelist.csv")     # tx -> tx
if any(G.nodes[u]["type"] == "wallet" and G.nodes[v]["type"] == "wallet"
       for u, v in G.edges()):
    directed |= scan_directed("data/Elliptic++/AddrAddr_edgelist.csv")  # wallet -> wallet

for u, v in G.edges():
    if (u, v) in directed:
        G[u][v]["dir"] = (u, v)
    elif (v, u) in directed:
        G[u][v]["dir"] = (v, u)
    elif G[u][v]["main"]:  # fallback: causal parent (higher pos) -> child
        pu, pv = G.nodes[u]["pos_idx"], G.nodes[v]["pos_idx"]
        G[u][v]["dir"] = (u, v) if pu > pv else (v, u)
    else:  # fallback: chain node -> neighbour
        cn = u if G.nodes[u]["chain"] else v
        G[u][v]["dir"] = (cn, v if cn == u else u)

# ── layout: fix chain along a gentle path, spring the neighbours around ─────
pos = {}
n_chain = len(chain_ids)
for i, rid in enumerate(chain_ids):
    x = i * 2.6
    y = 1.4 * math.sin(i * 0.9)  # gentle snake
    pos[rid] = (x, y)

pos = nx.spring_layout(G, pos=pos, fixed=chain_ids, seed=7, k=1.3, iterations=200)

# ── draw ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")

# per-node sizes so arrowheads shrink exactly to each node's boundary
size_map = {n: (2550 if G.nodes[n]["is_target"] else
                1500 if G.nodes[n]["chain"] else 825) for n in G}
nodelist = list(G)
node_sizes = [size_map[n] for n in nodelist]

# every edge is directed; neighbours thin grey, main chain bold black
if not BLANK:
    nb_dir = [G[u][v]["dir"] for u, v in G.edges() if not G[u][v]["main"]]
    main_dir = [G[u][v]["dir"] for u, v in G.edges() if G[u][v]["main"]]
    nx.draw_networkx_edges(G, pos, edgelist=nb_dir, edge_color="#8a8a8a", width=1.2,
                           arrows=True, arrowstyle="-|>", arrowsize=12,
                           nodelist=nodelist, node_size=node_sizes,
                           min_source_margin=5, min_target_margin=5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=main_dir, edge_color="#111111", width=2.6,
                           arrows=True, arrowstyle="-|>", arrowsize=20,
                           nodelist=nodelist, node_size=node_sizes,
                           min_source_margin=5, min_target_margin=5, ax=ax)


def draw_group(nodes, shape, base_size):
    if not nodes:
        return
    colors = [CLS_COLOR[G.nodes[n]["cls"]] for n in nodes]
    sizes = [base_size * (1.7 if G.nodes[n]["is_target"] else
                          1.0 if G.nodes[n]["chain"] else 0.55) for n in nodes]
    edgec = ["black" if G.nodes[n]["is_target"] else "#333333" for n in nodes]
    lw = [3.2 if G.nodes[n]["is_target"] else 0.6 for n in nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_shape=shape,
                           node_color=colors, node_size=sizes,
                           edgecolors=edgec, linewidths=lw, ax=ax)


tx_nodes = [n for n in G if G.nodes[n]["type"] == "transaction"]
w_nodes = [n for n in G if G.nodes[n]["type"] == "wallet"]
draw_group(tx_nodes, "s", 1500)
draw_group(w_nodes, "o", 1500)

# labels: sequential anonymised codes (transaction k / address k), no real ids
if not BLANK:
    labels = {}
    tk = ak = 0
    nb_nodes = [n for n in G if not G.nodes[n]["chain"]]
    for n in list(chain_ids) + nb_nodes:  # chain first (pos order), then neighbours
        if G.nodes[n]["type"] == "transaction":
            if G.nodes[n]["is_target"]:
                labels[n] = "Target\ntransaction"
            else:
                tk += 1
                labels[n] = f"transaction {tk}"
        else:
            ak += 1
            labels[n] = f"address {ak}"

    chain_lab = {n: labels[n] for n in G if G.nodes[n]["chain"]}
    nb_lab = {n: labels[n] for n in G if not G.nodes[n]["chain"]}

    # chain labels: above the node (target lifted higher to clear the square)
    chain_lp = {}
    for n in chain_lab:
        x, y = pos[n]
        chain_lp[n] = (x, y + (0.66 if G.nodes[n]["is_target"] else 0.42))
    nx.draw_networkx_labels(G, chain_lp, labels=chain_lab, font_size=8,
                            font_weight="bold", font_color="#111111", ax=ax)

    # neighbour labels: pushed radially OUTWARD past the node (off its edge)
    nb_lp = {}
    for n in nb_lab:
        x, y = pos[n]
        px, py = pos[G.nodes[n]["parent"]]
        dx, dy = x - px, y - py
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        nb_lp[n] = (x + dx / d * 0.66, y + dy / d * 0.66)
    nx.draw_networkx_labels(G, nb_lp, labels=nb_lab, font_size=7,
                            font_color="#444444", ax=ax)

# ── legend: colour = status, shape = type (5 entries) ──────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

handles = [
    Patch(facecolor=RED, edgecolor="#333333", label="Illicit"),
    Patch(facecolor=GREEN, edgecolor="#333333", label="Licit"),
    Patch(facecolor=BLUE, edgecolor="#333333", label="Unknown"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#cfcfcf",
           markeredgecolor="#333333", markersize=13, linewidth=0, label="Transaction"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#cfcfcf",
           markeredgecolor="#333333", markersize=13, linewidth=0, label="Address"),
]
ax.legend(handles=handles, loc="upper right", fontsize=11, frameon=True,
          borderpad=0.8, labelspacing=0.7)

# ── CE labels on each chain edge + target / root annotations ────────────────
if SHOW_CE:
    # each main edge (down, up): CE lives on the upstream node (CE: up -> down)
    edge_labels = {}
    for i in range(len(chain_ids) - 1):
        down, up = chain_ids[i], chain_ids[i + 1]
        ce = G.nodes[up]["ce"]
        if ce is not None:
            edge_labels[(down, up)] = f"CE={ce:.2f}"
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8.5, font_color="#a80000",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d0d0d0", alpha=0.9),
        rotate=False, ax=ax)

    tgt, root = chain_ids[0], chain_ids[-1]
    ax.annotate("Detected fraud\n(known illicit — tracing start)",
                xy=pos[tgt], xytext=(-4, -70), textcoords="offset points",
                ha="center", fontsize=9.5, fontweight="bold", color="#a80000",
                arrowprops=dict(arrowstyle="-|>", color="#a80000", lw=1.6))
    ax.annotate("Traced source\n(root cause)",
                xy=pos[root], xytext=(0, -68), textcoords="offset points",
                ha="center", fontsize=9.5, fontweight="bold", color="#0a6b0a",
                arrowprops=dict(arrowstyle="-|>", color="#0a6b0a", lw=1.6))

if not BLANK:
    ax.set_title(f"Causal fraud chain  (depth {chain['depth']}, joint variant)",
                 fontsize=14, pad=14)

plt.tight_layout()
out = ("viz/fraud_chain_226669358_blank.svg" if BLANK
       else "viz/fraud_chain_226669358_ce.svg" if SHOW_CE
       else "viz/fraud_chain_226669358.svg")
plt.savefig(out, format="svg", bbox_inches="tight")
print("wrote", out, "| nodes", G.number_of_nodes(), "edges", G.number_of_edges())
