"""
export_crime_chains.py — trace fraud transactions back through the temporal
causal DAG and dump each chain with its REAL Elliptic++ identities (txId /
Bitcoin address) for the crime-chain viewer (viz/crime_chain.html).

Every node the tracer walks through is a real dataset entity — the causal
graph's global IDs map 1-to-1 back to a transaction's txId or a wallet's
address. This script makes that mapping explicit so you can see the actual
"follow-the-money" path, e.g.:

    tx 3321  ◀──(CE −0.42)── wallet 1A1zP…  ◀──── tx 88017  ◀──── wallet 3J98t…

By default it traces only TRUE-POSITIVE fraud (predicted fraud AND actually
illicit), so the chains are genuine fraud trails — not classifier mistakes.

Usage (run where the checkpoint + data live, e.g. the server):
    cd CI-RCT
    python scripts/export_crime_chains.py \
        --data_root data --include_addr_addr true \
        --checkpoint checkpoints/ci_rct_elliptic++_best.pt \
        --num_hgt_layers 3 --n_chains 30 --max_hops 20 \
        --prefer_root_types wallet --prefer_reachable_depth 3 \
        --out viz/crime_chains.json --serve
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from configs.config import CI_RCT_Config  # noqa: E402
from model.ci_rct import CI_RCT  # noqa: E402
from model.root_cause_tracer import RootCauseTracer  # noqa: E402
from utils.data_utils import (  # noqa: E402
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
    default_blocked_edge_types,
    default_rare_edge_types,
)
from utils.elliptic_plus_loader import load_elliptic_plus_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export real-identity fraud crime chains")
    p.add_argument("--data_root", default="data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="viz/crime_chains.json")
    p.add_argument("--n_chains", type=int, default=30,
                   help="How many fraud transactions to trace.")
    p.add_argument("--only_true_positive", type=lambda x: x.lower() == "true",
                   default=True,
                   help="Trace only predicted-fraud AND actually-illicit tx.")
    # graph / tracer
    p.add_argument("--num_seeds", type=int, default=40)
    p.add_argument("--hop_limit", type=int, default=3)
    p.add_argument("--node_limit", type=int, default=200_000)
    p.add_argument("--max_hops", type=int, default=20)
    p.add_argument("--ce_threshold", type=float, default=0.0001)
    p.add_argument("--prefer_root_types", default="wallet")
    p.add_argument("--prefer_reachable_depth", type=int, default=3)
    # loader flags (must match training)
    p.add_argument("--include_addr_addr", type=lambda x: x.lower() == "true",
                   default=True)
    p.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true",
                   default=False)
    p.add_argument("--fraud_subgraph_hops", type=int, default=2)
    # model
    p.add_argument("--num_hgt_layers", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--type_emb_dim", type=int, default=16)
    p.add_argument("--device", default="cpu")
    # one-command viewing
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no_open", action="store_true")
    return p.parse_args()


def build_reverse_maps(args):
    """
    Build {tx_local_idx -> txId} and {wallet_local_idx -> address} that match
    the loader's node ordering exactly (so global IDs decode to real entities).

    Wallet ordering is reproduced with lfpn_utils._rebuild_wallet_to_idx — the
    same helper evaluate.py uses — under the SAME loader flags, then inverted.
    """
    from pathlib import Path
    from utils.lfpn_utils import _rebuild_wallet_to_idx

    root = Path(os.path.join(args.data_root, "Elliptic++"))

    # transaction: file order == local index
    txid_list = pd.read_csv(root / "txs_features.csv", usecols=["txId"])["txId"].tolist()
    idx_to_txid = {i: str(t) for i, t in enumerate(txid_list)}
    tx_to_idx = {t: i for i, t in enumerate(txid_list)}

    # wallet: rebuild the loader's address->idx, then invert
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
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
        verbose=False,
    )
    idx_to_addr = {idx: addr for addr, idx in wallet_to_idx.items()}
    return idx_to_txid, idx_to_addr


def load_model(args, data):
    """Build CI_RCT, restore arch from a v2 checkpoint, warm up, load weights."""
    arch = CI_RCT.read_arch_metadata(args.checkpoint, device=args.device)

    def g(key, cli):
        return arch[key] if (arch and arch.get(key) is not None) else cli

    config = CI_RCT_Config(
        dataset="elliptic++",
        target_node_type="transaction",
        max_hops=args.max_hops,
        ce_threshold=args.ce_threshold,
        hidden_dim=g("hidden_dim", args.hidden_dim),
        num_hgt_layers=g("num_hgt_layers", args.num_hgt_layers),
        num_heads=g("num_heads", args.num_heads),
        node_type_emb_dim=g("node_type_emb_dim", args.type_emb_dim),
    )
    in_channels = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types) if data[nt].x is not None
    }
    model = CI_RCT(
        config=config, metadata=data.metadata(),
        in_channels_dict=in_channels, use_gan=False,
    ).to(args.device)
    model.eval()
    with torch.no_grad():
        model.forward(data)              # materialise lazy HGT weights
    model.load_checkpoint(args.checkpoint, device=args.device)
    return model


def _node_record(gid, type_offsets, tcg, data, idx_to_txid, idx_to_addr):
    """Decode one global id into a real-identity record."""
    ntype = tcg.node_type.get(gid, "unknown")
    local = gid - type_offsets.get(ntype, 0)
    if ntype == "transaction":
        real_id = idx_to_txid.get(local, f"tx#{local}")
    elif ntype == "wallet":
        real_id = idx_to_addr.get(local, f"wallet#{local}")
    else:
        real_id = f"{ntype}#{local}"
    y = getattr(data[ntype], "y", None)
    fraud = bool(y is not None and 0 <= local < y.size(0) and int(y[local]) == 1)
    t = tcg.timestamps.get(gid)
    return {
        "global": int(gid),
        "type": ntype,
        "real_id": str(real_id),
        "time": (int(t) if t is not None else None),
        "fraud": fraud,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device(args.device)

    print("Loading Elliptic++ …")
    data, target_type = load_elliptic_plus_dataset(
        os.path.join(args.data_root, "Elliptic++"),
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )
    data = data.to(device)
    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    labels = data[target_type].y
    test_mask = data[target_type].test_mask
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()

    print("Building reverse identity maps (txId / address) …")
    idx_to_txid, idx_to_addr = build_reverse_maps(args)

    # Causal graph seeded from illicit test tx (same recipe as evaluate.py).
    fraud_global_ids = [offset + i for i in test_indices if labels[i].item() == 1]
    seed_ids = fraud_global_ids[: args.num_seeds]
    blocked = default_blocked_edge_types("elliptic++")
    rare = default_rare_edge_types("elliptic++")
    print("Building temporal causal graph …")
    tcg = build_typed_causal_graph_from_hetero(
        data, seed_node_ids=seed_ids, hop_limit=args.hop_limit,
        node_limit=args.node_limit,
        blocked_edge_types=blocked or None, rare_edge_types=rare or None,
    )

    print(f"Loading model from {args.checkpoint} …")
    model = load_model(args, data)
    with torch.no_grad():
        logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)
    causal_effects = model.compute_causal_effects(flat_h, causal_graph=tcg)
    preds = logits.argmax(dim=-1)

    prefer = {t.strip() for t in args.prefer_root_types.split(",") if t.strip()} or None
    tracer = RootCauseTracer(
        causal_graph=tcg, max_hops=args.max_hops, threshold=args.ce_threshold,
        prefer_root_types=prefer, prefer_reachable_depth=args.prefer_reachable_depth,
    )

    # Targets: predicted fraud (+ optionally truly illicit) and in the graph.
    targets = [
        offset + idx for idx in test_indices
        if preds[idx].item() == 1
        and (offset + idx) in tcg.set_v
        and (not args.only_true_positive or labels[idx].item() == 1)
    ][: args.n_chains]

    print(f"Tracing {len(targets)} fraud transactions …")
    chains = []
    for gid in targets:
        root, chain = tracer.trace_root_cause(gid, causal_effects)
        nodes = []
        for pos, node_gid in enumerate(chain):
            rec = _node_record(node_gid, type_offsets, tcg, data,
                               idx_to_txid, idx_to_addr)
            rec["pos"] = pos
            if pos == 0:
                rec["is_target"] = True
            else:
                # CE of this (more-upstream) node on the previous one.
                rec["ce"] = float(causal_effects.get((node_gid, chain[pos - 1]), 0.0))
            nodes.append(rec)
        chains.append({
            "target_txid": nodes[0]["real_id"],
            "depth": len(chain) - 1,
            "root_type": nodes[-1]["type"],
            "root_real_id": nodes[-1]["real_id"],
            "root_is_fraud": nodes[-1]["fraud"],
            "nodes": nodes,
        })

    # Sort: deepest, fraud-root chains first (most interesting to look at).
    chains.sort(key=lambda c: (c["root_is_fraud"], c["depth"]), reverse=True)

    meta = {
        "dataset": "elliptic++",
        "checkpoint": os.path.basename(args.checkpoint),
        "n_chains": len(chains),
        "only_true_positive": args.only_true_positive,
        "n_fraud_root": sum(1 for c in chains if c["root_is_fraud"]),
        "mean_depth": (round(float(np.mean([c["depth"] for c in chains])), 2)
                       if chains else 0.0),
        "prefer_root_types": sorted(prefer) if prefer else [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "chains": chains}, f)

    print(f"\nWrote {args.out}")
    print(f"  {len(chains)} chains, mean depth {meta['mean_depth']}, "
          f"{meta['n_fraud_root']} end at a labelled-fraud wallet/tx")
    if args.serve:
        _serve(args.out, args.port, not args.no_open)
    else:
        print("\nView: python -m http.server 8000 --directory viz "
              "→ http://localhost:8000/crime_chain.html")


def _serve(out_path, port, auto_open):
    import functools
    import http.server
    import webbrowser

    serve_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=serve_dir)
    httpd, bound = None, port
    for p in range(port, port + 10):
        try:
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", p), handler)
            bound = p
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"No free port in {port}..{port + 9}.")
    url = f"http://localhost:{bound}/crime_chain.html"
    print(f"\nServing {serve_dir}  →  {url}\n  (Ctrl+C 停止)")
    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止 server。")
        httpd.shutdown()


if __name__ == "__main__":
    main()
