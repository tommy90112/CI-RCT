"""
export_crime_chains.py — standalone tracer that dumps fraud crime chains with
their REAL Elliptic++ identities (txId / Bitcoin address) for the viewer.

This is a convenience wrapper: it loads the checkpoint, builds the temporal
causal graph, traces a handful of fraud transactions, and decodes each chain to
real entities. For the EXACT set of chains behind evaluate.py's depth histogram
(num_traced), prefer `evaluate.py --dump_chains viz/crime_chains.json` instead —
that reuses the identical trace.

Every node a chain walks through is a real dataset entity: the causal graph's
global IDs map 1-to-1 to a transaction's txId or a wallet's address.

Usage (run where the checkpoint + data live):
    cd CI-RCT
    python scripts/export_crime_chains.py \
        --data_root data --include_addr_addr true \
        --checkpoint checkpoints/ci_rct_elliptic++_best.pt \
        --num_hgt_layers 3 --n_chains 50 --max_hops 20 \
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
from utils.elliptic_identity import build_reverse_maps, chain_to_record  # noqa: E402
from utils.elliptic_plus_loader import load_elliptic_plus_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export real-identity fraud crime chains")
    p.add_argument("--data_root", default="data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="viz/crime_chains.json")
    p.add_argument("--n_chains", type=int, default=50)
    p.add_argument("--only_true_positive", type=lambda x: x.lower() == "true",
                   default=False,
                   help="If true, trace only predicted-fraud AND actually-illicit "
                        "tx. Default false → all predicted-fraud (like evaluate).")
    p.add_argument("--num_seeds", type=int, default=40)
    p.add_argument("--hop_limit", type=int, default=2)
    p.add_argument("--node_limit", type=int, default=1_000_000)
    p.add_argument("--max_hops", type=int, default=20)
    p.add_argument("--ce_threshold", type=float, default=0.0001)
    p.add_argument("--prefer_root_types", default="wallet")
    p.add_argument("--prefer_reachable_depth", type=int, default=3)
    p.add_argument("--include_addr_addr", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--fraud_subgraph_hops", type=int, default=2)
    p.add_argument("--wallet_per_address", type=lambda x: x.lower() == "true",
                   default=False,
                   help="Collapse wallet nodes to one-per-address and keep "
                        "'Time step' as a feature (→56-dim wallet, SAGE-FIN "
                        "style). Must match how the checkpoint was trained — "
                        "joint/per-address checkpoints need true.")
    p.add_argument("--num_hgt_layers", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--type_emb_dim", type=int, default=16)
    p.add_argument("--ncm_baseline", type=str, default="zero",
                   choices=["zero", "type_mean", "marginal"],
                   help="CE null-intervention baseline: 'zero' (legacy), "
                        "'type_mean' (recentres CE so sign is interpretable) or "
                        "'marginal' (E[MLP(h)]; no Jensen gap, E[CE]=0 per type). "
                        "No retraining needed — recomputes CE from same weights.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no_open", action="store_true")
    return p.parse_args()


def load_model(args, data):
    """Build CI_RCT, restore arch from a v2 checkpoint, warm up, load weights."""
    arch = CI_RCT.read_arch_metadata(args.checkpoint, device=args.device)

    def g(key, cli):
        return arch[key] if (arch and arch.get(key) is not None) else cli

    config = CI_RCT_Config(
        dataset="elliptic++", target_node_type="transaction",
        max_hops=args.max_hops, ce_threshold=args.ce_threshold,
        hidden_dim=g("hidden_dim", args.hidden_dim),
        num_hgt_layers=g("num_hgt_layers", args.num_hgt_layers),
        num_heads=g("num_heads", args.num_heads),
        node_type_emb_dim=g("node_type_emb_dim", args.type_emb_dim),
        ncm_baseline=args.ncm_baseline,
    )
    in_channels = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types) if data[nt].x is not None
    }
    model = CI_RCT(config=config, metadata=data.metadata(),
                   in_channels_dict=in_channels, use_gan=False).to(args.device)
    model.eval()
    with torch.no_grad():
        model.forward(data)
    model.load_checkpoint(args.checkpoint, device=args.device)
    return model


def serve_dir(out_path, port, auto_open):
    import functools
    import http.server
    import webbrowser
    d = os.path.dirname(os.path.abspath(out_path)) or "."
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
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
    url = f"http://localhost:{bound}/crime_chain_3d.html"
    print(f"\nServing {d}  →  {url}\n  (Ctrl+C 停止)")
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
        wallet_per_address=args.wallet_per_address,
    )
    data = data.to(device)
    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    labels = data[target_type].y
    test_indices = data[target_type].test_mask.nonzero(as_tuple=True)[0].tolist()

    print("Building reverse identity maps (txId / address) …")
    idx_to_txid, idx_to_addr = build_reverse_maps(
        args.data_root, include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph, fraud_subgraph_hops=args.fraud_subgraph_hops,
        wallet_per_address=args.wallet_per_address,
    )

    seed_ids = [offset + i for i in test_indices if labels[i].item() == 1][: args.num_seeds]
    print("Building temporal causal graph …")
    tcg = build_typed_causal_graph_from_hetero(
        data, seed_node_ids=seed_ids, hop_limit=args.hop_limit,
        node_limit=args.node_limit,
        blocked_edge_types=default_blocked_edge_types("elliptic++") or None,
        rare_edge_types=default_rare_edge_types("elliptic++") or None,
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

    targets = [
        offset + idx for idx in test_indices
        if preds[idx].item() == 1 and (offset + idx) in tcg.set_v
        and (not args.only_true_positive or labels[idx].item() == 1)
    ][: args.n_chains]

    print(f"Tracing {len(targets)} fraud transactions …")
    chains = []
    for gid in targets:
        _, chain = tracer.trace_root_cause(gid, causal_effects)
        chains.append(chain_to_record(
            chain, causal_effects, labels[gid - offset].item() == 1,
            type_offsets, tcg, data, idx_to_txid, idx_to_addr,
        ))
    chains.sort(key=lambda c: (c["is_true_positive"], c["root_is_fraud"], c["depth"]),
                reverse=True)

    meta = {
        "dataset": "elliptic++",
        "checkpoint": os.path.basename(args.checkpoint),
        "n_chains": len(chains),
        "n_true_positive": sum(1 for c in chains if c["is_true_positive"]),
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
          f"{meta['n_fraud_root']} end at a labelled-fraud node")
    if args.serve:
        serve_dir(args.out, args.port, not args.no_open)
    else:
        print("\nView: python -m http.server 8000 --directory viz "
              "→ http://localhost:8000/crime_chain_3d.html")


if __name__ == "__main__":
    main()
