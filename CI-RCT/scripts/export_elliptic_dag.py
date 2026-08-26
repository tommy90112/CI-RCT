"""
export_elliptic_dag.py — build the follow-the-money temporal DAG and view it.

Two ways to use it:

  A) one command — build JSON, start a local server, open the browser:
        cd CI-RCT
        python scripts/export_elliptic_dag.py --serve --view 3d \
            --data_root data --n_seeds 8 --node_limit 400 --include_addr_addr true

  B) export only (static JSON for a thesis figure / re-open later):
        python scripts/export_elliptic_dag.py \
            --data_root data --n_seeds 8 --node_limit 400 \
            --include_addr_addr true --out viz/dag_elliptic.json
        python -m http.server 8000 --directory viz   # then open the .html

Why a build step at all? The whole graph pipeline (read 1.3 GB CSVs → PyG
HeteroData → fraud-seeded BFS → TypedCausalGraph + temporal pruning) is Python
+ torch; the browser only draws. This script is the Python→JS hand-off, and it
exports the ACTUAL TypedCausalGraph the model uses (backward-in-time edges
already removed), so the picture matches training/eval exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from utils.data_utils import (  # noqa: E402
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
    default_blocked_edge_types,
    default_rare_edge_types,
)
from utils.elliptic_plus_loader import load_elliptic_plus_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/serve the Elliptic++ temporal DAG")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out", default="viz/dag_elliptic.json")
    p.add_argument("--n_seeds", type=int, default=8,
                   help="Fraud transactions used as multi-source BFS seeds.")
    p.add_argument("--hop_limit", type=int, default=4)
    p.add_argument("--node_limit", type=int, default=400,
                   help="Hard cap on subgraph size (keeps the viz readable).")
    p.add_argument("--include_addr_addr", type=lambda x: x.lower() == "true",
                   default=False)
    p.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true",
                   default=False)
    p.add_argument("--fraud_subgraph_hops", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # ── one-command viewing ──────────────────────────────────────────────────
    p.add_argument("--serve", action="store_true",
                   help="After export, start a local http server and open the "
                        "viewer in the browser (Ctrl+C to stop).")
    p.add_argument("--view", choices=("3d", "2d"), default="3d",
                   help="Which viewer to open with --serve (default 3d).")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no_open", action="store_true",
                   help="With --serve, do NOT auto-open the browser (just serve).")
    return p.parse_args()


def _illicit_checker(data, type_offsets):
    """Return fn(global_id) -> bool: True iff that node's label == 1."""
    def is_illicit(gid: int) -> bool:
        ntype, local = None, None
        for nt, off in type_offsets.items():
            if off <= gid < off + data[nt].num_nodes:
                ntype, local = nt, gid - off
                break
        if ntype is None:
            return False
        y = getattr(data[ntype], "y", None)
        if y is None:
            return False
        return 0 <= local < y.size(0) and int(y[local].item()) == 1
    return is_illicit


def build_dag_json(args) -> dict:
    """Load Elliptic++, build the fraud-seeded temporal causal graph, and return
    the viewer payload {meta, nodes, links}."""
    print("Loading Elliptic++ …")
    data, target_type = load_elliptic_plus_dataset(
        os.path.join(args.data_root, "Elliptic++"),
        include_addr_addr=args.include_addr_addr,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )

    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    labels = data[target_type].y

    fraud_local = (labels == 1).nonzero(as_tuple=True)[0].tolist()
    seed_ids = [offset + i for i in fraud_local[: args.n_seeds]]
    if not seed_ids:
        raise SystemExit("No fraud transactions found — cannot seed the DAG.")

    blocked = default_blocked_edge_types("elliptic++")
    rare = default_rare_edge_types("elliptic++")

    print(f"Building temporal causal graph (seeds={len(seed_ids)}, "
          f"node_limit={args.node_limit}) …")
    tcg = build_typed_causal_graph_from_hetero(
        data,
        seed_node_ids=seed_ids,
        hop_limit=args.hop_limit,
        node_limit=args.node_limit,
        blocked_edge_types=blocked if blocked else None,
        rare_edge_types=rare if rare else None,
    )

    is_illicit = _illicit_checker(data, type_offsets)
    seed_set = set(seed_ids)

    nodes, times = [], []
    for gid in tcg.v:
        ntype = tcg.node_type.get(gid, "unknown")
        local = gid - type_offsets.get(ntype, 0)
        t = tcg.timestamps.get(gid)
        if t is not None:
            times.append(int(t))
        nodes.append({
            "id": int(gid),
            "type": ntype,
            "time": (int(t) if t is not None else None),
            "label": f"{ntype}#{int(local)}",
            "fraud": bool(is_illicit(gid)),
            "is_seed": gid in seed_set,
        })

    links = [
        {"source": int(s), "target": int(d), "etype": et}
        for (s, d), et in tcg.edge_type_map.items()
    ]

    by_type = Counter(n["type"] for n in nodes)
    meta = {
        "dataset": "elliptic++",
        "target_type": target_type,
        "n_nodes": len(nodes),
        "n_links": len(links),
        "n_fraud": sum(1 for n in nodes if n["fraud"]),
        "n_seed": len(seed_ids),
        "n_timed": len(times),
        "time_min": (min(times) if times else None),
        "time_max": (max(times) if times else None),
        "node_types": dict(by_type),
    }
    return {"meta": meta, "nodes": nodes, "links": links}


def serve_and_open(out_path: str, port: int, view: str, auto_open: bool) -> None:
    """Serve the directory holding out_path and open the chosen viewer."""
    import functools
    import http.server
    import webbrowser

    serve_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    view_html = "dag_timeline_3d.html" if view == "3d" else "dag_timeline.html"
    if not os.path.exists(os.path.join(serve_dir, view_html)):
        print(f"  [warn] {view_html} not found in {serve_dir}; serving anyway.")

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=serve_dir
    )
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

    url = f"http://localhost:{bound}/{view_html}"
    print(f"\nServing {serve_dir}  →  {url}")
    print("  (Ctrl+C 停止 server)")
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
    torch.manual_seed(args.seed)

    payload = build_dag_json(args)
    meta = payload["meta"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"\nWrote {args.out}")
    print(f"  nodes {meta['n_nodes']} / links {meta['n_links']} "
          f"(fraud {meta['n_fraud']}, timed {meta['n_timed']})")
    print(f"  time range: {meta['time_min']}..{meta['time_max']}")
    for t, c in meta["node_types"].items():
        print(f"    {t:14s} {c}")

    if args.serve:
        serve_and_open(args.out, args.port, args.view, not args.no_open)
    else:
        print("\nView: python -m http.server 8000 --directory viz "
              "→ http://localhost:8000/dag_timeline_3d.html"
              "\n  (或下次直接加 --serve 一鍵開啟)")


if __name__ == "__main__":
    main()
