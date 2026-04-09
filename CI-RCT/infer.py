"""
CI-RCT Inference — Root Cause Tracing for a single fraud node.

Given a trained model and a target transaction node ID,
traces back the causal chain and outputs the fraud path.

Usage:
  # Trace by local node index (within target node type)
  python infer.py \\
    --dataset elliptic++ \\
    --checkpoint checkpoints/ci_rct_elliptic++_best.pt \\
    --target_node 12345 \\
    --top_k 3

  # Show top-3 causal paths with Shapley scores
  python infer.py \\
    --dataset elliptic++ \\
    --checkpoint checkpoints/ci_rct_elliptic++_best.pt \\
    --target_node 12345 \\
    --top_k 3 \\
    --device cuda
"""
import argparse
import os

import torch

from configs.config import CI_RCT_Config
from model.causal_shapley import (
    compute_asymmetric_causal_shapley,
    compute_shapley_edge_scores,
)
from model.ci_rct import CI_RCT
from model.root_cause_tracer import RootCauseTracer
from utils.data_utils import build_typed_causal_graph_from_hetero, compute_type_offsets


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI-RCT: Fraud Root Cause Tracer")
    parser.add_argument("--dataset", type=str, default="elliptic++",
                        choices=["dblp", "acm", "imdb", "elliptic", "elliptic++"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--target_node", type=int, required=True,
                        help="Local node index within the target node type")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of causal paths to enumerate")
    parser.add_argument("--max_hops", type=int, default=5)
    parser.add_argument("--ce_threshold", type=float, default=0.1)
    parser.add_argument("--node_limit", type=int, default=5000,
                        help="Max nodes in TypedCausalGraph BFS (default 5000)")
    parser.add_argument("--device", type=str, default="cpu")
    # Must match training data loading
    parser.add_argument("--fraud_subgraph", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Must match training: use fraud-anchored wallet subgraph")
    parser.add_argument("--fraud_subgraph_hops", type=int, default=2)
    # Must match training config
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_hgt_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--type_emb_dim", type=int, default=16)
    return parser.parse_args()


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(name: str, root: str, **kwargs):
    if name == "dblp":
        from torch_geometric.datasets import DBLP
        return DBLP(root=os.path.join(root, "dblp"))[0], "author"
    if name == "acm":
        from torch_geometric.datasets import ACM
        return ACM(root=os.path.join(root, "acm"))[0], "paper"
    if name == "imdb":
        from torch_geometric.datasets import IMDB
        return IMDB(root=os.path.join(root, "imdb"))[0], "movie"
    if name == "elliptic":
        from utils.elliptic_loader import load_elliptic_dataset
        return load_elliptic_dataset(root)
    if name == "elliptic++":
        from utils.elliptic_plus_loader import load_elliptic_plus_dataset
        return load_elliptic_plus_dataset(
            os.path.join(root, "Elliptic++"),
            include_addr_addr=False,
            fraud_subgraph=kwargs.get("fraud_subgraph", False),
            fraud_subgraph_hops=kwargs.get("fraud_subgraph_hops", 2),
        )
    raise ValueError(f"Unknown dataset: {name!r}")


# ── Node label helper ──────────────────────────────────────────────────────────

def _node_label(global_id: int, type_offsets: dict) -> str:
    """Convert global node ID back to 'type_localIdx' string for display."""
    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
    node_type = sorted_types[0][0]
    local_idx = global_id

    for i, (ntype, offset) in enumerate(sorted_types):
        next_offset = sorted_types[i + 1][1] if i + 1 < len(sorted_types) else float("inf")
        if offset <= global_id < next_offset:
            node_type = ntype
            local_idx = global_id - offset
            break

    return f"{node_type}[{local_idx}]"


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading dataset: {args.dataset}")
    data, target_type = load_dataset(
        args.dataset, args.data_root,
        fraud_subgraph=args.fraud_subgraph,
        fraud_subgraph_hops=args.fraud_subgraph_hops,
    )
    data = data.to(device)

    # Convert local index → global ID
    type_offsets = compute_type_offsets(data)
    offset = type_offsets[target_type]
    global_target = offset + args.target_node

    n_target_nodes = data[target_type].num_nodes
    if args.target_node < 0 or args.target_node >= n_target_nodes:
        raise ValueError(
            f"target_node {args.target_node} out of range "
            f"[0, {n_target_nodes - 1}] for type '{target_type}'"
        )

    # Build causal graph seeded from target node
    print("Building TypedCausalGraph...")
    causal_graph = build_typed_causal_graph_from_hetero(
        data,
        seed_node_ids=[global_target],
        node_limit=args.node_limit,
    )
    print(f"  Causal graph: {len(causal_graph.v)} nodes, "
          f"{len(causal_graph.edge_type_map)} directed edges")

    if global_target not in causal_graph.set_v:
        print(f"\n[Warning] Target node {_node_label(global_target, type_offsets)} "
              f"is not in the causal graph (isolated node or no edges within hop limit).")

    # Load model
    config = CI_RCT_Config(
        dataset=args.dataset,
        target_node_type=target_type,
        max_hops=args.max_hops,
        ce_threshold=args.ce_threshold,
        hidden_dim=args.hidden_dim,
        num_hgt_layers=args.num_hgt_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        node_type_emb_dim=args.type_emb_dim,
    )

    in_channels_dict = {
        nt: data[nt].x.size(-1)
        for nt in sorted(data.node_types)
        if data[nt].x is not None
    }

    model = CI_RCT(
        config=config,
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        use_gan=False,
    ).to(device)

    model.load_checkpoint(args.checkpoint, device=args.device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, h_dict = model.forward(data)
        flat_h = model._build_flat_h(h_dict)

    # Fraud probability of target node
    probs = torch.softmax(logits[args.target_node], dim=-1)
    fraud_prob = probs[1].item()
    pred_label = "ILLICIT" if logits[args.target_node].argmax().item() == 1 else "LICIT"

    print(f"\n{'═' * 60}")
    print(f"  Target node : {_node_label(global_target, type_offsets)}")
    print(f"  Prediction  : {pred_label}  (fraud prob = {fraud_prob:.4f})")
    print(f"{'═' * 60}")

    # Compute causal effects
    causal_effects = model.compute_causal_effects(flat_h, causal_graph)

    # Shapley values for direct parents
    phi = compute_asymmetric_causal_shapley(causal_effects, causal_graph, global_target)
    if phi:
        print("\n  Causal Shapley values (direct parents):")
        for node, score in sorted(phi.items(), key=lambda x: -x[1]):
            print(f"    {_node_label(node, type_offsets):30s}  φ = {score:.4f}")

    # Top-k causal paths (beam search)
    tracer = RootCauseTracer(
        causal_graph=causal_graph,
        max_hops=args.max_hops,
        threshold=args.ce_threshold,
    )

    top_paths = tracer.trace_top_k_paths(global_target, causal_effects, k=args.top_k)

    print(f"\n  Top-{args.top_k} causal paths (fraud chain):")
    print(f"{'─' * 60}")

    if not top_paths:
        print("  No causal paths found. "
              "Try lowering --ce_threshold or increasing --max_hops.")
    else:
        for rank, (root, chain, score) in enumerate(top_paths, 1):
            chain_str = " → ".join(_node_label(n, type_offsets) for n in chain)
            print(f"\n  Path #{rank}  (score = {score:.4f})")
            print(f"    {chain_str}")
            print(f"    Root cause: {_node_label(root, type_offsets)}")

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()
