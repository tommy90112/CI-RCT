"""
Data utilities for CI-RCT.

Provides:
  - compute_type_offsets():              global node ID scheme per type
  - build_typed_causal_graph_from_hetero(): BFS-based TypedCausalGraph construction
  - heterodata_to_flat_feature_dict():   flat {global_id: feature_tensor}
"""
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from model.typed_causal_graph import TypedCausalGraph


def compute_type_offsets(data: HeteroData) -> Dict[str, int]:
    """
    Compute global node ID offsets for each node type.

    Node types are sorted alphabetically for reproducibility.

    Example:
        types = ['actor', 'transaction']  (sorted)
        actor has 100 nodes   → offset = 0
        transaction has 200   → offset = 100

    Args:
        data: HeteroData graph

    Returns:
        dict: {node_type: global_start_offset}
    """
    offsets: Dict[str, int] = {}
    offset = 0
    for ntype in sorted(data.node_types):
        offsets[ntype] = offset
        offset += data[ntype].num_nodes
    return offsets


def build_typed_causal_graph_from_hetero(
    data: HeteroData,
    target_node_id: Optional[int] = None,
    seed_node_ids: Optional[List[int]] = None,
    hop_limit: int = 2,
    node_limit: int = 500,
    directed: bool = True,
) -> TypedCausalGraph:
    """
    Build a TypedCausalGraph from a HeteroData object.

    If target_node_id is given, performs BFS from that node up to hop_limit
    hops and includes only the resulting subgraph.  Otherwise includes all
    nodes up to node_limit.

    Global node IDs follow compute_type_offsets() scheme.

    Args:
        data:           PyG HeteroData graph
        target_node_id: Global node ID to centre the subgraph on (optional)
        hop_limit:      BFS depth for subgraph extraction
        node_limit:     Hard cap on number of nodes to include
        directed:       Whether to register edges as directed (for upstream BFS)

    Returns:
        TypedCausalGraph with typed nodes and edges
    """
    if not data.node_types:
        raise ValueError("HeteroData has no node types.")

    type_offsets = compute_type_offsets(data)

    # --- Build adjacency list in global ID space ---
    # adj[node] = list of (neighbour, edge_type_str)
    adj: Dict[int, List[Tuple[int, str]]] = {}

    for src_type, rel, dst_type in data.edge_types:
        edge_key = (src_type, rel, dst_type)
        if not hasattr(data[edge_key], "edge_index"):
            continue
        ei = data[edge_key].edge_index
        src_off = type_offsets[src_type]
        dst_off = type_offsets[dst_type]
        etype_str = f"{src_type}__to__{dst_type}"

        for i in range(ei.size(1)):
            src_g = src_off + int(ei[0, i].item())
            dst_g = dst_off + int(ei[1, i].item())
            adj.setdefault(src_g, []).append((dst_g, etype_str))
            if not directed:
                adj.setdefault(dst_g, []).append((src_g, etype_str))

    # --- Determine which nodes to include ---
    total_nodes = sum(data[nt].num_nodes for nt in data.node_types)

    if target_node_id is not None:
        included_nodes = _bfs_subgraph(target_node_id, adj, hop_limit, node_limit)
    elif seed_node_ids:
        # Multi-source BFS from seed nodes — guarantees connectivity
        included_nodes = _multi_source_bfs(seed_node_ids, adj, hop_limit, node_limit)
    else:
        included_nodes = _sample_nodes_proportional(
            data, type_offsets, node_limit
        )

    # --- Build node_type dict for included nodes ---
    node_type_dict: Dict[int, str] = {}
    for ntype, off in type_offsets.items():
        n = data[ntype].num_nodes
        for local_idx in range(n):
            gid = off + local_idx
            if gid in included_nodes:
                node_type_dict[gid] = ntype

    # --- Create TypedCausalGraph ---
    tcg = TypedCausalGraph(
        V=list(included_nodes),
        node_types=node_type_dict,
    )

    # --- Add edges ---
    for src_g, neighbours in adj.items():
        if src_g not in included_nodes:
            continue
        for dst_g, etype_str in neighbours:
            if dst_g not in included_nodes:
                continue
            tcg.add_edge(src_g, dst_g, etype_str)

    return tcg


def heterodata_to_flat_feature_dict(
    data: HeteroData,
    type_offsets: Dict[str, int],
) -> Dict[int, Tensor]:
    """
    Convert HeteroData node features to a flat {global_node_id: feature_tensor}.

    Args:
        data:         HeteroData graph
        type_offsets: Output of compute_type_offsets(data)

    Returns:
        dict: {global_id: Tensor [feature_dim]}
    """
    flat: Dict[int, Tensor] = {}
    for ntype, off in type_offsets.items():
        if data[ntype].x is None:
            continue
        x: Tensor = data[ntype].x
        for local_idx in range(x.size(0)):
            flat[off + local_idx] = x[local_idx]
    return flat


# ── Private helpers ────────────────────────────────────────────────────────────


def _multi_source_bfs(
    seeds: List[int],
    adj: Dict[int, List[Tuple[int, str]]],
    hop_limit: int,
    node_limit: int,
) -> set:
    """
    BFS from multiple seed nodes simultaneously.
    Each seed expands up to hop_limit hops; stops when node_limit reached.
    """
    visited = set(seeds)
    queue = deque([(s, 0) for s in seeds])

    while queue and len(visited) < node_limit:
        node, depth = queue.popleft()
        if depth >= hop_limit:
            continue
        for neighbour, _ in adj.get(node, []):
            if neighbour not in visited and len(visited) < node_limit:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))

    return visited


def _sample_nodes_proportional(
    data: HeteroData,
    type_offsets: Dict[str, int],
    node_limit: int,
) -> set:
    """
    Sample up to node_limit nodes spread proportionally across all node types.

    Each type gets floor(node_limit * type_size / total_nodes) nodes,
    taken from the start of its ID range.  Remainder slots are filled
    from the largest types first.
    """
    node_counts = {nt: data[nt].num_nodes for nt in sorted(data.node_types)}
    total = sum(node_counts.values())
    cap = min(node_limit, total)

    # Base allocation (floor)
    alloc = {nt: int(cap * n / total) for nt, n in node_counts.items()}

    # Distribute remainder slots to largest types first
    remainder = cap - sum(alloc.values())
    for nt in sorted(node_counts, key=lambda t: node_counts[t], reverse=True):
        if remainder <= 0:
            break
        extra = min(remainder, node_counts[nt] - alloc[nt])
        alloc[nt] += extra
        remainder -= extra

    included: set = set()
    for nt, take in alloc.items():
        off = type_offsets[nt]
        for i in range(take):
            included.add(off + i)
    return included


def _bfs_subgraph(
    start: int,
    adj: Dict[int, List[Tuple[int, str]]],
    hop_limit: int,
    node_limit: int,
) -> set:
    """BFS from start up to hop_limit hops; return node set (at most node_limit)."""
    visited = {start}
    queue = deque([(start, 0)])

    while queue and len(visited) < node_limit:
        node, depth = queue.popleft()
        if depth >= hop_limit:
            continue
        for neighbour, _ in adj.get(node, []):
            if neighbour not in visited and len(visited) < node_limit:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))

    return visited
