"""
Data utilities for CI-RCT.

Provides:
  - compute_type_offsets():              global node ID scheme per type
  - build_typed_causal_graph_from_hetero(): BFS-based TypedCausalGraph construction
  - heterodata_to_flat_feature_dict():   flat {global_id: feature_tensor}

Tracer-aligned BFS (Apr 2026)
─────────────────────────────
On bipartite-style fraud graphs (e.g. Elliptic++: tx ↔ wallet),
addr→addr edges (e.g. wallet→wallet AddrAddr links, ~2.87M on Elliptic++)
let BFS waste its node budget on lateral wallet-to-wallet expansion
without ever advancing to upstream tx, which means the RootCauseTracer
runs out of tx parents to follow and stops at depth 1.

We therefore enforce alternating tx ↔ wallet traversal at TWO points:

  1. BFS subgraph extraction      (skip addr→addr neighbours)
  2. TypedCausalGraph edge insert (drop addr→addr edges entirely)

Both are necessary: skipping only at BFS still leaves addr→addr edges
inside the final causal graph if both endpoints happen to be included
via other tx-mediated paths, which would let the tracer wander
laterally between wallets at evaluation time.

For dataset-agnostic use the filter is a no-op on graphs without
addr-like same-type edges (DBLP, ACM, IMDB, UNSW-NB15 — none of these
have wallet→wallet-style same-type edges except possibly tx→tx,
which we deliberately allow).
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
    block_addr_to_addr: bool = True,
) -> TypedCausalGraph:
    """
    Build a TypedCausalGraph from a HeteroData object.

    If target_node_id is given, performs BFS from that node up to hop_limit
    hops and includes only the resulting subgraph.  Otherwise includes all
    nodes up to node_limit.

    Global node IDs follow compute_type_offsets() scheme.

    Args:
        data:               PyG HeteroData graph
        target_node_id:     Global node ID to centre the subgraph on (optional)
        seed_node_ids:      Multi-source BFS seeds (optional, for evaluation)
        hop_limit:          BFS depth for subgraph extraction
        node_limit:         Hard cap on number of nodes to include
        directed:           Whether to register edges as directed (for upstream BFS)
        block_addr_to_addr: If True (default), drop addr→addr edges from BOTH
                            BFS expansion and the final causal graph.  This
                            forces the tracer to traverse tx↔addr alternating
                            paths instead of getting stuck in lateral wallet
                            chains.  Set False to mirror the legacy behaviour
                            (e.g. for ablation experiments).

    Returns:
        TypedCausalGraph with typed nodes and edges
    """
    if not data.node_types:
        raise ValueError("HeteroData has no node types.")

    type_offsets = compute_type_offsets(data)

    # --- Build adjacency lists in global ID space ---
    # causal_adj: directed forward edges only → used to add edges to TypedCausalGraph
    # bfs_adj:    bidirectional → used for BFS node discovery so upstream nodes
    #             (causal parents) are reachable from fraud seed nodes
    causal_adj: Dict[int, List[Tuple[int, str]]] = {}
    bfs_adj:    Dict[int, List[Tuple[int, str]]] = {}

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
            # Causal (directed) adjacency
            causal_adj.setdefault(src_g, []).append((dst_g, etype_str))
            # BFS (bidirectional) adjacency — allows finding upstream parents
            bfs_adj.setdefault(src_g, []).append((dst_g, etype_str))
            bfs_adj.setdefault(dst_g, []).append((src_g, etype_str))

    # --- Determine which nodes to include ---
    if target_node_id is not None:
        included_nodes = _bfs_subgraph(
            target_node_id, bfs_adj, hop_limit, node_limit,
            block_addr_to_addr=block_addr_to_addr,
        )
    elif seed_node_ids:
        # Multi-source BFS from seed nodes — guarantees connectivity
        included_nodes = _multi_source_bfs(
            seed_node_ids, bfs_adj, hop_limit, node_limit,
            block_addr_to_addr=block_addr_to_addr,
        )
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

    # --- Add directed causal edges (forward only) ---
    # Block addr→addr edges from the final graph too, not just from BFS,
    # otherwise the tracer can still drift between wallets that BFS
    # happened to include via tx-mediated paths.
    n_aa_skipped = 0
    for src_g, neighbours in causal_adj.items():
        if src_g not in included_nodes:
            continue
        for dst_g, etype_str in neighbours:
            if dst_g not in included_nodes:
                continue
            if block_addr_to_addr and _is_addr_to_addr_edge(etype_str):
                n_aa_skipped += 1
                continue
            tcg.add_edge(src_g, dst_g, etype_str)

    if block_addr_to_addr and n_aa_skipped > 0:
        # Diagnostic: prints once per call so user knows the filter fired.
        print(f"  [data_utils] Skipped {n_aa_skipped:,} addr→addr edges "
              f"from causal graph (block_addr_to_addr=True).")

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


def _is_addr_to_addr_edge(etype_str: str) -> bool:
    """Return True if `etype_str` is a same-type non-tx edge (e.g. wallet↔wallet).

    Enforces tx ↔ addr alternating traversal by blocking addr→addr shortcuts
    such as `wallet__to__wallet` or `address__to__address`.

    Same-type tx edges (`transaction__to__transaction`, `tx__to__tx`) are
    intentionally allowed: a tx can plausibly be the upstream cause of
    another tx via a `flows_to` relation, and the tracer needs that link
    to chain multiple tx hops.

    For graphs whose edge-type strings don't follow `src__to__dst` (e.g. a
    custom serialisation), the function falls back to False, which means
    the filter no-ops and we keep the legacy behaviour — safe default.
    """
    parts = etype_str.split("__to__")
    if len(parts) != 2:
        return False
    src_type, dst_type = parts
    if src_type != dst_type:
        return False
    # Allow same-type tx links (tx→tx flows_to is a legitimate hop).
    return "tx" not in src_type and "transaction" not in src_type


def _multi_source_bfs(
    seeds: List[int],
    adj: Dict[int, List[Tuple[int, str]]],
    hop_limit: int,
    node_limit: int,
    block_addr_to_addr: bool = True,
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
        for neighbour, etype_str in adj.get(node, []):
            if block_addr_to_addr and _is_addr_to_addr_edge(etype_str):
                continue  # skip addr→addr; force tx↔addr alternating path
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
    block_addr_to_addr: bool = True,
) -> set:
    """BFS from start up to hop_limit hops; return node set (at most node_limit)."""
    visited = {start}
    queue = deque([(start, 0)])

    while queue and len(visited) < node_limit:
        node, depth = queue.popleft()
        if depth >= hop_limit:
            continue
        for neighbour, etype_str in adj.get(node, []):
            if block_addr_to_addr and _is_addr_to_addr_edge(etype_str):
                continue  # skip addr→addr; force tx↔addr alternating path
            if neighbour not in visited and len(visited) < node_limit:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))

    return visited