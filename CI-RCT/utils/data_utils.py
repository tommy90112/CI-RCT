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
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from model.typed_causal_graph import TypedCausalGraph


def default_rare_edge_types(dataset: str) -> Set[str]:
    """
    Default rare-edge-type set for each dataset — edge types that must be
    preserved through chained BFS expansion, or an abundant edge type would
    crowd them out of the sampled subgraph.

    Returns an empty set for datasets where the rare-edge pass is not needed
    (the BFS then runs identically to the plain behaviour). Elliptic++ needs
    no such pass: its edge-type counts are within one order of magnitude.
    """
    return set()


def default_blocked_edge_types(dataset: str) -> Set[str]:
    """
    Default blocked-edge-type set for each dataset — edges dropped from
    BOTH the BFS expansion and the final causal graph.

    Symmetric to `default_rare_edge_types`: where `rare` lists edges that
    MUST be preserved through chained expansion, `blocked` lists edges
    that MUST be skipped. Both are per-dataset because the structural
    concern is dataset-specific.

    Elliptic / Elliptic++:
        Wallet↔wallet and address↔address shortcuts let BFS waste its
        node budget on lateral wallet-to-wallet expansion without ever
        advancing to upstream tx. We block them so the tracer has to
        traverse tx ↔ addr alternating paths.

    Other datasets (MG24, UNSW-NB15, DBLP, …):
        No analogous shortcut concern; returns empty set so the filter
        no-ops.

    Replaces the legacy `_is_addr_to_addr_edge` heuristic (May 2026):
    the old version was a blacklist ("same-type and does not contain
    'tx'/'transaction'") which silently dropped MG24's
    `host_node__to__host_node` (DD-11 audit↔ip bridge) and
    `process_node__to__process_node` (process forks) too — meaning
    the bridge edges never appeared in the causal graph and NCM never
    trained on them. Allowlist via dataset key is unambiguous.
    """
    if dataset in ("elliptic", "elliptic++"):
        return {
            "wallet__to__wallet",
            "address__to__address",
        }
    return set()


def build_global_timestamps(
    data: HeteroData,
    type_offsets: Dict[str, int],
) -> Dict[int, int]:
    """
    Collect per-node timestamps into a single {global_id: timestamp} dict.

    Reads each node type's optional `.time` tensor (set e.g. by
    elliptic_plus_loader: transaction = its discrete step, wallet = earliest
    funding step). Entries with a negative sentinel (-1 = "no known time") are
    skipped so those nodes stay temporally unconstrained — the TypedCausalGraph
    guard only fires when BOTH endpoints carry a timestamp.

    Returns an empty dict when no node type exposes `.time` (the causal graph
    then behaves exactly as before — direction from edge orientation only).
    """
    timestamps: Dict[int, int] = {}
    for ntype, off in type_offsets.items():
        t = getattr(data[ntype], "time", None)
        if t is None:
            continue
        t_cpu = t.detach().cpu()
        for local_idx in range(t_cpu.size(0)):
            val = int(t_cpu[local_idx].item())
            if val < 0:
                continue
            timestamps[off + local_idx] = val
    return timestamps


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
    blocked_edge_types: Optional[Set[str]] = None,
    rare_edge_types: Optional[Set[str]] = None,
    rare_reserve: int = 500,
    rare_max_hops: int = 5,
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
        blocked_edge_types: Optional set of edge-type strings ("src__to__dst")
                            dropped from BOTH BFS expansion and the final
                            causal graph. See `default_blocked_edge_types()`
                            for per-dataset defaults — e.g. Elliptic blocks
                            wallet/address self-loops so the tracer must
                            traverse tx↔addr alternating paths.  None / empty
                            = no edges blocked.
        rare_edge_types:    Optional set of edge-type strings that the BFS
                            should guarantee inclusion of.  After the main
                            BFS fills `node_limit - rare_reserve` nodes, a
                            second pass walks ONLY these edge types from
                            the current visited set, up to `rare_max_hops`
                            chained levels.  Designed for sparse bridge
                            edges (e.g. DD-11 host_resolves_to_ip with
                            only ~70 edges) that would otherwise be
                            crowded out by high-degree edge types like
                            host_sources_flow (~1.2M).  No-op if None.
        rare_reserve:       Node-budget reserve for the rare-edge pass.
                            Total visited remains <= node_limit.
        rare_max_hops:      Maximum chain length followed during the
                            rare-edge pass (e.g. 5 lets us walk
                            bridge → audit_source → process → forks
                            and still have room).

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
            blocked_edge_types=blocked_edge_types,
            rare_edge_types=rare_edge_types,
            rare_reserve=rare_reserve,
            rare_max_hops=rare_max_hops,
        )
    elif seed_node_ids:
        # Multi-source BFS from seed nodes — guarantees connectivity
        included_nodes = _multi_source_bfs(
            seed_node_ids, bfs_adj, hop_limit, node_limit,
            blocked_edge_types=blocked_edge_types,
            rare_edge_types=rare_edge_types,
            rare_reserve=rare_reserve,
            rare_max_hops=rare_max_hops,
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
    # Temporal precedence: when node types expose `.time`, edges that run
    # strictly backward in time are rejected at add_edge, turning the causal
    # graph into a time-respecting DAG (follow-the-money). Empty dict → the
    # guard no-ops and direction comes from edge orientation only (legacy).
    timestamps = build_global_timestamps(data, type_offsets)
    tcg = TypedCausalGraph(
        V=list(included_nodes),
        node_types=node_type_dict,
        timestamps=timestamps if timestamps else None,
    )

    # --- Add directed causal edges (forward only) ---
    # Drop blocked edge types from the final graph too, not just from BFS,
    # otherwise the tracer can still drift via included-by-other-paths.
    n_blocked = 0
    n_temporal_reject = 0
    for src_g, neighbours in causal_adj.items():
        if src_g not in included_nodes:
            continue
        for dst_g, etype_str in neighbours:
            if dst_g not in included_nodes:
                continue
            if blocked_edge_types and etype_str in blocked_edge_types:
                n_blocked += 1
                continue
            # Both endpoints are in V, so a False return here means the
            # temporal guard rejected a strictly-backward-in-time edge.
            if not tcg.add_edge(src_g, dst_g, etype_str):
                n_temporal_reject += 1

    if blocked_edge_types and n_blocked > 0:
        # Diagnostic: prints once per call so user knows the filter fired.
        print(f"  [data_utils] Skipped {n_blocked:,} blocked edges "
              f"from causal graph (types={sorted(blocked_edge_types)}).")
    if timestamps:
        # Count timed nodes *within this subgraph* (not the global dict), so the
        # ratio is meaningful — e.g. "480/500 timed" rather than the full-graph
        # timestamp count over the subgraph size.
        n_timed_sub = sum(1 for n in included_nodes if n in timestamps)
        print(f"  [data_utils] Time-respecting DAG: "
              f"{n_timed_sub:,}/{len(included_nodes):,} subgraph nodes timed, "
              f"{n_temporal_reject:,} backward edges rejected.")

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
    blocked_edge_types: Optional[Set[str]] = None,
    rare_edge_types: Optional[Set[str]] = None,
    rare_reserve: int = 500,
    rare_max_hops: int = 5,
) -> set:
    """
    BFS from multiple seed nodes simultaneously.
    Each seed expands up to hop_limit hops; stops when node_limit reached.

    If `blocked_edge_types` is non-empty, edges of those types are skipped
    during the traversal (and also dropped when building the final causal
    graph in `build_typed_causal_graph_from_hetero`).

    If `rare_edge_types` is set, the main BFS is capped at
    `node_limit - rare_reserve` and the remaining budget is spent on a
    chained walk that follows only those rare edge types — see
    `_expand_rare_edge_chain`.
    """
    visited = set(seeds)
    queue = deque([(s, 0) for s in seeds])

    main_cap = (
        max(len(seeds), node_limit - rare_reserve)
        if rare_edge_types else node_limit
    )

    while queue and len(visited) < main_cap:
        node, depth = queue.popleft()
        if depth >= hop_limit:
            continue
        for neighbour, etype_str in adj.get(node, []):
            if blocked_edge_types and etype_str in blocked_edge_types:
                continue
            if neighbour not in visited and len(visited) < main_cap:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))

    if rare_edge_types:
        _expand_rare_edge_chain(
            visited, adj, rare_edge_types,
            max_total=node_limit, max_hops=rare_max_hops,
        )

    return visited


def _expand_rare_edge_chain(
    visited: set,
    adj: Dict[int, List[Tuple[int, str]]],
    rare_edge_types: Set[str],
    max_total: int,
    max_hops: int,
) -> None:
    """
    Mutate `visited` by walking only `rare_edge_types` edges from each
    currently-visited node, up to `max_hops` chained levels deep.

    Used as a second pass after the main BFS to guarantee that sparse
    edge types (e.g. DD-11 bridge with ~70 edges out of 1.27M total)
    contribute nodes to the final causal-graph subgraph instead of being
    crowded out by high-degree edge types in the main BFS.
    """
    queue = deque([(n, 0) for n in list(visited)])
    while queue and len(visited) < max_total:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour, etype_str in adj.get(node, []):
            if etype_str not in rare_edge_types:
                continue
            if neighbour not in visited and len(visited) < max_total:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))


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
    blocked_edge_types: Optional[Set[str]] = None,
    rare_edge_types: Optional[Set[str]] = None,
    rare_reserve: int = 500,
    rare_max_hops: int = 5,
) -> set:
    """BFS from start up to hop_limit hops; return node set (at most node_limit).

    `blocked_edge_types`/`rare_edge_types`/`rare_reserve`/`rare_max_hops`
    work identically to `_multi_source_bfs` — see that function for details.
    """
    visited = {start}
    queue = deque([(start, 0)])

    main_cap = (
        max(1, node_limit - rare_reserve) if rare_edge_types else node_limit
    )

    while queue and len(visited) < main_cap:
        node, depth = queue.popleft()
        if depth >= hop_limit:
            continue
        for neighbour, etype_str in adj.get(node, []):
            if blocked_edge_types and etype_str in blocked_edge_types:
                continue
            if neighbour not in visited and len(visited) < main_cap:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))

    if rare_edge_types:
        _expand_rare_edge_chain(
            visited, adj, rare_edge_types,
            max_total=node_limit, max_hops=rare_max_hops,
        )

    return visited