"""
TypedCausalGraph — Module 2a of CI-RCT.

Standalone directed causal graph for heterogeneous graphs.
Implements Pearl SCM's directed DAG structure (pa / ch dicts) with:
  - Node type labels τ_v
  - Edge type labels τ_e
  - Timestamps for temporal precedence verification
  - Kahn's algorithm topological ordering (upstream → downstream)

Replaces the previous CXGNN-inheriting version.  No external causal
module dependency.

Reference: CI-RCT_Thesis_Plan.md § 5.3.1
"""
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple


class TypedCausalGraph:
    """
    Type-aware directed causal graph (Typed Directed Acyclic Graph).

    Corresponds to Pearl SCM's DAG structure, extended with:
      - Directed parent / child adjacency (pa[v], ch[v])
      - Node type mapping  node_type[v]  -> type_str
      - Edge type mapping  edge_type_map[(src, dst)] -> edge_type_str
      - Optional timestamp mapping  timestamps[v] -> float / int
        (used to enforce temporal precedence during edge insertion)

    Args:
        V:          Iterable of node IDs
        node_types: {node_id: type_str} for every node in V
        timestamps: Optional {node_id: timestamp}.  When provided, edges
                    violating temporal precedence (ts_src >= ts_dst) are
                    silently rejected.
    """

    def __init__(
        self,
        V: Iterable,
        node_types: Dict,
        timestamps: Optional[Dict] = None,
    ) -> None:
        self.v: List = list(V)
        self.set_v: Set = set(self.v)

        missing = self.set_v - set(node_types.keys())
        if missing:
            raise ValueError(f"node_types missing entries for nodes: {missing}")

        self.node_type: Dict = dict(node_types)
        self.timestamps: Dict = dict(timestamps) if timestamps else {}

        # Directed adjacency — core SCM structure
        self.pa: Dict[object, Set] = {v: set() for v in self.v}   # parent set
        self.ch: Dict[object, Set] = {v: set() for v in self.v}   # child set

        # Type-annotated adjacency
        # pa_typed[v][u] = (edge_type, u_node_type)  -- upstream info
        self.pa_typed: Dict = {v: {} for v in self.v}

        # Canonical edge type map  (src, dst) -> edge_type
        self.edge_type_map: Dict[Tuple, str] = {}

        # Cached topological order — computed on first topological_order()
        # call and invalidated when add_edge mutates the graph.
        self._topo_cache: Optional[List] = None
        self._topo_idx_cache: Optional[Dict] = None

    # ── Edge manipulation ──────────────────────────────────────────────────────

    def add_edge(self, src, dst, edge_type: str) -> bool:
        """
        Add directed edge src → dst with the given semantic edge type.

        Semantic: src is a direct causal parent (cause) of dst (effect).

        Temporal guard: if timestamps are provided and ts(src) >= ts(dst),
        the edge is rejected (violates Granger temporal precedence) and
        False is returned.

        Args:
            src:       Source node (cause)
            dst:       Destination node (effect)
            edge_type: Semantic edge type label

        Returns:
            True if the edge was added, False if rejected.
        """
        if src not in self.set_v or dst not in self.set_v:
            return False

        # Temporal precedence guard
        if self.timestamps:
            ts_src = self.timestamps.get(src)
            ts_dst = self.timestamps.get(dst)
            if ts_src is not None and ts_dst is not None and ts_src >= ts_dst:
                return False

        self.pa[dst].add(src)
        self.ch[src].add(dst)
        self.pa_typed[dst][src] = (edge_type, self.node_type.get(src, "unknown"))
        self.edge_type_map[(src, dst)] = edge_type
        # Mutating the graph invalidates the cached topological order.
        self._topo_cache = None
        self._topo_idx_cache = None
        return True

    # ── Parent / child accessors ───────────────────────────────────────────────

    def parents(self, node) -> Set:
        """Return the set of direct causal parents Pa(node)."""
        return self.pa.get(node, set())

    def children(self, node) -> Set:
        """Return the set of direct causal children Ch(node)."""
        return self.ch.get(node, set())

    def get_upstream_neighbors(self, node) -> List:
        """Return nodes that can causally influence `node` (i.e. Pa(node))."""
        return list(self.pa.get(node, set()))

    # ── Topological ordering ───────────────────────────────────────────────────

    def topological_order(self) -> List:
        """
        Kahn's algorithm topological sort — upstream (root) → downstream.

        Returns list of all nodes in topological order.  If the graph
        contains a cycle the remaining nodes are omitted from the result.

        The result is cached on first call and reused thereafter.  This
        is safe because the public API of TypedCausalGraph adds nodes /
        edges only via __init__ and add_edge, both of which invalidate
        the cache.  For evaluation pipelines that call
        compute_asymmetric_causal_shapley repeatedly (one call per fraud
        node × original/perturbed = 2 × max_explain), this turns the
        per-call cost from O(V + E) Kahn's algorithm into a dict lookup,
        which on a 1M-node, 1.5M-edge causal graph is the difference
        between a 60-minute eval and a 30-second eval.
        """
        if getattr(self, "_topo_cache", None) is not None:
            return self._topo_cache

        in_deg = {v: len(self.pa[v]) for v in self.v}
        queue = deque(v for v in self.v if in_deg[v] == 0)
        order: List = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.ch[node]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)

        self._topo_cache = order
        return order

    def topological_index(self) -> Dict:
        """
        Return a {node: position_in_topological_order} dict, also cached.

        compute_asymmetric_causal_shapley re-builds this dict from
        topological_order() on every call — the cache avoids both the
        Kahn's pass AND the dict reconstruction.
        """
        if getattr(self, "_topo_idx_cache", None) is not None:
            return self._topo_idx_cache
        self._topo_idx_cache = {v: i for i, v in enumerate(self.topological_order())}
        return self._topo_idx_cache

    # ── Type accessors ─────────────────────────────────────────────────────────

    def get_edge_type(self, src, dst) -> Optional[str]:
        """Return edge type for directed edge src → dst, or None."""
        return self.edge_type_map.get((src, dst))

    def get_all_edge_types(self) -> List[str]:
        """Sorted list of unique edge types in the graph."""
        return sorted(set(self.edge_type_map.values()))

    def get_all_node_types(self) -> List[str]:
        """Sorted list of unique node types in the graph."""
        return sorted(set(self.node_type.values()))

    def get_node_type_index(self) -> Dict[str, int]:
        """Return {type_str: int_index} for nn.Embedding construction."""
        return {t: i for i, t in enumerate(self.get_all_node_types())}

    # ── Utility ───────────────────────────────────────────────────────────────

    def source_nodes(self) -> List:
        """Return nodes with no parents (potential root causes)."""
        return [v for v in self.v if not self.pa[v]]

    def __len__(self) -> int:
        return len(self.v)

    def __repr__(self) -> str:
        n_edges = len(self.edge_type_map)
        return (
            f"TypedCausalGraph(nodes={len(self.v)}, edges={n_edges}, "
            f"node_types={self.get_all_node_types()}, "
            f"edge_types={self.get_all_edge_types()})"
        )