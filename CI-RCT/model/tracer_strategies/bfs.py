"""
Unweighted BFS tracer strategy — weight-blind lower bound / reachability oracle.

Fewest-hops backward path to a root-capable terminal, IGNORING |CE| magnitude
entirely (edges are not threshold-filtered — this is the pure reachability
arm).  It exposes the structural ceiling: whatever a complete search can reach,
BFS reaches at minimum depth.  Because it discards the causal weights its RCP /
CCV are a lower bound, but it never dead-ends on a reachable root the way greedy
|CE|-max does.
"""
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from model.tracer_strategies.base import reconstruct_chain


def trace(
    target,
    causal_effects: Dict[Tuple, float],
    *,
    graph,
    max_hops: int,
    threshold: float,
    prefer_root_types: Optional[Set[str]] = None,
    ce_eps: float = 1e-12,
    objective: str = "product",
) -> Tuple[object, List]:
    prefer = prefer_root_types or set()

    def is_terminal(n) -> bool:
        if n == target:
            return False
        if prefer:
            return graph.node_type.get(n) in prefer
        return len(graph.get_upstream_neighbors(n)) == 0  # true source

    pred: Dict[object, object] = {target: None}
    visited = {target}
    queue = deque([(target, 0)])

    while queue:
        u, depth = queue.popleft()
        if is_terminal(u):
            return u, reconstruct_chain(pred, target, u)
        if depth >= max_hops:
            continue
        for p in graph.get_upstream_neighbors(u):
            if p not in visited:
                visited.add(p)
                pred[p] = u
                queue.append((p, depth + 1))

    return target, [target]
