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

from model.tracer_strategies.base import abs_ce, reconstruct_chain


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

    def is_dead_leaf(n) -> bool:
        # causal trail goes cold here — same stop condition as greedy / dag_dp
        return all(
            abs_ce(causal_effects, p, n) < threshold
            for p in graph.get_upstream_neighbors(n)
        )

    def is_terminal(n) -> bool:
        if n == target or not is_dead_leaf(n):
            return False
        return (not prefer) or graph.node_type.get(n) in prefer

    pred: Dict[object, object] = {target: None}
    visited = {target}
    queue = deque([(target, 0)])
    deepest = target  # BFS pops in non-decreasing depth, so the last is deepest

    while queue:
        u, depth = queue.popleft()
        if is_terminal(u):
            return u, reconstruct_chain(pred, target, u)
        if u != target:
            deepest = u
        if depth >= max_hops:
            continue
        for p in graph.get_upstream_neighbors(u):
            if p not in visited:
                visited.add(p)
                pred[p] = u
                queue.append((p, depth + 1))

    # no preferred-type dead-leaf within the hop budget → deepest reached node
    if deepest == target:
        return target, [target]
    return deepest, reconstruct_chain(pred, target, deepest)
