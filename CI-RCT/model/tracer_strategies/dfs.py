"""
DFS single-deep tracer strategy — lower bound, demonstrates DFS ≡ greedy.

Recursively descends into the highest-|CE| above-threshold parent at each hop
without backtracking.  This is exactly the legacy greedy |CE|-max walk re-cast
as a depth-first descent, so its metrics should match the greedy baseline (minus
the prefer_root_types / lookahead tie-breaks) — the ablation uses it to show the
"DFS" label adds nothing on its own; only an EXHAUSTIVE, memoised descent
degenerates into DAG-DP.
"""
from typing import Dict, List, Optional, Set, Tuple

from model.tracer_strategies.base import abs_ce


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
    chain: List = [target]
    current = target
    visited = {target}

    for _ in range(max_hops):
        candidates = [
            (abs_ce(causal_effects, p, current), p)
            for p in graph.get_upstream_neighbors(current)
            if abs_ce(causal_effects, p, current) >= threshold and p not in visited
        ]
        if not candidates:
            break
        # deterministic: highest |CE|, break ties by a stable node ordering
        best_ce = max(c for c, _ in candidates)
        nxt = min(p for c, p in candidates if c == best_ce)
        chain = chain + [nxt]
        visited = visited | {nxt}
        current = nxt

    return chain[-1], chain
