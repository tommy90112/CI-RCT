"""
Dijkstra tracer strategy — strong-comparison arm.

Single-source shortest path on the backward cost graph (cost = -log|CE| for the
``product`` objective).  On the DAG this returns the SAME optimum as
``dag_dp`` but pays the priority-queue ``log V`` overhead — the ablation uses it
to confirm that "global optimum" is what drives any gain, and that DAG-DP is the
cheaper way to reach it.

Dijkstra requires non-negative edge costs, which holds only for the ``product``
objective ( -log|CE| ≥ 0 ).  The ``sum`` objective has negative costs ( -|CE| ),
which would break Dijkstra's optimality, so it delegates to ``dag_dp`` (same
optimum via layered relaxation).
"""
import heapq
import itertools
from typing import Dict, List, Optional, Set, Tuple

from model.tracer_strategies.base import (
    abs_ce,
    edge_cost,
    reconstruct_chain,
    select_terminal,
)


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
    if objective != "product":
        from model.tracer_strategies import dag_dp
        return dag_dp.trace(
            target, causal_effects, graph=graph, max_hops=max_hops,
            threshold=threshold, prefer_root_types=prefer_root_types,
            ce_eps=ce_eps, objective=objective,
        )

    best: Dict[object, float] = {target: 0.0}
    pred: Dict[object, object] = {target: None}
    settled = set()
    counter = itertools.count()  # deterministic FIFO tie-break, avoids comparing nodes
    heap: List[Tuple[float, int, object, int]] = [(0.0, next(counter), target, 0)]

    while heap:
        cost, _, u, hops = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        if hops >= max_hops:
            continue
        for p in graph.get_upstream_neighbors(u):
            ce = abs_ce(causal_effects, p, u)
            if ce < threshold:
                continue
            new_cost = cost + edge_cost(ce, "product", ce_eps)
            if p not in best or new_cost < best[p] - 1e-12:
                best[p] = new_cost
                pred[p] = u
                heapq.heappush(heap, (new_cost, next(counter), p, hops + 1))

    root = select_terminal(best, target, graph, causal_effects, threshold, prefer_root_types)
    if root is None:
        return target, [target]
    return root, reconstruct_chain(pred, target, root)
