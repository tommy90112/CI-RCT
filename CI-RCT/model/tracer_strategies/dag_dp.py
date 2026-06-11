"""
DAG-DP / Viterbi tracer strategy — the recommended main method.

On a time-respecting DAG the heaviest / most-probable backward causal path is
found EXACTLY in O(V + E) over the sink's ancestor cone by a topological-order
dynamic program.  Because the graph is acyclic a backward path can never revisit
a node, so no per-path ``visited`` set is needed; we bound the path to
``max_hops`` edges with a Bellman-Ford-style layered relaxation (each round
relaxes one more backward hop from a snapshot of the previous round, so after
``i`` rounds ``best[n]`` is the optimal cost reaching ``n`` with ≤ i edges).

The terminal node is constrained to a root-capable type via ``select_terminal``,
which is what removes the need for the ad-hoc per-hop reachability lookahead
(see ``base.select_terminal``).
"""
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
    best: Dict[object, float] = {target: 0.0}
    pred: Dict[object, object] = {target: None}

    for _ in range(max_hops):
        new_best = dict(best)
        new_pred = dict(pred)
        updated = False
        for u, cost_u in best.items():
            for p in graph.get_upstream_neighbors(u):
                ce = abs_ce(causal_effects, p, u)
                if ce < threshold:
                    continue
                new_cost = cost_u + edge_cost(ce, objective, ce_eps)
                if p not in new_best or new_cost < new_best[p] - 1e-12:
                    new_best[p] = new_cost
                    new_pred[p] = u
                    updated = True
        best, pred = new_best, new_pred
        if not updated:
            break

    root = select_terminal(best, target, graph, causal_effects, threshold, prefer_root_types)
    if root is None:
        return target, [target]
    return root, reconstruct_chain(pred, target, root)
