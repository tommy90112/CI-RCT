"""
Shared helpers for pluggable RootCauseTracer search strategies.

Every strategy is a pure function with the signature

    trace(target, causal_effects, *, graph, max_hops, threshold,
          prefer_root_types=None, ce_eps=1e-12, objective="product")
        -> (root_node, causal_chain)

where ``causal_chain`` is ordered ``[target, ..., root]`` (the same convention
as the legacy greedy ``RootCauseTracer.trace_root_cause``).  Strategies rank
edges by ``|CE|`` magnitude — they never mutate ``causal_effects`` or ``graph``.

The two objectives:
    * ``"product"`` — MAX-PRODUCT (most-probable causal chain, MPE).  Each edge
      contributes ``-log(max(|CE|, ce_eps))`` and we MINIMISE the sum, i.e. the
      |CE| in (0, 1] are read as per-edge causal-survival probabilities and the
      best chain maximises their product.  Clamping by ``ce_eps`` keeps a |CE|=0
      edge (already dropped by ``threshold``) from blowing ``-log`` up to +inf.
    * ``"sum"``     — MAX-SUM of |CE| along the chain (additive, matches the
      legacy ``score_chain`` / beam convention).  Each edge contributes ``-|CE|``
      so the same "minimise cost" machinery yields the maximum-sum path.

Note on edge direction: ``causal_effects`` is keyed ``(parent, child)`` and the
graph stores parent→child edges, so a backward hop from ``child`` to a parent
``p`` weighs ``|CE[(p, child)]|``.
"""
import math
from typing import Dict, List, Optional, Set, Tuple


def abs_ce(causal_effects: Dict[Tuple, float], parent, child) -> float:
    """|CE| of the parent→child edge (0.0 when absent)."""
    return abs(causal_effects.get((parent, child), 0.0))


def edge_cost(ce_magnitude: float, objective: str, ce_eps: float) -> float:
    """
    Per-edge cost to MINIMISE.  product ⇒ -log|CE| (≥ 0); sum ⇒ -|CE| (≤ 0).
    The caller must already have dropped sub-threshold edges.
    """
    if objective == "product":
        return -math.log(max(ce_magnitude, ce_eps))
    if objective == "sum":
        return -ce_magnitude
    raise ValueError(f"unknown objective '{objective}' (expected 'product' or 'sum')")


def reconstruct_chain(pred: Dict, target, terminal) -> List:
    """
    Walk predecessor pointers from ``terminal`` back to ``target`` and return the
    chain in ``[target, ..., terminal]`` order.  ``pred[n]`` is the node one hop
    *towards* the target (its child on the traced path).
    """
    rev = [terminal]
    node = terminal
    seen = {terminal}
    while node != target:
        node = pred[node]
        if node in seen:  # defensive: pred cycle (a true DAG can't produce one)
            break
        seen.add(node)
        rev.append(node)
    rev.reverse()  # [target, ..., terminal]
    return rev


def select_terminal(
    best: Dict,
    target,
    graph,
    causal_effects: Dict[Tuple, float],
    threshold: float,
    prefer_root_types: Optional[Set[str]],
) -> Optional[object]:
    """
    Pick the best-scoring valid terminal among the nodes reached by the search.

    ``best[n]`` is the minimal accumulated cost to reach ``n`` from ``target``
    (lower cost ⇒ higher product / higher sum).  Terminal validity, in priority
    order (this is what structurally subsumes the DD-18 reachability lookahead —
    the terminal is constrained over the WHOLE path, not myopically per hop):

        1. When ``prefer_root_types`` is set: a terminal MUST be a node of that
           root-capable type.  A dead-end bridge that is a true source but NOT a
           root type is therefore NOT a valid terminal — even if a sub-path to it
           has higher product — so the optimiser is forced onto the branch that
           actually reaches a root.  Among root-type terminals the lowest-cost
           (most-probable / heaviest) one wins.
        2. Fallback (no root-type node reachable — the dataset "unreachable root"
           ceiling): the lowest-cost dead-leaf, else the lowest-cost reached node.
        3. When ``prefer_root_types`` is None: any dead-leaf (a node with no
           above-threshold parent) is a valid terminal — forcing full-depth
           chains to their natural stop — else any reached node.

    Returns None when nothing but ``target`` was reached (caller returns the
    degenerate ``(target, [target])``, mirroring greedy stopping at depth 0).
    """
    reached = [n for n in best if n != target]
    if not reached:
        return None

    prefer = prefer_root_types or set()
    if prefer:
        typed = [n for n in reached if graph.node_type.get(n) in prefer]
        if typed:
            return min(typed, key=lambda n: best[n])
        # else: unreachable-root fallback handled below

    def is_dead_leaf(n) -> bool:
        return all(
            abs_ce(causal_effects, p, n) < threshold
            for p in graph.get_upstream_neighbors(n)
        )

    leaves = [n for n in reached if is_dead_leaf(n)]
    pool = leaves or reached
    return min(pool, key=lambda n: best[n])
