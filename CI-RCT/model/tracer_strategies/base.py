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
    Pick the chain's endpoint = a |CE| DEAD-LEAF (a node where the causal signal
    stops: no parent with |CE| >= threshold). This is the SAME stop condition the
    greedy tracer uses, so chains run DEEP to where the causal trail goes cold.

    ``prefer_root_types`` is then a SOFT tie-break AMONG those dead-leaves (prefer
    e.g. wallet-typed origins), mirroring how the greedy tracer uses it as a
    ranking nudge. It is NOT a hard "stop at the first node of this type" terminal
    — that earlier semantics let a global optimiser stop at the fraud tx's
    immediate input wallet (a NON-dead-leaf with a strong funding parent),
    collapsing chains to depth 1.

    ``best[n]`` is the minimal accumulated cost to reach ``n`` (lower ⇒ higher
    product / heavier sum). Order:
        1. dead-leaves of a preferred type, lowest cost
        2. any dead-leaf, lowest cost
        3. fallback (no parent's signal died within the hop budget): lowest-cost
           reached node
    Returns None when only ``target`` was reached (caller returns the degenerate
    ``(target, [target])``, mirroring greedy stopping at depth 0).
    """
    reached = [n for n in best if n != target]
    if not reached:
        return None

    def is_dead_leaf(n) -> bool:
        return all(
            abs_ce(causal_effects, p, n) < threshold
            for p in graph.get_upstream_neighbors(n)
        )

    leaves = [n for n in reached if is_dead_leaf(n)]
    pool = leaves or reached  # fallback: nothing died within budget → any reached

    prefer = prefer_root_types or set()
    if prefer:
        typed = [n for n in pool if graph.node_type.get(n) in prefer]
        if typed:
            return min(typed, key=lambda n: best[n])
    return min(pool, key=lambda n: best[n])
