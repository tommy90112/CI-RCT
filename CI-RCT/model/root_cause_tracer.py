"""
RootCauseTracer — Module 3 of CI-RCT.

Backward causal path tracing over CE scores on a TypedCausalGraph.

Starting from a fraud-predicted target node, the tracer follows the
upstream edge with the highest |CE| at each hop until a stopping
criterion is met.

Why magnitude (|CE|) and not signed CE?
───────────────────────────────────────
CE(u → v) = P(v is fraud | do(h_u = h_actual)) − P(v is fraud | do(h_u = 0))

A positive CE means "if u is observed, v's fraud probability rises"
(promoter parent).  A negative CE means "if u is observed, v's fraud
probability drops" (suppressor parent).  Both directions encode a
*causal influence*; only the sign differs.

For root-cause tracing the question is "which upstream node has the
strongest causal influence on the target", not "which one promotes
fraud most".  Using raw signed CE causes the tracer to discard
suppressor-style edges entirely, even when |CE| is large — this
leaves the tracer stuck at depth 0 whenever the dominant parents
happen to have learned a suppression direction (which empirically
occurs on Elliptic++ wallet→tx edges, mean CE ≈ −0.35).

Switching to |CE| is consistent with the framework's theoretical
foundation: Asymmetric Causal Shapley already takes |φ| when
computing edge scores (see causal_shapley.compute_shapley_edge_scores),
so the tracer here uses the same convention.

Stopping conditions (priority order):
    1. No upstream neighbours       → current node is the root
    2. |best CE| < threshold        → causal signal too weak
    3. Cycle detected (visited set) → prevent infinite loops
    4. max_hops depth reached       → hard depth limit

Reference: CI-RCT_Thesis_Plan.md § 5.4
"""
import heapq
from typing import Dict, List, Optional, Tuple

from model.typed_causal_graph import TypedCausalGraph


class RootCauseTracer:
    """
    Greedy backward BFS root-cause tracer using |CE| for ranking.

    Args:
        causal_graph: TypedCausalGraph
        max_hops:     Maximum number of hops to trace
        threshold:    Minimum |CE| required to continue
    """

    def __init__(
        self,
        causal_graph: TypedCausalGraph,
        max_hops: int = 5,
        threshold: float = 0.1,
    ) -> None:
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.graph = causal_graph
        self.max_hops = max_hops
        self.threshold = threshold

    # ── Primary tracing API ───────────────────────────────────────────────────

    def trace_root_cause(
        self,
        target_node,
        causal_effects: Dict[Tuple, float],
    ) -> Tuple[object, List]:
        """
        Greedy single-path root cause tracing using |CE|.

        Returns:
            (root_cause_node, causal_chain) — chain in [target, ..., root] order.
        """
        chain = [target_node]
        current = target_node
        visited = {target_node}

        for _ in range(self.max_hops):
            upstream = self.graph.get_upstream_neighbors(current)
            if not upstream:
                break  # condition 1

            best_upstream, best_abs_ce = self._select_best_upstream(
                current, upstream, causal_effects
            )

            if best_abs_ce < self.threshold:
                break  # condition 2

            if best_upstream in visited:
                break  # condition 3

            chain = chain + [best_upstream]
            visited = visited | {best_upstream}
            current = best_upstream

        return chain[-1], chain

    def trace_top_k_paths(
        self,
        target_node,
        causal_effects: Dict[Tuple, float],
        k: int = 3,
    ) -> List[Tuple[object, List, float]]:
        """
        Beam-search enumeration of top-k highest-|CE|-summed paths.

        Path score = sum of |CE| along chain edges (always non-negative).
        """
        # Heap entries: (neg_score, current, chain, frozen_visited)
        heap: List = [(-0.0, target_node, [target_node], frozenset({target_node}))]
        completed: List[Tuple] = []

        for _ in range(self.max_hops):
            if not heap:
                break

            next_heap: List = []
            for _ in range(len(heap)):
                neg_score, current, chain, frozen_visited = heapq.heappop(heap)
                score = -neg_score

                upstream = self.graph.get_upstream_neighbors(current)
                if not upstream:
                    completed.append((chain[-1], chain, score))
                    continue

                expanded = False
                for u in upstream:
                    abs_ce = abs(causal_effects.get((u, current), 0.0))
                    if abs_ce < self.threshold:
                        continue
                    if u in frozen_visited:
                        continue
                    new_chain = chain + [u]
                    new_visited = frozen_visited | {u}
                    new_score = score + abs_ce
                    heapq.heappush(next_heap, (-new_score, u, new_chain, new_visited))
                    expanded = True

                if not expanded:
                    completed.append((chain[-1], chain, score))

            heap = heapq.nsmallest(k, next_heap)

        for neg_score, _, chain, _ in heap:
            completed.append((chain[-1], chain, -neg_score))

        completed.sort(key=lambda x: x[2], reverse=True)
        return completed[:k]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_best_upstream(
        self,
        current,
        upstream_nodes: List,
        causal_effects: Dict[Tuple, float],
    ) -> Tuple[object, float]:
        """
        Return (upstream_node, |CE|) with the strongest absolute causal
        influence on `current`.  Magnitude is what determines tracing
        priority — see the module docstring for rationale.
        """
        best = max(
            upstream_nodes,
            key=lambda u: abs(causal_effects.get((u, current), 0.0)),
        )
        return best, abs(causal_effects.get((best, current), 0.0))

    @staticmethod
    def score_chain(chain: List, causal_effects: Dict[Tuple, float]) -> float:
        """
        Additive |CE| score of a chain.  Sum of |CE| along
        chain[i+1] → chain[i] edges (direction follows tracer convention:
        chain[0] is the target, chain[-1] is the root).
        """
        if len(chain) < 2:
            return 0.0
        return sum(
            abs(causal_effects.get((chain[i + 1], chain[i]), 0.0))
            for i in range(len(chain) - 1)
        )