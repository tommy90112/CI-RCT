"""
RootCauseTracer — Module 3 of CI-RCT.

Backward causal path tracing over CE scores on a TypedCausalGraph.

Starting from a fraud-predicted target node, the tracer follows the highest
CE-scored upstream edge at each hop until a stopping criterion is met.

Stopping conditions (evaluated in priority order):
    1. No upstream neighbours exist  → current node is the root
    2. Best upstream CE < threshold  → causal signal too weak to continue
    3. Cycle detected (visited set)  → prevent infinite loops
    4. max_hops depth reached        → hard depth limit

Also provides trace_top_k_paths() for beam-search enumeration of top-k paths.

Reference: CI-RCT_Thesis_Plan.md § 5.4
"""
import heapq
from typing import Dict, List, Optional, Tuple

from model.typed_causal_graph import TypedCausalGraph


class RootCauseTracer:
    """
    Greedy backward BFS root cause tracer.

    Args:
        causal_graph: TypedCausalGraph that defines graph structure and
                      directed/undirected upstream adjacency
        max_hops:     Maximum number of hops to trace (depth limit)
        threshold:    Minimum CE score required to continue tracing
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
        Greedy single-path root cause tracing.

        Args:
            target_node:    Starting node (predicted as fraud / anomalous)
            causal_effects: {(source, target): CE_score}  (plain float values)

        Returns:
            (root_cause_node, causal_chain)
            causal_chain: [target_node, …, root_cause_node]
        """
        chain = [target_node]
        current = target_node
        visited = {target_node}

        for _ in range(self.max_hops):
            upstream = self.graph.get_upstream_neighbors(current)

            if not upstream:
                break  # Condition 1: no further upstream nodes

            best_upstream, best_ce = self._select_best_upstream(
                current, upstream, causal_effects
            )

            if best_ce < self.threshold:
                break  # Condition 2: CE below minimum signal threshold

            if best_upstream in visited:
                break  # Condition 3: cycle guard

            chain = chain + [best_upstream]   # immutable: new list each hop
            visited = visited | {best_upstream}
            current = best_upstream
            # Condition 4 handled by for-loop bound

        root_cause_node = chain[-1]
        return root_cause_node, chain

    def trace_top_k_paths(
        self,
        target_node,
        causal_effects: Dict[Tuple, float],
        k: int = 3,
    ) -> List[Tuple[object, List, float]]:
        """
        Beam-search enumeration of top-k highest-scoring causal paths.

        Path score = sum of CE values along the chain edges.

        Args:
            target_node:    Starting node
            causal_effects: {(source, target): CE_score}
            k:              Number of candidate paths to return

        Returns:
            List of (root_cause_node, causal_chain, path_score),
            sorted by path_score descending.
        """
        # Max-heap via negated scores: (neg_score, current, chain, visited)
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
                    ce = causal_effects.get((u, current), 0.0)
                    if ce < self.threshold:
                        continue
                    if u in frozen_visited:
                        continue
                    new_chain = chain + [u]
                    new_visited = frozen_visited | {u}
                    new_score = score + ce
                    heapq.heappush(next_heap, (-new_score, u, new_chain, new_visited))
                    expanded = True

                if not expanded:
                    completed.append((chain[-1], chain, score))

            # Keep only the top-k beams to bound memory
            heap = heapq.nsmallest(k, next_heap)

        # Flush remaining active beams
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
        """Return (node, ce_score) for the upstream node with highest CE toward current."""
        best = max(
            upstream_nodes,
            key=lambda u: causal_effects.get((u, current), 0.0),
        )
        return best, causal_effects.get((best, current), 0.0)

    @staticmethod
    def score_chain(chain: List, causal_effects: Dict[Tuple, float]) -> float:
        """
        Compute the additive CE score of a causal chain.

        Args:
            chain:          [target, n1, n2, ..., root]
            causal_effects: {(source, target): CE_score}

        Returns:
            float: Sum of CE values along edges chain[i+1] → chain[i]
        """
        if len(chain) < 2:
            return 0.0
        return sum(
            causal_effects.get((chain[i + 1], chain[i]), 0.0)
            for i in range(len(chain) - 1)
        )
