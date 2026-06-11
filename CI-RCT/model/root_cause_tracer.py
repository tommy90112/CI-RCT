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
from typing import Callable, Dict, List, Optional, Set, Tuple

from model.typed_causal_graph import TypedCausalGraph

# (current_node, upstream_nodes) -> {upstream_node: ranking_score}. Lets a
# caller override the default |CE| ranking with e.g. asymmetric Shapley φ.
UpstreamScoreFn = Callable[[object, List], Dict[object, float]]


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
        prefer_root_types: Optional[Set[str]] = None,
        prefer_reachable_depth: int = 0,
        tracer_algorithm: str = "greedy",
        tracer_objective: str = "product",
        ce_eps: float = 1e-12,
    ) -> None:
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if prefer_reachable_depth < 0:
            raise ValueError(
                f"prefer_reachable_depth must be >= 0, got {prefer_reachable_depth}"
            )
        # Ablation knob: which backward-search algorithm trace_root_cause runs.
        # "greedy" (default) and "beam" use the legacy code paths below, kept
        # byte-for-byte identical; every other value dispatches to a pluggable
        # strategy in model/tracer_strategies (see trace_root_cause). The
        # objective / ce_eps only matter for the weighted-path strategies
        # (dag_dp, dijkstra): product = max-product (-log|CE|), sum = max-sum|CE|.
        from model.tracer_strategies import ALL_ALGORITHMS
        if tracer_algorithm not in ALL_ALGORITHMS:
            raise ValueError(
                f"tracer_algorithm must be one of {ALL_ALGORITHMS}, got "
                f"'{tracer_algorithm}'"
            )
        if tracer_objective not in ("product", "sum"):
            raise ValueError(
                f"tracer_objective must be 'product' or 'sum', got '{tracer_objective}'"
            )
        self.tracer_algorithm = tracer_algorithm
        self.tracer_objective = tracer_objective
        self.ce_eps = ce_eps

        self.graph = causal_graph
        self.max_hops = max_hops
        self.threshold = threshold
        # Opt-in type-aware tie-break. When set, among the upstream parents
        # that clear the CE threshold, those whose node type is in
        # prefer_root_types (the root-capable / labelable types, e.g.
        # {"process_node"}) win over other types, ranked by |CE| within the
        # preferred group. This stops the greedy search from being diverted
        # into same-type relay hops (e.g. the host→host DD-11 bridge whose
        # |CE| can marginally exceed process→host) and dead-ending at a node
        # that can never be a true root cause. None ⇒ legacy |CE|-only ranking.
        self.prefer_root_types = prefer_root_types
        # Opt-in LOOKAHEAD tie-break (DD-18). prefer_root_types alone only
        # inspects the *immediate* upstream's type — but on MG24 a flow's only
        # parent is an IP host whose 12 upstream parents are ALL host→host
        # bridges (no process in sight); the greedy |CE|-max then picks a
        # dead-end bridge host (0 parents) while the 2 bridge branches that
        # actually lead to a process_node have marginally smaller |CE| and are
        # skipped. With prefer_reachable_depth = d > 0, among threshold-passing
        # candidates we first prefer those whose ancestors within d backward
        # hops include a prefer_root_types node — i.e. candidates from which a
        # true root is still REACHABLE — ranked by |CE| within that group.
        # 0 ⇒ disabled, behaviour identical to prefer_root_types-only / legacy.
        self.prefer_reachable_depth = prefer_reachable_depth
        # Memoised reachability: gid → bool (can reach a prefer type within
        # prefer_reachable_depth backward hops). Built lazily per node.
        self._reach_cache: Dict[object, bool] = {}

    # ── Primary tracing API ───────────────────────────────────────────────────

    def trace_root_cause(
        self,
        target_node,
        causal_effects: Dict[Tuple, float],
        upstream_score_fn: Optional["UpstreamScoreFn"] = None,
    ) -> Tuple[object, List]:
        """
        Greedy single-path root cause tracing.

        By default ranks upstream parents by |CE| (the legacy behaviour). When
        ``upstream_score_fn`` is given, that callable supplies the ranking score
        per upstream node instead — e.g. asymmetric Causal Shapley φ recomputed
        for the current node's parents (see evaluate.py's φ-weighted explainer).
        The threshold / prefer_root_types / lookahead tie-breaks all operate on
        whichever score is in effect.

        Args:
            target_node:        Node to trace back from.
            causal_effects:     {(src, dst): CE_float}; used for the |CE| score
                                and by the additive top-k path scorer.
            upstream_score_fn:  Optional (current, upstream_nodes) -> {u: score}.
                                None ⇒ legacy |CE| ranking (byte-identical).

        Returns:
            (root_cause_node, causal_chain) — chain in [target, ..., root] order.
        """
        # ── Pluggable strategy dispatch (ablation) ──────────────────────────
        # "greedy" runs the legacy code below (byte-identical). "beam" returns
        # the top-1 of the legacy beam search. Every other algorithm delegates
        # to a model.tracer_strategies function; those rank purely by |CE|, so
        # upstream_score_fn (phi-weighted ranking) stays a greedy-only feature.
        if self.tracer_algorithm == "beam":
            paths = self.trace_top_k_paths(target_node, causal_effects, k=3)
            if not paths:
                return target_node, [target_node]
            root, beam_chain, _score = paths[0]
            return root, beam_chain
        if self.tracer_algorithm != "greedy":
            from model.tracer_strategies import resolve
            return resolve(self.tracer_algorithm)(
                target_node,
                causal_effects,
                graph=self.graph,
                max_hops=self.max_hops,
                threshold=self.threshold,
                prefer_root_types=self.prefer_root_types,
                ce_eps=self.ce_eps,
                objective=self.tracer_objective,
            )

        chain = [target_node]
        current = target_node
        visited = {target_node}

        for _ in range(self.max_hops):
            upstream = self.graph.get_upstream_neighbors(current)
            if not upstream:
                break  # condition 1

            best_upstream, best_score = self._select_best_upstream(
                current, upstream, causal_effects, upstream_score_fn
            )

            if best_score < self.threshold:
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
        upstream_score_fn: Optional["UpstreamScoreFn"] = None,
    ) -> Tuple[object, float]:
        """
        Return (upstream_node, score) with the strongest ranking score on
        `current`.  The default score is |CE| magnitude — see the module
        docstring for rationale; an ``upstream_score_fn`` overrides it (e.g.
        asymmetric Shapley φ over `current`'s parents).

        Preference order among threshold-passing candidates (each tier ranked
        by score internally; fall through to the next tier only when a tier is
        empty):
            1. prefer_root_types  — candidate IS a root-capable type
            2. prefer_reachable_depth > 0 — candidate can still REACH a
               root-capable type within d backward hops (lookahead): skips the
               dead-end bridge hosts that the greedy score-max would otherwise
               pick, when a true root lies just past a marginally-smaller branch
               (see __init__ DD-18 note).
            3. global score-max over all parents — legacy fallback.

        With prefer_root_types=None, prefer_reachable_depth=0 and
        upstream_score_fn=None this is byte-for-byte the legacy |CE|-only
        ranking.
        """
        if upstream_score_fn is not None:
            scores = upstream_score_fn(current, list(upstream_nodes))

            def abs_ce(u):
                return scores.get(u, 0.0)
        else:
            def abs_ce(u):
                return abs(causal_effects.get((u, current), 0.0))

        if self.prefer_root_types:
            preferred = [
                u for u in upstream_nodes
                if abs_ce(u) >= self.threshold
                and self.graph.node_type.get(u) in self.prefer_root_types
            ]
            if preferred:
                best = max(preferred, key=abs_ce)
                return best, abs_ce(best)

        if self.prefer_reachable_depth > 0 and self.prefer_root_types:
            reachable = [
                u for u in upstream_nodes
                if abs_ce(u) >= self.threshold
                and self._can_reach_prefer(u, self.prefer_reachable_depth)
            ]
            if reachable:
                best = max(reachable, key=abs_ce)
                return best, abs_ce(best)

        best = max(upstream_nodes, key=abs_ce)
        return best, abs_ce(best)

    def _can_reach_prefer(self, node, depth: int) -> bool:
        """
        True if `node` itself is a prefer_root_types node, or one of its
        backward ancestors within `depth` hops is. Memoised on `node`
        (cache keyed by the start node; the depth is fixed per tracer run).

        Used by the lookahead tie-break so the greedy search prefers upstream
        branches from which a true root cause is still reachable, instead of
        dead-ending at a same-type relay node whose single-hop |CE| happens to
        be largest.
        """
        if node in self._reach_cache:
            return self._reach_cache[node]

        target_types = self.prefer_root_types or set()
        visited = {node}
        # BFS layer by layer up to `depth` hops; the start node counts as a
        # hit too (handles the case where `node` is already a prefer type).
        frontier = [node]
        hops = 0
        found = False
        while frontier and hops <= depth:
            next_frontier = []
            for v in frontier:
                if self.graph.node_type.get(v) in target_types:
                    found = True
                    break
                if hops == depth:
                    continue  # don't expand past the budget
                for p in self.graph.get_upstream_neighbors(v):
                    if p not in visited:
                        visited.add(p)
                        next_frontier.append(p)
            if found:
                break
            frontier = next_frontier
            hops += 1

        self._reach_cache[node] = found
        return found

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