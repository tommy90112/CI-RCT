"""
Asymmetric Causal Shapley Value — Module 2c of CI-RCT.

Computes Asymmetric Causal Shapley Values (Frye et al., NeurIPS 2020;
Heskes et al., NeurIPS 2020) for parent nodes of a target fraud node.

Core insight:
    Standard (symmetric) Causal Shapley considers all 2^n parent subsets,
    treating every ordering equally.  In a temporal fund-flow graph, however,
    the causal ordering is known — earlier nodes should bear more causal
    responsibility ("bias towards distal causes").

    Asymmetric Causal Shapley restricts coalitions to topologically ordered
    prefixes: S_k = {u_0, u_1, ..., u_{k-1}} where u_0 is the earliest
    (most upstream) parent.  Only n coalitions are evaluated instead of 2^n,
    making exact computation tractable.

    φ_k^{asym} = [ v(S_k ∪ {u_k}) − v(S_k) ] / n

    where the coalition value v(S) is the expected target-node fraud
    probability under do-calculus intervention: non-S parents are set to
    the null baseline (zero embedding).

Under the additive independence assumption (used here as a practical
approximation), the coalition value factorises as:
    v(S) ≈ Σ_{u ∈ S} CE(u → target)
so φ_k^{asym} ≈ CE(u_k → target) / n.

The key property preserved: because earlier (root-cause) nodes contribute
CE without being "pre-empted" by earlier coalition members, whereas later
(proximate) nodes are evaluated on top of an already-present coalition,
the scores are semantically consistent for backward tracing.

Non-additive coalition values
------------------------------
The additive approximation v(S) ≈ Σ CE collapses the *ordering* information
that distinguishes asymmetric from symmetric Shapley: under additivity every
parent's marginal is its own CE regardless of coalition, so asymmetric ==
symmetric == CE-ranking (the per-edge HeteroNCM forces exactly this regime —
see hetero_ncm.py).  To make "asymmetric" carry empirical weight, pass an
explicit `coalition_value_fn: frozenset -> float` — e.g. a backbone
do-intervention value (model/coalition_value.py) whose GNN aggregation over the
retained parents is genuinely non-additive.  With such a v(S) the prefix
asymmetric value and the all-permutation symmetric value diverge.

Reference: CI-RCT_Thesis_Plan.md § 5.3.3, § 6.2
"""
import random
from typing import Callable, Dict, List, Optional, Tuple

from model.hetero_ncm import HeteroNCM
from model.typed_causal_graph import TypedCausalGraph

# A coalition value: maps a subset of parent node ids to the target's expected
# fraud score under that coalition (others intervened to baseline).
CoalitionValueFn = Callable[[frozenset], float]


def _topo_sorted_parents(causal_graph: TypedCausalGraph, target_node: int) -> List[int]:
    """Parents of target_node sorted by topological order (earliest = root)."""
    parents = list(causal_graph.parents(target_node))
    if not parents:
        return []
    # Use the cached topological_index — building this dict from scratch on
    # every call costs O(V) and was the dominant evaluation bottleneck on
    # 1M-node Elliptic++ graphs (200 calls × ~20s each ≈ 67 minutes).
    topo_idx = causal_graph.topological_index()
    n_nodes = len(topo_idx)
    return sorted(parents, key=lambda p: topo_idx.get(p, n_nodes))


def compute_asymmetric_causal_shapley(
    causal_effects: Dict[Tuple[int, int], float],
    causal_graph: TypedCausalGraph,
    target_node: int,
    coalition_value_fn: Optional[CoalitionValueFn] = None,
) -> Dict[int, float]:
    """
    Compute Asymmetric Causal Shapley Values for all parents of target_node.

    Uses prefix-coalition approach:
      1. Get Pa(target_node) from the causal graph.
      2. Sort parents by topological order (earliest = index 0).
      3. For each parent u_k, φ_k = [v(S_k ∪ {u_k}) − v(S_k)] / n where the
         prefix coalition is S_k = {u_0, ..., u_{k-1}}.

    Args:
        causal_effects:     Pre-computed {(src, dst): CE_float} from HeteroNCM.
                            Used only for the additive fallback.
        causal_graph:       TypedCausalGraph (provides parents + topo order).
        target_node:        The fraud node whose parents we are attributing.
        coalition_value_fn: Optional v(S) callable. When given, the true
                            (potentially non-additive) marginal
                            v(S_k ∪ {u_k}) − v(S_k) is used. When None, falls
                            back to the legacy additive approximation
                            v(S) ≈ Σ CE ⇒ φ = CE(u → target) / n (byte-identical
                            to the pre-ablation behaviour; Metric D relies on it).

    Returns:
        {parent_node_id: phi_asymmetric} — Shapley values for each parent.
        Empty dict if the target node has no parents.
    """
    parents_sorted = _topo_sorted_parents(causal_graph, target_node)
    n = len(parents_sorted)
    if n == 0:
        return {}

    phi: Dict[int, float] = {}

    if coalition_value_fn is None:
        # ── Legacy additive approximation: φ_k = CE(u_k → target) / n ──────────
        for parent in parents_sorted:
            ce = causal_effects.get((parent, target_node), 0.0)
            phi[parent] = ce / n
        return phi

    # ── True prefix-coalition marginals via the injected v(S) ─────────────────
    prefix: List[int] = []
    v_prefix = coalition_value_fn(frozenset())  # v(∅)
    for parent in parents_sorted:
        v_with = coalition_value_fn(frozenset(prefix + [parent]))
        phi[parent] = (v_with - v_prefix) / n
        prefix.append(parent)
        v_prefix = v_with

    return phi


def compute_symmetric_causal_shapley(
    causal_graph: TypedCausalGraph,
    target_node: int,
    coalition_value_fn: CoalitionValueFn,
    n_permutations: int = 64,
    rng_seed: int = 0,
) -> Dict[int, float]:
    """
    Standard (symmetric) Shapley values for the parents of target_node.

    Averages each parent's marginal contribution v(S ∪ {i}) − v(S) over random
    permutations of the parent set (Monte-Carlo Shapley).  Unlike the
    asymmetric variant it gives *every* ordering equal weight, so it ignores
    the temporal/topological precedence of upstream causes.

    For small parent sets the permutation sampling converges to the exact
    Shapley value; the divergence from the prefix-only asymmetric value is the
    quantity the ablation isolates ("what does temporal asymmetry buy us?").

    Args:
        causal_graph:       TypedCausalGraph (provides the parent set).
        target_node:        The fraud node whose parents we are attributing.
        coalition_value_fn: v(S) callable (required — symmetric Shapley has no
                            CE additive fallback; it is only meaningful with a
                            real coalition value).
        n_permutations:     Number of random permutations to average over.
        rng_seed:           Deterministic seed (no global RNG state touched).

    Returns:
        {parent_node_id: phi_symmetric}. Empty dict if no parents.
    """
    parents = list(causal_graph.parents(target_node))
    n = len(parents)
    if n == 0:
        return {}

    # Exact enumeration is cheaper and exact for tiny parent sets; sampling is
    # only needed when n! blows up. n! ≤ n_permutations ⇒ enumerate all.
    import itertools
    import math

    if math.factorial(n) <= max(n_permutations, 1):
        orderings = list(itertools.permutations(parents))
    else:
        rng = random.Random(rng_seed)
        orderings = []
        for _ in range(n_permutations):
            perm = parents[:]
            rng.shuffle(perm)
            orderings.append(tuple(perm))

    totals: Dict[int, float] = {p: 0.0 for p in parents}
    # Memoise v(S) across permutations — many prefixes repeat.
    cache: Dict[frozenset, float] = {}

    def v(S: frozenset) -> float:
        if S not in cache:
            cache[S] = coalition_value_fn(S)
        return cache[S]

    for perm in orderings:
        prefix: List[int] = []
        v_prefix = v(frozenset())
        for parent in perm:
            v_with = v(frozenset(prefix + [parent]))
            totals[parent] += v_with - v_prefix
            prefix.append(parent)
            v_prefix = v_with

    return {p: totals[p] / len(orderings) for p in parents}


def compute_shapley_edge_scores(
    phi: Dict[int, float],
    causal_effects: Dict[Tuple[int, int], float],
    causal_graph: TypedCausalGraph,
    target_node: int,
) -> Dict[Tuple[int, int], float]:
    """
    Derive edge-level Shapley scores for visualisation.

    Each directed edge (u → v) in the local neighbourhood receives:
        edge_score(u → v) = φ(u) × CE(u → v)

    For edges not directly pointing at the target node:
        edge_score(u → v) = φ(u) × φ(v) × CE(u → v)

    Args:
        phi:            Asymmetric Shapley values from compute_asymmetric_causal_shapley
        causal_effects: {(src, dst): CE_float}
        causal_graph:   TypedCausalGraph
        target_node:    Fraud target node

    Returns:
        {(src, dst): edge_score}
    """
    edge_scores: Dict[Tuple[int, int], float] = {}

    for (src, dst), ce in causal_effects.items():
        phi_src = phi.get(src, 0.0)
        phi_dst = phi.get(dst, 0.0)

        if dst == target_node:
            # Direct parent edge: weight = φ_src
            edge_scores[(src, dst)] = abs(phi_src) * ce
        else:
            # Intermediate edge: weight = φ_src × φ_dst
            edge_scores[(src, dst)] = abs(phi_src) * abs(phi_dst) * ce

    return edge_scores