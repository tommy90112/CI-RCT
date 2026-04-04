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

Reference: CI-RCT_Thesis_Plan.md § 5.3.3, § 6.2
"""
from typing import Dict, List, Optional, Tuple

from model.hetero_ncm import HeteroNCM
from model.typed_causal_graph import TypedCausalGraph


def compute_asymmetric_causal_shapley(
    causal_effects: Dict[Tuple[int, int], float],
    causal_graph: TypedCausalGraph,
    target_node: int,
) -> Dict[int, float]:
    """
    Compute Asymmetric Causal Shapley Values for all parents of target_node.

    Uses prefix-coalition approach:
      1. Get Pa(target_node) from the causal graph.
      2. Sort parents by topological order (earliest = index 0).
      3. For each parent u_k, φ_k = [v(S∪{u_k}) − v(S)] / n
         where v(S) is approximated as Σ_{u ∈ S} CE(u → target).

    Args:
        causal_effects: Pre-computed {(src, dst): CE_float} from HeteroNCM
        causal_graph:   TypedCausalGraph (provides parents + topological order)
        target_node:    The fraud node whose parents we are attributing

    Returns:
        {parent_node_id: phi_asymmetric} — Shapley values for each parent.
        Empty dict if the target node has no parents.
    """
    parents = list(causal_graph.parents(target_node))
    n = len(parents)

    if n == 0:
        return {}

    # Sort parents by global topological order (earlier = lower index = root)
    topo_order = causal_graph.topological_order()
    topo_idx: Dict[object, int] = {v: i for i, v in enumerate(topo_order)}

    parents_sorted = sorted(
        parents,
        key=lambda p: topo_idx.get(p, len(topo_order)),
    )

    phi: Dict[int, float] = {}
    coalition_value = 0.0  # v(S_k), starts at 0 for empty coalition

    for k, parent in enumerate(parents_sorted):
        ce = causal_effects.get((parent, target_node), 0.0)

        # v(S_k ∪ {parent}) − v(S_k) under additive approximation = CE(parent → target)
        marginal = ce
        phi[parent] = marginal / n

        # Update coalition value for next iteration
        coalition_value += ce

    return phi


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
