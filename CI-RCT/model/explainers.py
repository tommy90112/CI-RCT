"""
Explainer registry — pluggable "explain one fraud node → set of nodes".

All explainers share the same socket so evaluate.py's Metric C / D can score
them identically (recall / gt-match against the LFPN ground-truth set):

    explain(target_node, causal_effects) -> set[int]   # the explanatory nodes

Registered explainers (selected by ``--explainer``):

  * ``ce_only``  — legacy: greedy backward trace ranked by raw |CE| (no
                   Shapley). This is the pre-ablation behaviour; the φ machinery
                   is bypassed entirely. Byte-identical to the prior Metric C.
  * ``phi_asym`` — main method: φ-weighted trace where each hop ranks the
                   current node's parents by ASYMMETRIC (prefix) Causal Shapley
                   φ, computed from the backbone do-intervention coalition value
                   (model/coalition_value.py). Temporal precedence credits
                   distal causes.
  * ``phi_sym``  — ablation Abl-1: identical but with SYMMETRIC Shapley φ (all
                   orderings weighted equally). The gap phi_asym − phi_sym is
                   the empirical value of temporal asymmetry.
  * ``cxgnn_ncm``— external SOTA baseline (CXGNN, ECCV 2024): per-target local
                   NCM subgraph. Implemented in model/cxgnn_ncm_adapter.py and
                   wired in here when available.

The φ explainers read out the ORIGINAL target's fraud probability while
intervening on each hop's parent edges (readout/intervene decoupling), so the
score measures how distal upstream nodes propagate to the queried fraud node
through the GNN — not the fraud-ness of intermediate (e.g. wallet) nodes that
have no classifier head.
"""
from typing import Callable, Dict, Set, Tuple

from model.causal_shapley import (
    compute_asymmetric_causal_shapley,
    compute_symmetric_causal_shapley,
)
from model.coalition_value import make_backbone_coalition_value_fn

# explain(target_node, causal_effects) -> set of explanatory node ids.
ExplainFn = Callable[[int, Dict[Tuple[int, int], float]], Set[int]]

EXPLAINER_CHOICES = ("ce_only", "phi_asym", "phi_sym", "cxgnn_ncm")


def _make_phi_score_fn(
    *, model, data, causal_graph, target_node, type_offsets,
    target_node_type, fraud_class, mode, n_permutations,
    causal_effects, shapley_topk=None,
):
    """Build an upstream_score_fn that ranks a hop's parents by Causal Shapley φ.

    The coalition value reads out ``target_node``'s fraud probability while the
    coalition controls the *current* hop's parent edges (intervene_node=current).

    ``shapley_topk`` (when > 0) caps the parents fed to Shapley to the top-k by
    |CE| BEFORE the coalition forwards. Each coalition is a full backbone
    forward, so an uncapped high-in-degree node triggers O(n) (asym) / O(n·perm)
    (sym) full-graph forwards and becomes intractable on Elliptic++. The dropped
    low-|CE| parents contribute negligibly. None/0 ⇒ no cap (byte-identical).
    """
    def score_fn(current, upstream):
        vfn = make_backbone_coalition_value_fn(
            model=model,
            data=data,
            causal_graph=causal_graph,
            target_node=target_node,
            type_offsets=type_offsets,
            target_node_type=target_node_type,
            fraud_class=fraud_class,
            intervene_node=current,
        )
        parents = None
        if shapley_topk and shapley_topk > 0:
            all_parents = list(causal_graph.parents(current))
            if len(all_parents) > shapley_topk:
                parents = sorted(
                    all_parents,
                    key=lambda p: abs(causal_effects.get((p, current), 0.0)),
                    reverse=True,
                )[:shapley_topk]
        if mode == "asym":
            return compute_asymmetric_causal_shapley(
                {}, causal_graph, current, coalition_value_fn=vfn, parents=parents
            )
        return compute_symmetric_causal_shapley(
            causal_graph, current, coalition_value_fn=vfn,
            n_permutations=n_permutations, parents=parents,
        )

    return score_fn


def build_explainer(
    name: str,
    *,
    model,
    data,
    causal_graph,
    tracer,
    type_offsets: Dict[str, int],
    target_node_type: str,
    fraud_class: int = 1,
    n_permutations: int = 64,
    shapley_topk: int = None,
    cxgnn_kwargs: dict = None,
) -> ExplainFn:
    """Return an ``explain(target, causal_effects) -> set[int]`` for ``name``.

    Args:
        name:             One of EXPLAINER_CHOICES.
        model/data:       Trained CI_RCT model and the (local) HeteroData.
        causal_graph:     TypedCausalGraph for the local subgraph.
        tracer:           A configured RootCauseTracer (shared across the run).
        type_offsets:     {node_type: global offset}.
        target_node_type: The classifier's target node type (readout type).
        fraud_class:      Logit column of the fraud class (default 1).
        n_permutations:   Monte-Carlo permutations for symmetric Shapley.
        cxgnn_kwargs:     Extra args forwarded to the CXGNN-NCM adapter.
    """
    if name not in EXPLAINER_CHOICES:
        raise ValueError(
            f"Unknown explainer '{name}'. Choices: {EXPLAINER_CHOICES}"
        )

    if name == "ce_only":
        def explain(target, causal_effects):
            _, chain = tracer.trace_root_cause(target, causal_effects)
            return set(chain)
        return explain

    if name in ("phi_asym", "phi_sym"):
        mode = "asym" if name == "phi_asym" else "sym"

        def explain(target, causal_effects):
            score_fn = _make_phi_score_fn(
                model=model, data=data, causal_graph=causal_graph,
                target_node=target, type_offsets=type_offsets,
                target_node_type=target_node_type, fraud_class=fraud_class,
                mode=mode, n_permutations=n_permutations,
                causal_effects=causal_effects, shapley_topk=shapley_topk,
            )
            _, chain = tracer.trace_root_cause(
                target, causal_effects, upstream_score_fn=score_fn
            )
            return set(chain)
        return explain

    if name == "cxgnn_ncm":
        from model.cxgnn_ncm_adapter import build_cxgnn_ncm_explainer
        return build_cxgnn_ncm_explainer(
            model=model, data=data, causal_graph=causal_graph,
            type_offsets=type_offsets, target_node_type=target_node_type,
            **(cxgnn_kwargs or {}),
        )

    raise AssertionError("unreachable")  # pragma: no cover
