"""
Gradient-saliency upstream scorer — the CORRELATIONAL (non-causal) baseline.

Block-B ablation arm. Every other explainer in the registry ranks a hop's
parents by an *interventional* signal: ``ce_only`` uses |CE| (a do-intervention
on the parent edge), ``phi_*`` use Causal Shapley built on the backbone
do-intervention coalition value. None of them tests the actual scientific claim
behind the model's name — that an *intervention* explains better than plain
*correlation*.

This arm supplies that contrast. It shares the φ explainers' readout EXACTLY
(same original target, same L-hop receptive field, same fraud-class logit), but
replaces the interventional score with a Grad×Input saliency:

    saliency(u) = ‖ x_u ⊙ ∂P_fraud(target) / ∂x_u ‖₂

i.e. how sensitive the target's fraud probability is to node ``u``'s input
features, WITHOUT cutting any edge. This is the canonical correlational GNN
attribution (Saliency / Grad×Input family). The gap between ``phi_asym`` and
``saliency`` is the empirical value of causal intervention over correlation.

One forward + one backward per target (the map is computed once and cached over
all hops), so this arm is markedly cheaper than the φ arms' n coalition
forwards.
"""
from typing import Callable, Dict, List, Optional

import torch
from torch_geometric.data import HeteroData

from model.coalition_value import _num_rows, build_receptive_field_subgraph
from model.typed_causal_graph import TypedCausalGraph


def make_saliency_score_fn(
    *,
    model: torch.nn.Module,
    data: HeteroData,
    causal_graph: TypedCausalGraph,
    target_node: int,
    type_offsets: Dict[str, int],
    target_node_type: str,
    fraud_class: int = 1,
    use_subgraph: bool = False,
    num_layers: Optional[int] = None,
) -> Callable[[object, List], Dict[object, float]]:
    """Return an ``upstream_score_fn(current, upstream) -> {u: saliency}``.

    The saliency map over the target's receptive field is computed lazily on the
    first call and reused for every hop (it depends only on the fixed target, not
    on ``current``), so the tracer pays a single forward+backward per target.

    Args mirror ``_make_phi_score_fn`` so the two arms are drop-in comparable.
    Parents outside the target's ``num_layers``-hop receptive field get score 0.0
    (they cannot reach the readout — the same semantics as φ ≈ 0 there).
    """
    if target_node_type not in type_offsets:
        raise KeyError(
            f"target_node_type '{target_node_type}' not in type_offsets "
            f"{sorted(type_offsets)}"
        )

    cache: Dict[str, Dict[int, float]] = {}

    def _compute_saliency_map() -> Dict[int, float]:
        target_local = target_node - type_offsets[target_node_type]
        active = data
        row2global = {
            nt: torch.arange(_num_rows(data, nt), dtype=torch.long)
            + type_offsets.get(nt, 0)
            for nt in data.node_types
        }

        # Restrict to the L-hop incoming receptive field: a purely-local
        # message-passing backbone makes the target logit depend ONLY on it, so
        # the saliency outside it is exactly 0 and the backward stays cheap.
        if use_subgraph and num_layers:
            sub, keep_old, _remap, new_tl = build_receptive_field_subgraph(
                data, target_node_type, target_local, num_layers
            )
            if sub is not None and new_tl is not None:
                active = sub
                target_local = new_tl
                row2global = {
                    nt: torch.tensor(olds, dtype=torch.long)
                    + type_offsets.get(nt, 0)
                    for nt, olds in keep_old.items()
                }

        # Build a graph whose node features are differentiable leaves; edges are
        # shared by reference (the forward never writes them).
        leaves: Dict[str, torch.Tensor] = {}
        grad_graph = HeteroData()
        for nt in active.node_types:
            x = getattr(active[nt], "x", None)
            if x is None:
                grad_graph[nt].num_nodes = _num_rows(active, nt)
                continue
            leaf = x.detach().clone().requires_grad_(True)
            leaves[nt] = leaf
            grad_graph[nt].x = leaf
        for etype in active.edge_types:
            grad_graph[etype].edge_index = active[etype].edge_index

        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits, _ = model.forward(grad_graph)
            prob = torch.softmax(logits[target_local], dim=-1)[fraud_class]
            prob.backward()

        saliency: Dict[int, float] = {}
        for nt, leaf in leaves.items():
            if leaf.grad is None:
                continue
            # Grad×Input, L2 over the feature dim → one scalar per node. Scale
            # aware, the standard correlational attribution readout.
            contrib = (leaf.grad * leaf).norm(dim=1)
            g_ids = row2global[nt]
            for i in range(contrib.size(0)):
                saliency[int(g_ids[i])] = float(contrib[i].item())
        return saliency

    def score_fn(current, upstream):
        if "map" not in cache:
            cache["map"] = _compute_saliency_map()
        smap = cache["map"]
        return {u: smap.get(int(u), 0.0) for u in upstream}

    return score_fn
