"""L3 causal feature attribution — the "why is this node dangerous" drill-down.

Where φ_asym attributes the target's fraud prediction across upstream *nodes*,
this module zooms into ONE node (the pivot) and attributes across its input
*features*. It is kept causally consistent with the rest of CI-RCT:

    CFE(j) = P_fraud(target) − P_fraud(target | do(x_pivot[j] = baseline))

i.e. the *feature-level analogue of CE* (which is the do-intervention on an
edge). We set the standardised feature to its population mean (baseline=0.0,
features are z-scored by the loader) and re-run the backbone, reading out the
ORIGINAL transaction target's fraud probability — so it works whether the pivot
is a wallet or a transaction (the readout never moves).

Cost control: a single Grad×Input backward shortlists the top-K candidate
features; only those get the (more expensive) per-feature do-intervention. The
Grad×Input saliency vector is returned alongside each feature and also serves as
the **fallback** ranking when every causal effect is ~0 (spec §12: "保留
saliency 作為因果效果不好的備案").

All forwards run on the target's L-hop receptive-field subgraph (numerically
exact for a purely-local backbone), so each do-intervention is cheap.

Immutability: the input ``data`` is never mutated. Each intervention builds a
lightweight HeteroData that shares every tensor by reference except the pivot
type's feature matrix, which is cloned before a single column is overwritten.
"""
from typing import Dict, List, Optional

import torch
from torch_geometric.data import HeteroData

from model.coalition_value import _num_rows, build_receptive_field_subgraph
from model.typed_causal_graph import TypedCausalGraph
from utils.feature_names import is_anonymous_feature

# A causal effect smaller than this (across all features) means the do-pass found
# nothing → fall back to the saliency ranking.
_CFE_EPS = 1e-5


def _graph_with_replaced_x(graph: HeteroData, ntype: str, new_x: torch.Tensor) -> HeteroData:
    """Shallow copy of ``graph`` sharing all tensors, overriding ``graph[ntype].x``."""
    new = HeteroData()
    for nt in graph.node_types:
        store = graph[nt]
        x = getattr(store, "x", None)
        if nt == ntype:
            new[nt].x = new_x
        elif x is not None:
            new[nt].x = x  # shared reference; forward only reads it
        else:
            new[nt].num_nodes = _num_rows(graph, nt)
    for etype in graph.edge_types:
        new[etype].edge_index = graph[etype].edge_index
    return new


def _target_prob(model: torch.nn.Module, graph: HeteroData, target_local: int, fraud_class: int) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward(graph)
    return float(torch.softmax(logits[target_local], dim=-1)[fraud_class].item())


def compute_causal_feature_attribution(
    *,
    model: torch.nn.Module,
    data: HeteroData,
    causal_graph: TypedCausalGraph,
    target_node: int,
    pivot_node: int,
    type_offsets: Dict[str, int],
    target_node_type: str,
    feature_names: Dict[str, List[str]],
    fraud_class: int = 1,
    use_subgraph: bool = True,
    num_layers: Optional[int] = None,
    shortlist_k: int = 20,
    top_k: int = 12,
    baseline: float = 0.0,
) -> dict:
    """Attribute ``target``'s fraud probability to ``pivot``'s input features.

    Returns a record::

        {
          "node": pivot_node,
          "node_type": "wallet" | "transaction",
          "method": "causal_do" | "saliency_fallback",
          "anonymous": bool,            # pivot type's features are mostly opaque
          "features": [ {name, value, saliency}, ... ]   # ≤ top_k, |value|-sorted
        }

    ``value`` is the signed causal feature effect CFE(j) (or the saliency value
    under the fallback). Returns an empty ``features`` list when the pivot lies
    outside the target's receptive field (it cannot reach the readout → no
    attribution, mirroring φ ≈ 0 there).
    """
    pivot_type = causal_graph.node_type.get(pivot_node, "")
    names = feature_names.get(pivot_type, [])
    base_record = {
        "node": int(pivot_node),
        "node_type": pivot_type,
        "method": "causal_do",
        "anonymous": bool(names and all(is_anonymous_feature(n) for n in names)),
        "features": [],
    }
    if target_node_type not in type_offsets or pivot_type not in type_offsets:
        return base_record
    # The readout head only exists for target_node_type; a mixed-type caller
    # (joint dual seeds) handing us a target of another type would turn its
    # GLOBAL id into a bogus local index and corrupt the receptive-field seed.
    if causal_graph.node_type.get(target_node) != target_node_type:
        return base_record

    target_local = target_node - type_offsets[target_node_type]
    pivot_local = pivot_node - type_offsets[pivot_type]
    if not (0 <= target_local < _num_rows(data, target_node_type)):
        return base_record
    if not (0 <= pivot_local < _num_rows(data, pivot_type)):
        return base_record

    # ── Restrict to the target's L-hop receptive field (exact + cheap) ──────────
    active = data
    if use_subgraph and num_layers:
        sub, _keep_old, remap, new_tl = build_receptive_field_subgraph(
            data, target_node_type, target_local, num_layers
        )
        new_pivot = remap.get(pivot_type, {}).get(pivot_local) if remap else None
        if sub is None or new_tl is None or new_pivot is None:
            return base_record  # pivot unreachable → no attribution
        active = sub
        target_local = new_tl
        pivot_local = new_pivot

    pivot_x = getattr(active[pivot_type], "x", None)
    if pivot_x is None or not (0 <= pivot_local < pivot_x.size(0)):
        return base_record
    n_feat = pivot_x.size(1)

    # ── Grad×Input saliency over the pivot's features (one backward) ────────────
    leaf = pivot_x.detach().clone().requires_grad_(True)
    grad_graph = _graph_with_replaced_x(active, pivot_type, leaf)
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits, _ = model.forward(grad_graph)
        prob = torch.softmax(logits[target_local], dim=-1)[fraud_class]
        prob.backward()
    saliency_vec = (
        (leaf.grad[pivot_local] * leaf[pivot_local]).detach()
        if leaf.grad is not None
        else torch.zeros(n_feat)
    )

    # Shortlist the features worth a (costlier) do-intervention.
    k = min(shortlist_k, n_feat)
    shortlist = torch.topk(saliency_vec.abs(), k).indices.tolist()

    # ── Per-feature do-intervention: CFE(j) = P0 − P(do x_j = baseline) ─────────
    p0 = _target_prob(model, active, target_local, fraud_class)
    rows = []
    for j in shortlist:
        x_do = pivot_x.detach().clone()
        x_do[pivot_local, j] = baseline
        pj = _target_prob(model, _graph_with_replaced_x(active, pivot_type, x_do), target_local, fraud_class)
        rows.append({
            "name": names[j] if j < len(names) else f"feature_{j}",
            "value": float(p0 - pj),
            "saliency": float(saliency_vec[j].item()),
        })

    causal_ok = any(abs(r["value"]) >= _CFE_EPS for r in rows)
    if not causal_ok:
        # Fallback: causal pass was flat → rank by saliency (spec §12 contingency).
        base_record["method"] = "saliency_fallback"
        for r in rows:
            r["value"] = r["saliency"]

    rows.sort(key=lambda r: abs(r["value"]), reverse=True)
    base_record["features"] = rows[:top_k]
    return base_record
