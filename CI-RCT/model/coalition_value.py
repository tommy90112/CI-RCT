"""
Backbone do-intervention coalition value — Module 2c-bis of CI-RCT.

Provides the *non-additive* coalition value v(S) that the asymmetric Causal
Shapley needs in order to be more than CE-ranking.

Motivation
----------
The per-edge HeteroNCM (hetero_ncm.py) computes CE(u → v) from u's embedding
alone, so the only coalition value it can induce is the strictly additive
v(S) = Σ_{u∈S} CE(u → target).  Under additivity asymmetric == symmetric ==
CE-ranking, and "asymmetric Shapley" buys nothing (see causal_shapley.py).

This module instead defines

    v(S) = P( ŷ_target = fraud | do( cut every parent-edge p → target
                                     for p ∈ Pa(target) \ S ) )

evaluated by re-running the HGT backbone on a graph where the non-coalition
parent edges are removed.  Because HGT message passing aggregates the retained
parents non-linearly, v is genuinely non-additive — v({a,b}) ≠ v({a}) + v({b})
in general — which is exactly what lets the prefix asymmetric value diverge
from the symmetric one.

This is Pearl/Heskes-style *interventional* (do-calculus) Shapley: the baseline
for an excluded parent is "edge removed" (no influence), not "embedding zeroed".

Design notes
------------
* Immutability: the original `data` is never mutated.  Each v(S) builds a
  lightweight HeteroData that *shares* node-feature tensors by reference (the
  forward pass only reads them) and substitutes a fresh edge_index only for the
  relations that carry controllable parent edges.  All other relations reuse the
  original tensors.
* Cost: one backbone forward per distinct S.  The asymmetric value needs n
  forwards per target (the topological prefixes); the symmetric value memoises
  v(S) across permutations.  Callers should restrict this to the Metric-C
  explanation set (``--max_explain``), not the full graph.
"""
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from model.typed_causal_graph import TypedCausalGraph

EdgeType = Tuple[str, str, str]


def _shallow_graph_with_edges(
    data: HeteroData,
    replaced_edge_index: Dict[EdgeType, Tensor],
) -> HeteroData:
    """Build a HeteroData sharing node features, overriding some edge_index.

    The backbone forward reads only ``data[ntype].x`` and
    ``data[etype].edge_index``; we copy references for both and swap in the
    replacement edge_index for the affected relations.  The source ``data`` is
    left untouched (immutability).
    """
    new = HeteroData()
    for ntype in data.node_types:
        store = data[ntype]
        if hasattr(store, "x") and store.x is not None:
            new[ntype].x = store.x  # shared reference; forward never writes it
        else:
            # Preserve node count for relations even when x is absent.
            new[ntype].num_nodes = store.num_nodes
    for etype in data.edge_types:
        new[etype].edge_index = replaced_edge_index.get(
            etype, data[etype].edge_index
        )
    return new


class BackboneCoalitionValue:
    """Callable v(S) for one target node, backed by backbone do-intervention.

    Instantiate once per target (it pre-indexes the controllable parent edges),
    then call it with parent-id subsets.  Results are cached per coalition.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: HeteroData,
        causal_graph: TypedCausalGraph,
        target_node: int,
        type_offsets: Dict[str, int],
        target_node_type: str,
        fraud_class: int = 1,
        intervene_node: Optional[int] = None,
    ) -> None:
        """
        Args:
            target_node:      The node whose fraud probability is *read out*
                              (the original fraud target — only this type has a
                              classifier head).
            intervene_node:   The node whose parent edges are cut/kept by the
                              coalition S. Defaults to ``target_node`` (direct
                              parents). When tracing multi-hop, pass the current
                              node so v(S) measures how *that* node's upstream
                              parents propagate to the original target's fraud
                              probability through the GNN. If the intervene node
                              lies outside the target's receptive field (deeper
                              than the HGT layer count), the intervention cannot
                              reach the readout and v(S) is constant → φ ≈ 0,
                              which correctly reflects "no traceable influence".
        """
        self.model = model
        self.data = data
        self.fraud_class = fraud_class

        # ── Readout node (where we read the fraud probability) ────────────────
        self.target_type = target_node_type
        if target_node_type not in type_offsets:
            raise KeyError(
                f"target_node_type '{target_node_type}' not in type_offsets "
                f"{sorted(type_offsets)}"
            )
        self.target_local = target_node - type_offsets[target_node_type]

        # ── Intervention node (whose parent edges the coalition controls) ─────
        intervene = target_node if intervene_node is None else intervene_node
        intervene_type = causal_graph.node_type.get(intervene, target_node_type)
        intervene_off = type_offsets.get(intervene_type, 0)
        intervene_local = intervene - intervene_off

        self.parents: List[int] = list(causal_graph.parents(intervene))
        self.parent_set = set(self.parents)

        # Pre-index, per affected relation, the columns whose dst is the
        # intervention node AND whose src is a (controllable) parent. ``ctrl``
        # maps each such relation to (column_indices LongTensor, src_global
        # LongTensor aligned to those columns).
        self._ctrl: Dict[EdgeType, Tuple[Tensor, Tensor]] = {}
        for etype in data.edge_types:
            src_type, _, dst_type = etype
            if dst_type != intervene_type:
                continue
            ei = data[etype].edge_index
            if ei is None or ei.numel() == 0:
                continue
            dst_is_target = ei[1] == intervene_local
            if not bool(dst_is_target.any()):
                continue
            src_off = type_offsets.get(src_type, 0)
            cols = torch.nonzero(dst_is_target, as_tuple=False).flatten()
            src_global = ei[0, cols] + src_off
            # Keep only columns whose source is an actual Shapley parent;
            # non-parent edges into the node are background and stay fixed.
            is_parent = torch.tensor(
                [int(s) in self.parent_set for s in src_global.tolist()],
                dtype=torch.bool,
            )
            if not bool(is_parent.any()):
                continue
            self._ctrl[etype] = (cols[is_parent], src_global[is_parent])

        self._cache: Dict[frozenset, float] = {}

    # ── Callable interface ────────────────────────────────────────────────────

    def __call__(self, S) -> float:
        key = frozenset(S)
        if key not in self._cache:
            self._cache[key] = self._evaluate(key)
        return self._cache[key]

    def _evaluate(self, S: frozenset) -> float:
        # For each affected relation, drop the controllable parent columns whose
        # source is NOT in the coalition S (do: cut those parent edges).
        replaced: Dict[EdgeType, Tensor] = {}
        for etype, (cols, src_global) in self._ctrl.items():
            drop = torch.tensor(
                [int(s) not in S for s in src_global.tolist()],
                dtype=torch.bool,
            )
            if not bool(drop.any()):
                continue  # nothing removed for this relation → reuse original
            ei = self.data[etype].edge_index
            remove_cols = cols[drop]
            keep_mask = torch.ones(ei.size(1), dtype=torch.bool)
            keep_mask[remove_cols] = False
            replaced[etype] = ei[:, keep_mask]

        graph = (
            self.data
            if not replaced
            else _shallow_graph_with_edges(self.data, replaced)
        )

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model.forward(graph)
        prob = torch.softmax(logits[self.target_local], dim=-1)[self.fraud_class]
        return float(prob.item())


def make_backbone_coalition_value_fn(
    model: torch.nn.Module,
    data: HeteroData,
    causal_graph: TypedCausalGraph,
    target_node: int,
    type_offsets: Dict[str, int],
    target_node_type: str,
    fraud_class: int = 1,
    intervene_node: Optional[int] = None,
) -> Callable[[frozenset], float]:
    """Convenience factory returning a cached v(S) callable for one target.

    See :class:`BackboneCoalitionValue` for semantics.
    """
    return BackboneCoalitionValue(
        model=model,
        data=data,
        causal_graph=causal_graph,
        target_node=target_node,
        type_offsets=type_offsets,
        target_node_type=target_node_type,
        fraud_class=fraud_class,
        intervene_node=intervene_node,
    )
