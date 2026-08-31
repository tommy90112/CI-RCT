r"""
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


def _num_rows(data: HeteroData, ntype: str) -> int:
    store = data[ntype]
    x = getattr(store, "x", None)
    return int(x.size(0)) if x is not None else int(store.num_nodes)


def build_receptive_field_subgraph(
    data: HeteroData,
    target_type: str,
    target_local: int,
    num_layers: int,
):
    """L-hop backward receptive field of one node, as a remapped subgraph.

    A purely-local message-passing backbone (HGTConv: per-node attention over
    neighbours, no global ops) makes a node's L-layer output depend ONLY on the
    nodes within L incoming hops. Forwarding that subgraph reproduces the target
    logit numerically (not approximately) while skipping ~all of a huge graph.

    Returns ``(sub, keep_old, remap, new_target_local)``:
      sub:               HeteroData sharing sliced feature tensors, indices
                         compacted to 0..k per type.
      keep_old:          {ntype: [old_local, ...]} (sub-row i ↔ old keep_old[i]).
      remap:             {ntype: {old_local: new_local}}.
      new_target_local:  target's row in ``sub`` (all None ⇒ not constructible).
    """
    keep = {target_type: {int(target_local)}}
    frontier = {target_type: {int(target_local)}}
    for _ in range(num_layers):
        nxt: Dict[str, set] = {}
        for etype in data.edge_types:
            src_type, _, dst_type = etype
            front = frontier.get(dst_type)
            if not front:
                continue
            ei = data[etype].edge_index
            if ei is None or ei.numel() == 0:
                continue
            front_t = torch.tensor(sorted(front), device=ei.device)
            mask = torch.isin(ei[1], front_t)
            if not bool(mask.any()):
                continue
            seen = keep.setdefault(src_type, set())
            new_src = {int(s) for s in ei[0, mask].tolist()} - seen
            if new_src:
                seen |= new_src
                nxt.setdefault(src_type, set()).update(new_src)
        frontier = nxt
        if not frontier:
            break

    keep_old = {nt: sorted(s) for nt, s in keep.items() if s}
    remap = {nt: {old: new for new, old in enumerate(olds)}
             for nt, olds in keep_old.items()}
    new_target_local = remap.get(target_type, {}).get(int(target_local))
    if new_target_local is None:
        return None, None, None, None

    sub = HeteroData()
    for nt in data.node_types:
        olds = keep_old.get(nt)
        if not olds:
            continue
        x = getattr(data[nt], "x", None)
        if x is not None:
            sub[nt].x = x[torch.tensor(olds, dtype=torch.long, device=x.device)]
        else:
            sub[nt].num_nodes = len(olds)

    for etype in data.edge_types:
        src_type, _, dst_type = etype
        ei = data[etype].edge_index
        rs, rd = remap.get(src_type), remap.get(dst_type)
        if ei is None or ei.numel() == 0 or rs is None or rd is None:
            dev = ei.device if ei is not None else None
            sub[etype].edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
            continue
        map_s = torch.full((_num_rows(data, src_type),), -1, dtype=torch.long, device=ei.device)
        map_d = torch.full((_num_rows(data, dst_type),), -1, dtype=torch.long, device=ei.device)
        for old, new in rs.items():
            map_s[old] = new
        for old, new in rd.items():
            map_d[old] = new
        ns, nd = map_s[ei[0]], map_d[ei[1]]
        m = (ns >= 0) & (nd >= 0)
        sub[etype].edge_index = torch.stack([ns[m], nd[m]], dim=0)

    return sub, keep_old, remap, new_target_local


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
        use_subgraph: bool = False,
        num_layers: Optional[int] = None,
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
        # All bookkeeping index/mask tensors below must live on the same device
        # as the graph's edge_index (= the model/data device). Building them on
        # CPU and indexing a CUDA edge_index (or vice versa) raises a
        # device-mismatch RuntimeError, which only surfaces when running on GPU.
        try:
            self._dev = next(model.parameters()).device
        except StopIteration:
            self._dev = torch.device("cpu")

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

        # ── Optional receptive-field subgraph (large speedup) ─────────────────
        # The backbone is purely local message passing, so the readout's logit is
        # EXACTLY determined by its `num_layers`-hop incoming receptive field.
        # Forwarding only that subgraph per coalition (not the full ~822k-node
        # graph) is numerically identical but far cheaper. The self-check below
        # (subgraph vs full UNINTERVENED target logit — sufficient, since edge
        # removal only ever SHRINKS the receptive field) reverts to the full
        # graph on any mismatch, so this can never silently corrupt v(S).
        # `_row2global[t]`: active-graph local row of type t → its global id.
        active = data
        self._row2global = {
            nt: torch.arange(_num_rows(data, nt), dtype=torch.long, device=self._dev) + type_offsets.get(nt, 0)
            for nt in data.node_types
        }
        if use_subgraph and num_layers:
            sub, keep_old, remap, new_tl = build_receptive_field_subgraph(
                data, target_node_type, self.target_local, num_layers
            )
            new_iv = remap.get(intervene_type, {}).get(intervene_local) if remap else None
            if sub is not None and new_tl is not None and new_iv is not None:
                full_p = self._target_logit(data, self.target_local)
                sub_p = self._target_logit(sub, new_tl)
                if abs(full_p - sub_p) < 1e-4:
                    active = sub
                    self.data = sub
                    self.target_local = new_tl
                    intervene_local = new_iv
                    self._row2global = {
                        nt: torch.tensor(olds, dtype=torch.long, device=self._dev) + type_offsets.get(nt, 0)
                        for nt, olds in keep_old.items()
                    }

        # Pre-index, per affected relation, the columns whose dst is the
        # intervention node AND whose src is a (controllable) parent — on the
        # ACTIVE graph (subgraph or full). ``ctrl`` maps each such relation to
        # (column_indices, src_global aligned to those columns).
        self._ctrl: Dict[EdgeType, Tuple[Tensor, Tensor]] = {}
        for etype in active.edge_types:
            src_type, _, dst_type = etype
            if dst_type != intervene_type:
                continue
            ei = active[etype].edge_index
            if ei is None or ei.numel() == 0:
                continue
            dst_is_target = ei[1] == intervene_local
            if not bool(dst_is_target.any()):
                continue
            cols = torch.nonzero(dst_is_target, as_tuple=False).flatten()
            src_global = self._row2global[src_type][ei[0, cols]]
            # Keep only columns whose source is an actual Shapley parent;
            # non-parent edges into the node are background and stay fixed.
            is_parent = torch.tensor(
                [int(s) in self.parent_set for s in src_global.tolist()],
                dtype=torch.bool, device=self._dev,
            )
            if not bool(is_parent.any()):
                continue
            self._ctrl[etype] = (cols[is_parent], src_global[is_parent])

        self._cache: Dict[frozenset, float] = {}

    def _target_logit(self, graph: HeteroData, target_local: int) -> float:
        """Fraud probability of ``target_local`` under ``graph`` (no intervention)."""
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model.forward(graph)
        return float(torch.softmax(logits[target_local], dim=-1)[self.fraud_class].item())

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
                dtype=torch.bool, device=self._dev,
            )
            if not bool(drop.any()):
                continue  # nothing removed for this relation → reuse original
            ei = self.data[etype].edge_index
            remove_cols = cols[drop]
            keep_mask = torch.ones(ei.size(1), dtype=torch.bool, device=self._dev)
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
    use_subgraph: bool = False,
    num_layers: Optional[int] = None,
) -> Callable[[frozenset], float]:
    """Convenience factory returning a cached v(S) callable for one target.

    See :class:`BackboneCoalitionValue` for semantics. ``use_subgraph`` + the
    backbone's ``num_layers`` enable the self-verified receptive-field speedup.
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
        use_subgraph=use_subgraph,
        num_layers=num_layers,
    )
