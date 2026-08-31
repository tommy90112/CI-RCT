"""
CXGNN-NCM adapter — external SOTA baseline for the Metric-C explainer ablation.

Wraps CXGNN's GNN-NCM (ECCV 2024; CXGNN/model/alg1.py + causal.py) so it can
produce a per-fraud-node explanatory set scored by the same Metric C as the
CI-RCT explainers.  This is the apples-to-apples external comparator in route A.

Why an adapter is needed
------------------------
CXGNN was built for graph classification on small homogeneous graphs with a
clean motif ground-truth: it trains one NCM per node over the node's 1-/2-hop
neighbourhood and treats the underlying subgraph as the explanation.  Our task
is node-anchored fraud explanation on a large heterogeneous temporal graph, so
the adapter:

  1. extracts each query fraud node's backward k-hop causal neighbourhood from
     the TypedCausalGraph (capped at ``max_nodes`` — CXGNN's
     ``compute_probability_of_node_label`` enumerates labels^n_nodes, so the
     cap keeps it tractable; truncation keeps the nodes closest to the target
     and is logged);
  2. relabels the subgraph to dense local ids and builds a CXGNN
     ``CausalGraph(V, path)`` (undirected first-neighbourhood, as CXGNN uses);
  3. supplies per-node labels from the BACKBONE's own predictions (not the LFPN
     ground-truth — that would leak the metric), so CXGNN-NCM and the CI-RCT
     explainers both build on the identical backbone;
  4. trains the NCM anchored on the query node (CXGNN alg1.train) and returns
     ``{target} ∪ one_hop_neighbours`` mapped back to global ids — CXGNN's
     ``new_v`` explanatory set.

Faithfulness note: we anchor on the queried fraud node (alg1.train with
target=query) rather than alg2's whole-graph best-node search, because Metric C
is node-anchored; alg2 would re-centre the explanation on a different node.
"""
from typing import Callable, Dict, List, Set, Tuple

import torch

from _cxgnn_path import register_cxgnn_path

register_cxgnn_path()


def _backward_khop(causal_graph, target: int, k: int, max_nodes: int) -> List[int]:
    """Backward BFS up to k hops from target; cap at max_nodes (closest first).

    Returns global node ids including target.  When the neighbourhood exceeds
    max_nodes, nearer hops are kept preferentially (BFS order) and the overflow
    is dropped — callers should log the truncation.
    """
    order = [target]
    seen = {target}
    frontier = [target]
    for _ in range(k):
        nxt = []
        for v in frontier:
            for p in causal_graph.get_upstream_neighbors(v):
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
                    order.append(p)
                    if len(order) >= max_nodes:
                        return order
        frontier = nxt
        if not frontier:
            break
    return order


def build_cxgnn_ncm_explainer(
    *,
    model,
    data,
    causal_graph,
    type_offsets: Dict[str, int],
    target_node_type: str,
    fraud_class: int = 1,
    max_nodes: int = 8,
    khop: int = 2,
    num_epochs: int = 10,
    learning_rate: float = 0.005,
    h_size: int = 32,
    h_layers: int = 2,
) -> Callable[[int, Dict[Tuple[int, int], float]], Set[int]]:
    """Return an ``explain(target, causal_effects) -> set[int]`` backed by CXGNN-NCM.

    See module docstring for the adaptation rationale and the ``max_nodes`` cap.
    """
    import pandas as pd
    from causal import CausalGraph  # CXGNN (registered on import)
    import alg1  # CXGNN

    # Backbone predictions → per-node binary "role" labels (target type only;
    # other types default to 0). Computed once and shared across queries.
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward(data)
    target_pred = logits.argmax(dim=-1)  # [N_target_type]
    target_off = type_offsets[target_node_type]

    def _node_label(gid: int) -> int:
        ntype = causal_graph.node_type.get(gid)
        if ntype == target_node_type:
            local = gid - target_off
            if 0 <= local < target_pred.size(0):
                return int(target_pred[local].item())
        return 0

    _truncated = {"count": 0}

    def explain(target: int, causal_effects) -> Set[int]:
        globals_in = _backward_khop(causal_graph, target, khop, max_nodes)
        if len(globals_in) >= max_nodes:
            _truncated["count"] += 1
        # Dense local relabelling; target gets a stable local id.
        g2l = {g: i for i, g in enumerate(globals_in)}
        l2g = {i: g for g, i in g2l.items()}
        target_local = g2l[target]

        # Undirected first-neighbourhood edges within the subgraph.
        path: List[List[int]] = []
        for g in globals_in:
            for p in causal_graph.get_upstream_neighbors(g):
                if p in g2l:
                    path.append([g2l[p], g2l[g]])

        cg = CausalGraph(V=list(range(len(globals_in))), path=path)

        # Per-node labels from backbone predictions; force the queried node = 1.
        labels = {i: _node_label(l2g[i]) for i in range(len(globals_in))}
        labels[target_local] = 1
        role_id = [labels[i] for i in range(len(globals_in))]
        df = pd.DataFrame({"node_label": role_id})

        # A node with no neighbours has an empty NCM input → CXGNN can't train;
        # fall back to the singleton explanation. categorize_neighbors returns
        # the tiers as a tuple (alg1.train unpacks them onto cg attributes).
        cat = cg.categorize_neighbors(target_local)
        if cat is None:
            return {target}
        _, one_hop, two_hop, _ = cat
        if not (one_hop or two_hop):
            return {target}

        _, _, _, _, _, new_v = alg1.train(
            cg, learning_rate, h_size, h_layers, num_epochs,
            df, role_id, target_local,
        )
        explained = {l2g[v] for v in new_v if v in l2g}
        explained.add(target)
        return explained

    explain.truncated_counter = _truncated  # exposed for caller-side logging
    return explain
