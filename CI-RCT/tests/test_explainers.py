"""
Smoke + behaviour tests for the explainer registry (model/explainers.py).

Uses a small real HGT backbone so the φ explainers exercise the actual
backbone do-intervention coalition value.  cxgnn_ncm is covered separately
(test_cxgnn_ncm_adapter.py).
"""
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.explainers import build_explainer  # noqa: E402
from model.hetero_backbone import HeteroGNNBackbone  # noqa: E402
from model.root_cause_tracer import RootCauseTracer  # noqa: E402
from model.typed_causal_graph import TypedCausalGraph  # noqa: E402
from utils.data_utils import compute_type_offsets  # noqa: E402


def _fixture():
    torch.manual_seed(1)
    hd = HeteroData()
    hd["transaction"].x = torch.randn(2, 5)
    hd["wallet"].x = torch.randn(3, 4)
    # tx local-1 has wallet parents (global 2,3); tx local-1 also feeds tx? no.
    hd["wallet", "to", "transaction"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 1, 0]], dtype=torch.long,
    )
    model = HeteroGNNBackbone(
        metadata=hd.metadata(),
        in_channels_dict={"transaction": 5, "wallet": 4},
        hidden_dim=16, num_heads=2, num_layers=2,
        target_node_type="transaction",
    )
    model.eval()
    # Backbone is the model; explainers only need .forward and .backbone.* — but
    # ce_only/phi use model.forward directly, and build_explainer reads
    # model.backbone.target_node_type only inside evaluate.py, not here.
    g = TypedCausalGraph(
        V=[0, 1, 2, 3, 4],
        node_types={0: "transaction", 1: "transaction",
                    2: "wallet", 3: "wallet", 4: "wallet"},
    )
    g.add_edge(2, 1, "wallet__to__transaction")
    g.add_edge(3, 1, "wallet__to__transaction")
    offsets = compute_type_offsets(hd)
    tracer = RootCauseTracer(g, max_hops=3, threshold=0.0)
    return model, hd, g, offsets, tracer


@pytest.mark.parametrize("name", ["ce_only", "phi_asym", "phi_sym", "saliency"])
def test_explainer_returns_set_including_target(name):
    model, hd, g, offsets, tracer = _fixture()
    explain = build_explainer(
        name, model=model, data=hd, causal_graph=g, tracer=tracer,
        type_offsets=offsets, target_node_type="transaction",
        n_permutations=8,
    )
    # ce_only ranks by these CE; φ variants ignore them (use coalition value).
    causal_effects = {(2, 1): 0.6, (3, 1): 0.4}
    out = explain(1, causal_effects)
    assert isinstance(out, set)
    assert 1 in out  # the queried target is always part of its own explanation


def test_unknown_explainer_raises():
    model, hd, g, offsets, tracer = _fixture()
    with pytest.raises(ValueError, match="Unknown explainer"):
        build_explainer(
            "nope", model=model, data=hd, causal_graph=g, tracer=tracer,
            type_offsets=offsets, target_node_type="transaction",
        )


def test_ce_only_matches_raw_tracer():
    """ce_only must reproduce the legacy raw-|CE| trace exactly."""
    model, hd, g, offsets, tracer = _fixture()
    causal_effects = {(2, 1): 0.6, (3, 1): 0.4}
    explain = build_explainer(
        "ce_only", model=model, data=hd, causal_graph=g, tracer=tracer,
        type_offsets=offsets, target_node_type="transaction",
    )
    out = explain(1, causal_effects)
    _, chain = tracer.trace_root_cause(1, causal_effects)
    assert out == set(chain)


def test_saliency_scorer_is_non_causal_and_finite():
    """The saliency arm ranks parents by Grad×Input, ignoring the CE dict.

    It must return finite, non-negative scores for the in-receptive-field
    parents and 0.0 for a node outside the field — the same 'unreachable ⇒ 0'
    semantics as the φ arms — proving the ranking comes from the gradient, not
    the (here unused) causal_effects.
    """
    from model.saliency_explainer import make_saliency_score_fn

    model, hd, g, offsets, tracer = _fixture()
    score_fn = make_saliency_score_fn(
        model=model, data=hd, causal_graph=g, target_node=1,
        type_offsets=offsets, target_node_type="transaction",
        fraud_class=1, use_subgraph=False,
    )
    scores = score_fn(1, [2, 3, 4])  # 2,3 are tx-1's wallet parents; 4 is not
    assert set(scores) == {2, 3, 4}
    for u in (2, 3):
        assert scores[u] >= 0.0 and scores[u] == scores[u]  # finite, non-neg
    assert scores[4] == 0.0  # node 4 never feeds tx-1 → zero saliency
