"""
Smoke test for the CXGNN-NCM baseline adapter (model/cxgnn_ncm_adapter.py).

Skips cleanly if CXGNN's optional deps (matplotlib/networkx) are unavailable.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.hetero_backbone import HeteroGNNBackbone  # noqa: E402
from model.typed_causal_graph import TypedCausalGraph  # noqa: E402
from utils.data_utils import compute_type_offsets  # noqa: E402

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")


def _fixture():
    torch.manual_seed(2)
    hd = HeteroData()
    hd["transaction"].x = torch.randn(2, 5)
    hd["wallet"].x = torch.randn(3, 4)
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
    g = TypedCausalGraph(
        V=[0, 1, 2, 3, 4],
        node_types={0: "transaction", 1: "transaction",
                    2: "wallet", 3: "wallet", 4: "wallet"},
    )
    g.add_edge(2, 1, "wallet__to__transaction")
    g.add_edge(3, 1, "wallet__to__transaction")
    return model, hd, g, compute_type_offsets(hd)


def test_cxgnn_ncm_explainer_returns_set_with_target():
    from model.cxgnn_ncm_adapter import build_cxgnn_ncm_explainer

    model, hd, g, offsets = _fixture()
    explain = build_cxgnn_ncm_explainer(
        model=model, data=hd, causal_graph=g, type_offsets=offsets,
        target_node_type="transaction", max_nodes=8, num_epochs=3,
    )
    out = explain(1, causal_effects={})
    assert isinstance(out, set)
    assert 1 in out
    # parents are reachable within the 2-hop window → explanation is non-trivial
    assert out <= {1, 2, 3}


def test_isolated_target_falls_back_to_singleton():
    from model.cxgnn_ncm_adapter import build_cxgnn_ncm_explainer

    model, hd, g, offsets = _fixture()
    explain = build_cxgnn_ncm_explainer(
        model=model, data=hd, causal_graph=g, type_offsets=offsets,
        target_node_type="transaction", max_nodes=8, num_epochs=3,
    )
    # Node 0 (tx) has no parents in g → singleton explanation.
    out = explain(0, causal_effects={})
    assert out == {0}
