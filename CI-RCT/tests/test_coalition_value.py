"""
Tests for backbone do-intervention coalition value (model/coalition_value.py).

Verifies on a small real HGT backbone that:
  * v(full parent set) reproduces the unintervened target fraud probability;
  * cutting parent edges (smaller S) actually changes the value;
  * v is non-additive: v({a,b}) + v(∅) ≠ v({a}) + v({b}) — the property that
    lets asymmetric Shapley diverge from symmetric;
  * the original `data` is never mutated;
  * results are cached.
"""
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.coalition_value import (  # noqa: E402
    BackboneCoalitionValue,
    build_receptive_field_subgraph,
    make_backbone_coalition_value_fn,
)
from model.hetero_backbone import HeteroGNNBackbone  # noqa: E402
from model.typed_causal_graph import TypedCausalGraph  # noqa: E402
from utils.data_utils import compute_type_offsets  # noqa: E402


def _build_graph_and_model():
    torch.manual_seed(0)
    hd = HeteroData()
    hd["transaction"].x = torch.randn(2, 5)
    hd["wallet"].x = torch.randn(3, 4)
    # wallet→transaction edges. type_offsets (sorted): transaction=0, wallet=2.
    # Give target tx local-1 two wallet parents (wallet local 0,1 → global 2,3);
    # plus a background edge into tx local-0 so the graph isn't degenerate.
    hd["wallet", "to", "transaction"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 1, 0]], dtype=torch.long,
    )
    model = HeteroGNNBackbone(
        metadata=hd.metadata(),
        in_channels_dict={"transaction": 5, "wallet": 4},
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        target_node_type="transaction",
    )
    model.eval()
    return hd, model


def _build_causal_graph():
    # Global ids: transaction {0,1}, wallet {2,3,4}. Target = tx global 1.
    node_types = {0: "transaction", 1: "transaction",
                  2: "wallet", 3: "wallet", 4: "wallet"}
    g = TypedCausalGraph(V=[0, 1, 2, 3, 4], node_types=node_types)
    g.add_edge(2, 1, "wallet__to__transaction")
    g.add_edge(3, 1, "wallet__to__transaction")
    return g


def test_full_coalition_matches_unintervened_prob():
    hd, model = _build_graph_and_model()
    g = _build_causal_graph()
    offsets = compute_type_offsets(hd)
    target = 1  # tx global 1

    v = BackboneCoalitionValue(
        model, hd, g, target, offsets, target_node_type="transaction"
    )
    # v(all parents) keeps every parent edge → identical to a plain forward.
    full = v(frozenset({2, 3}))

    with torch.no_grad():
        logits, _ = model.forward(hd)
    expected = torch.softmax(logits[1], dim=-1)[1].item()
    assert abs(full - expected) < 1e-6


def test_cutting_parents_changes_value():
    hd, model = _build_graph_and_model()
    g = _build_causal_graph()
    offsets = compute_type_offsets(hd)
    v = make_backbone_coalition_value_fn(
        model, hd, g, 1, offsets, target_node_type="transaction"
    )
    full = v(frozenset({2, 3}))
    empty = v(frozenset())
    # Removing both parent edges should move the prediction.
    assert abs(full - empty) > 1e-7


def test_value_is_non_additive():
    hd, model = _build_graph_and_model()
    g = _build_causal_graph()
    offsets = compute_type_offsets(hd)
    v = BackboneCoalitionValue(
        model, hd, g, 1, offsets, target_node_type="transaction"
    )
    v_empty = v(frozenset())
    v_a = v(frozenset({2}))
    v_b = v(frozenset({3}))
    v_ab = v(frozenset({2, 3}))
    # Additive would mean v_ab - v_empty == (v_a - v_empty) + (v_b - v_empty),
    # i.e. v_ab + v_empty == v_a + v_b. HGT aggregation breaks this.
    interaction = (v_ab + v_empty) - (v_a + v_b)
    assert abs(interaction) > 1e-7


def test_original_data_not_mutated():
    hd, model = _build_graph_and_model()
    g = _build_causal_graph()
    offsets = compute_type_offsets(hd)
    before = hd["wallet", "to", "transaction"].edge_index.clone()
    v = BackboneCoalitionValue(
        model, hd, g, 1, offsets, target_node_type="transaction"
    )
    v(frozenset())
    v(frozenset({2}))
    after = hd["wallet", "to", "transaction"].edge_index
    assert torch.equal(before, after)


def test_cache_returns_consistent_value():
    hd, model = _build_graph_and_model()
    g = _build_causal_graph()
    offsets = compute_type_offsets(hd)
    v = BackboneCoalitionValue(
        model, hd, g, 1, offsets, target_node_type="transaction"
    )
    first = v(frozenset({2}))
    second = v(frozenset({2}))
    assert first == second
    assert frozenset({2}) in v._cache


# ── Receptive-field subgraph speedup: must be NUMERICALLY IDENTICAL ────────────

def _build_multihop_graph_and_model():
    """tx0's 2-hop receptive field excludes far nodes (tx3, w2-w5), so the
    subgraph is a STRICT subset — a real test that pruning preserves the logit."""
    torch.manual_seed(0)
    hd = HeteroData()
    hd["transaction"].x = torch.randn(4, 5)
    hd["wallet"].x = torch.randn(6, 4)
    hd["wallet", "to", "transaction"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 3, 3]], dtype=torch.long)
    hd["transaction", "rev", "wallet"].edge_index = torch.tensor(
        [[1, 2, 3, 3], [0, 1, 4, 5]], dtype=torch.long)
    model = HeteroGNNBackbone(
        metadata=hd.metadata(),
        in_channels_dict={"transaction": 5, "wallet": 4},
        hidden_dim=16, num_heads=2, num_layers=2,
        target_node_type="transaction")
    model.eval()
    g = TypedCausalGraph(
        V=list(range(10)),
        node_types={**{i: "transaction" for i in range(4)},
                    **{i: "wallet" for i in range(4, 10)}})
    g.add_edge(4, 0, "wallet__to__transaction")
    g.add_edge(5, 0, "wallet__to__transaction")
    return hd, model, g


def test_receptive_field_is_strict_subset():
    hd, _, _ = _build_multihop_graph_and_model()
    _, keep_old, _, new_tl = build_receptive_field_subgraph(hd, "transaction", 0, 2)
    assert keep_old["transaction"] == [0, 1, 2]   # tx3 excluded
    assert keep_old["wallet"] == [0, 1]           # w2..w5 excluded
    assert new_tl == 0


def test_subgraph_forward_matches_full():
    hd, model, _ = _build_multihop_graph_and_model()
    sub, _, _, new_tl = build_receptive_field_subgraph(hd, "transaction", 0, 2)
    with torch.no_grad():
        full = torch.softmax(model.forward(hd)[0][0], -1)[1]
        subp = torch.softmax(model.forward(sub)[0][new_tl], -1)[1]
    assert torch.allclose(full, subp, atol=1e-6)


def test_coalition_value_subgraph_matches_full():
    hd, model, g = _build_multihop_graph_and_model()
    offs = {"transaction": 0, "wallet": 4}
    v_full = BackboneCoalitionValue(model, hd, g, 0, offs, "transaction",
                                    use_subgraph=False)
    v_sub = BackboneCoalitionValue(model, hd, g, 0, offs, "transaction",
                                   use_subgraph=True, num_layers=2)
    for S in [frozenset(), frozenset({4}), frozenset({5}), frozenset({4, 5})]:
        assert abs(v_full(S) - v_sub(S)) < 1e-6
