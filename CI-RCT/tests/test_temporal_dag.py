"""
Unit tests for the follow-the-money temporal DAG.

Covers the three wired-together pieces:
  1. TypedCausalGraph.add_edge temporal guard  (> reject, == keep)
  2. utils.data_utils.build_global_timestamps   (.time → global dict, sentinel skip)
  3. build_typed_causal_graph_from_hetero       (strictly-backward edges dropped)

All synthetic — no dataset download. Global ID scheme (sorted types):
    'transaction' offset 0   (tx0=0, tx1=1, tx2=2)
    'wallet'      offset 3   (w0=3, w1=4)
"""
import os
import sys

import pytest
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.typed_causal_graph import TypedCausalGraph
from utils.data_utils import (
    build_global_timestamps,
    build_typed_causal_graph_from_hetero,
    compute_type_offsets,
)


# ── 1. TypedCausalGraph guard ───────────────────────────────────────────────

class TestTemporalGuard:
    def _graph(self):
        return TypedCausalGraph(
            V=[0, 1, 2],
            node_types={0: "transaction", 1: "transaction", 2: "transaction"},
            timestamps={0: 1, 1: 1, 2: 3},
        )

    def test_forward_edge_kept(self):
        g = self._graph()
        assert g.add_edge(0, 2, "tx__to__tx") is True   # t1 → t3
        assert 0 in g.parents(2)

    def test_equal_timestamp_edge_kept(self):
        g = self._graph()
        assert g.add_edge(0, 1, "tx__to__tx") is True   # t1 → t1 (same step)
        assert 0 in g.parents(1)

    def test_strictly_backward_edge_rejected(self):
        g = self._graph()
        assert g.add_edge(2, 0, "tx__to__tx") is False  # t3 → t1
        assert 2 not in g.parents(0)

    def test_missing_timestamp_is_unconstrained(self):
        g = TypedCausalGraph(
            V=[0, 1],
            node_types={0: "wallet", 1: "transaction"},
            timestamps={1: 1},   # node 0 has no timestamp
        )
        # Guard only fires when BOTH endpoints are timed → this passes.
        assert g.add_edge(0, 1, "wallet__to__tx") is True


# ── 2. build_global_timestamps ──────────────────────────────────────────────

@pytest.fixture
def timed_hetero():
    """tx time = [1, 1, 3]; wallet time = [2, -1] (w1 = sentinel)."""
    data = HeteroData()
    data["transaction"].x = torch.randn(3, 4)
    data["transaction"].time = torch.tensor([1, 1, 3], dtype=torch.long)
    data["wallet"].x = torch.randn(2, 5)
    data["wallet"].time = torch.tensor([2, -1], dtype=torch.long)
    # tx→tx: 0→1 (eq), 0→2 (fwd), 2→0 (back)
    data["transaction", "flows_to", "transaction"].edge_index = torch.tensor(
        [[0, 0, 2], [1, 2, 0]], dtype=torch.long
    )
    # wallet→tx: w0→tx2 (fwd t2→t3), w0→tx0 (back t2→t1), w1→tx0 (unconstrained)
    data["wallet", "sends", "transaction"].edge_index = torch.tensor(
        [[0, 0, 1], [2, 0, 0]], dtype=torch.long
    )
    return data


class TestBuildGlobalTimestamps:
    def test_sentinel_skipped_and_offsets_applied(self, timed_hetero):
        offsets = compute_type_offsets(timed_hetero)
        ts = build_global_timestamps(timed_hetero, offsets)
        # transaction offset 0, wallet offset 3
        assert ts == {0: 1, 1: 1, 2: 3, 3: 2}   # w1 (global 4) dropped (-1)
        assert 4 not in ts

    def test_empty_when_no_time(self):
        data = HeteroData()
        data["transaction"].x = torch.randn(2, 3)
        data["transaction", "flows_to", "transaction"].edge_index = torch.tensor(
            [[0], [1]], dtype=torch.long
        )
        offsets = compute_type_offsets(data)
        assert build_global_timestamps(data, offsets) == {}


# ── 3. End-to-end DAG construction ──────────────────────────────────────────

class TestTimeRespectingDAG:
    def _edges(self, timed_hetero):
        tcg = build_typed_causal_graph_from_hetero(timed_hetero, node_limit=100)
        return tcg, set(tcg.edge_type_map.keys())

    def test_forward_and_equal_edges_present(self, timed_hetero):
        _, edges = self._edges(timed_hetero)
        assert (0, 1) in edges   # tx eq
        assert (0, 2) in edges   # tx fwd
        assert (3, 2) in edges   # wallet0 → tx2 fwd

    def test_backward_edges_dropped(self, timed_hetero):
        _, edges = self._edges(timed_hetero)
        assert (2, 0) not in edges   # tx t3 → t1
        assert (3, 0) not in edges   # wallet0 t2 → tx0 t1

    def test_untimed_source_edge_kept(self, timed_hetero):
        _, edges = self._edges(timed_hetero)
        assert (4, 0) in edges   # wallet1 (no time) → tx0

    def test_topological_order_is_acyclic(self, timed_hetero):
        tcg, _ = self._edges(timed_hetero)
        order = tcg.topological_order()
        # No cycle dropped → every node appears exactly once.
        assert len(order) == len(tcg.v)
        pos = {n: i for i, n in enumerate(order)}
        for (src, dst) in tcg.edge_type_map:
            assert pos[src] < pos[dst]   # parent precedes child

    def test_legacy_no_time_keeps_all_forward_edges(self):
        data = HeteroData()
        data["transaction"].x = torch.randn(3, 4)
        data["transaction", "flows_to", "transaction"].edge_index = torch.tensor(
            [[0, 2], [1, 0]], dtype=torch.long   # 0→1, 2→0 (would be backward)
        )
        tcg = build_typed_causal_graph_from_hetero(data, node_limit=100)
        edges = set(tcg.edge_type_map.keys())
        assert (0, 1) in edges
        assert (2, 0) in edges   # no timestamps → not rejected
