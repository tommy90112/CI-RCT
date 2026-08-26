"""
Unit tests for TypedCausalGraph.

Uses a synthetic 4-node heterogeneous graph:
    Nodes: 0 (transaction), 1 (transaction), 2 (actor), 3 (actor)
    Edges: 0→1 (tx__to__tx), 0→2 (tx__to__actor), 1→3 (tx__to__actor)

Edges are directed cause→effect; the graph is a DAG and, when timestamps
are supplied, rejects any edge that would run backwards in time.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.typed_causal_graph import TypedCausalGraph


@pytest.fixture
def simple_graph():
    """4-node typed causal graph for reuse across tests."""
    V = [0, 1, 2, 3]
    node_types = {0: "transaction", 1: "transaction", 2: "actor", 3: "actor"}
    tcg = TypedCausalGraph(V, node_types)
    tcg.add_edge(0, 1, "tx__to__tx")
    tcg.add_edge(0, 2, "tx__to__actor")
    tcg.add_edge(1, 3, "tx__to__actor")
    return tcg


class TestInit:
    def test_node_set(self, simple_graph):
        assert set(simple_graph.v) == {0, 1, 2, 3}
        assert len(simple_graph) == 4

    def test_node_types_stored(self, simple_graph):
        assert simple_graph.node_type[0] == "transaction"
        assert simple_graph.node_type[2] == "actor"

    def test_missing_node_type_raises(self):
        with pytest.raises(ValueError, match="missing entries"):
            TypedCausalGraph([0, 1], node_types={0: "tx"})


class TestAddEdge:
    def test_parent_and_child_sets_updated(self, simple_graph):
        assert simple_graph.parents(1) == {0}
        assert simple_graph.children(0) == {1, 2}

    def test_edges_are_directed(self, simple_graph):
        """0→1 must not imply 1→0."""
        assert simple_graph.parents(0) == set()
        assert 0 not in simple_graph.children(1)

    def test_pa_typed_stores_edge_and_source_type(self, simple_graph):
        etype, ntype = simple_graph.pa_typed[2][0]
        assert etype == "tx__to__actor"
        assert ntype == "transaction"

    def test_unknown_node_rejected(self, simple_graph):
        assert simple_graph.add_edge(0, 99, "tx__to__tx") is False
        assert 99 not in simple_graph.set_v

    def test_temporal_guard_rejects_backwards_edge(self):
        """A cause may not post-date its effect."""
        tcg = TypedCausalGraph(
            [0, 1],
            node_types={0: "transaction", 1: "transaction"},
            timestamps={0: 10, 1: 5},
        )
        assert tcg.add_edge(0, 1, "tx__to__tx") is False
        assert tcg.add_edge(1, 0, "tx__to__tx") is True

    def test_equal_timestamps_allowed(self):
        tcg = TypedCausalGraph(
            [0, 1],
            node_types={0: "transaction", 1: "transaction"},
            timestamps={0: 7, 1: 7},
        )
        assert tcg.add_edge(0, 1, "tx__to__tx") is True


class TestGetEdgeType:
    def test_direct_lookup(self, simple_graph):
        assert simple_graph.get_edge_type(0, 1) == "tx__to__tx"

    def test_reverse_lookup_is_none(self, simple_graph):
        """edge_type_map is keyed by direction — (2, 0) was never added."""
        assert simple_graph.get_edge_type(2, 0) is None

    def test_nonexistent_edge(self, simple_graph):
        assert simple_graph.get_edge_type(2, 3) is None


class TestGetUpstreamNeighbors:
    def test_upstream_of_node_3(self, simple_graph):
        assert simple_graph.get_upstream_neighbors(3) == [1]

    def test_upstream_of_node_0(self, simple_graph):
        """Node 0 is a source — no incoming edges."""
        assert simple_graph.get_upstream_neighbors(0) == []


class TestTypeInformation:
    def test_all_node_types(self, simple_graph):
        assert simple_graph.get_all_node_types() == ["actor", "transaction"]

    def test_all_edge_types(self, simple_graph):
        etypes = simple_graph.get_all_edge_types()
        assert "tx__to__tx" in etypes
        assert "tx__to__actor" in etypes

    def test_node_type_index(self, simple_graph):
        idx = simple_graph.get_node_type_index()
        assert idx["actor"] == 0
        assert idx["transaction"] == 1


class TestTopology:
    def test_source_nodes(self, simple_graph):
        assert simple_graph.source_nodes() == [0]

    def test_topological_order_respects_edges(self, simple_graph):
        order = simple_graph.topological_order()
        pos = {v: i for i, v in enumerate(order)}
        assert len(order) == 4
        for (src, dst) in simple_graph.edge_type_map:
            assert pos[src] < pos[dst]

    def test_topological_index_matches_order(self, simple_graph):
        order = simple_graph.topological_order()
        assert simple_graph.topological_index() == {
            v: i for i, v in enumerate(order)
        }

    def test_cache_invalidated_on_mutation(self, simple_graph):
        simple_graph.topological_order()
        simple_graph.add_edge(2, 3, "actor__to__actor")
        pos = simple_graph.topological_index()
        assert pos[2] < pos[3]
