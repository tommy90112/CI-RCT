"""
Unit tests for utils/data_utils.py.

Uses a minimal synthetic HeteroData graph to avoid dataset download.
"""
import sys
import os

import pytest
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_utils import (
    compute_type_offsets,
    build_typed_causal_graph_from_hetero,
    heterodata_to_flat_feature_dict,
)


@pytest.fixture
def minimal_hetero():
    """
    Synthetic HeteroData:
        Nodes: 3 'transaction' nodes, 2 'actor' nodes
        Edges: transaction → actor (3 edges)
    """
    data = HeteroData()
    data["transaction"].x = torch.randn(3, 4)
    data["transaction"].num_nodes = 3
    data["actor"].x = torch.randn(2, 8)
    data["actor"].num_nodes = 2
    # 3 directed edges: tx[0]→act[0], tx[1]→act[1], tx[2]→act[0]
    data["transaction", "sends", "actor"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 0]], dtype=torch.long
    )
    return data


class TestComputeTypeOffsets:
    def test_sorted_order(self, minimal_hetero):
        offsets = compute_type_offsets(minimal_hetero)
        # sorted: ['actor', 'transaction']
        assert offsets["actor"] == 0
        assert offsets["transaction"] == 2  # 2 actor nodes come first

    def test_non_overlapping(self, minimal_hetero):
        offsets = compute_type_offsets(minimal_hetero)
        sizes = {nt: minimal_hetero[nt].num_nodes for nt in minimal_hetero.node_types}
        end_offsets = {nt: offsets[nt] + sizes[nt] for nt in offsets}
        starts = sorted(offsets.values())
        ends = sorted(end_offsets.values())
        # No overlap: each start >= previous end
        for i in range(1, len(starts)):
            assert starts[i] >= ends[i - 1]


class TestBuildTypedCausalGraph:
    def test_node_count(self, minimal_hetero):
        tcg = build_typed_causal_graph_from_hetero(minimal_hetero, node_limit=100)
        assert len(tcg.v) == 5  # 3 tx + 2 actor

    def test_node_types_assigned(self, minimal_hetero):
        tcg = build_typed_causal_graph_from_hetero(minimal_hetero, node_limit=100)
        types = set(tcg.node_type.values())
        assert "transaction" in types
        assert "actor" in types

    def test_edges_present(self, minimal_hetero):
        tcg = build_typed_causal_graph_from_hetero(minimal_hetero, node_limit=100, directed=True)
        assert len(tcg.edge_type_map) > 0

    def test_node_limit_respected(self, minimal_hetero):
        tcg = build_typed_causal_graph_from_hetero(minimal_hetero, node_limit=3)
        assert len(tcg.v) <= 3

    def test_bfs_from_target(self, minimal_hetero):
        offsets = compute_type_offsets(minimal_hetero)
        # Start BFS from global ID of transaction node 0
        target = offsets["transaction"] + 0
        tcg = build_typed_causal_graph_from_hetero(
            minimal_hetero, target_node_id=target, hop_limit=1, node_limit=100
        )
        # At least the target node should be in the graph
        assert target in set(tcg.v)

    def test_empty_node_types_raises(self):
        data = HeteroData()
        with pytest.raises((ValueError, AttributeError)):
            build_typed_causal_graph_from_hetero(data)


class TestHeterodataToFlatFeatureDict:
    def test_all_nodes_present(self, minimal_hetero):
        offsets = compute_type_offsets(minimal_hetero)
        flat = heterodata_to_flat_feature_dict(minimal_hetero, offsets)
        assert len(flat) == 5  # 3 tx + 2 actor

    def test_feature_dim_correct(self, minimal_hetero):
        offsets = compute_type_offsets(minimal_hetero)
        flat = heterodata_to_flat_feature_dict(minimal_hetero, offsets)
        # actor features are dim 8, transaction features are dim 4
        actor_start = offsets["actor"]
        assert flat[actor_start].shape == torch.Size([8])
        tx_start = offsets["transaction"]
        assert flat[tx_start].shape == torch.Size([4])

    def test_no_mutation(self, minimal_hetero):
        """Modifying flat dict values should not change original data."""
        offsets = compute_type_offsets(minimal_hetero)
        flat = heterodata_to_flat_feature_dict(minimal_hetero, offsets)
        original_val = flat[offsets["transaction"]].clone()
        flat[offsets["transaction"]] = torch.zeros(4)
        assert torch.allclose(
            minimal_hetero["transaction"].x[0], original_val
        )
