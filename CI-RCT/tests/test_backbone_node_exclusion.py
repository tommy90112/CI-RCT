"""
Unit tests for DD-8 Fix 4: HeteroGNNBackbone node-type exclusion.

Verifies that when `exclude_node_types=["host_node"]` is passed:
  - HGT layers and input projections are NOT constructed for host_node.
  - Edges touching host_node are filtered out of message passing.
  - forward() runs successfully on a graph that still contains host_node
    (so RootCauseTracer can use it downstream).
  - target_node_type cannot be excluded.
"""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from model.hetero_backbone import HeteroGNNBackbone


def _build_toy_hetero() -> HeteroData:
    """A 5-node-type toy graph matching DD-9 MG24 kill-chain DAG layout."""
    hd = HeteroData()
    hd["flow_node"].x = torch.randn(10, 8)
    hd["host_node"].x = torch.randn(3, 6)
    hd["process_node"].x = torch.randn(5, 4)
    hd["device_node"].x = torch.randn(2, 2)
    hd["measurement_node"].x = torch.randn(2, 3)
    # Edges follow DD-9 kill-chain order: flow → host → process → device → measurement
    # (no host_sources_flow; eliminates the 2-cycle with flow_targets_host).
    hd["flow_node", "targets", "host_node"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long,
    )
    hd["host_node", "runs", "process_node"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long,
    )
    hd["process_node", "forks", "process_node"].edge_index = torch.tensor(
        [[0], [1]], dtype=torch.long,
    )
    hd["process_node", "controls", "device_node"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long,
    )
    hd["device_node", "reports", "measurement_node"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long,
    )
    return hd


def test_excluded_type_dropped_from_input_proj():
    hd = _build_toy_hetero()
    backbone = HeteroGNNBackbone(
        metadata=hd.metadata(),
        in_channels_dict={"flow_node": 8, "process_node": 4,
                          "host_node": 6, "device_node": 2, "measurement_node": 3},
        hidden_dim=16,
        target_node_type="flow_node",
        exclude_node_types=["host_node"],
    )
    assert "host_node" not in backbone.input_proj
    assert "host_node" not in backbone.node_types
    assert backbone.excluded_types == ["host_node"]


def test_excluded_type_dropped_from_edge_types():
    hd = _build_toy_hetero()
    backbone = HeteroGNNBackbone(
        metadata=hd.metadata(),
        in_channels_dict={"flow_node": 8, "process_node": 4,
                          "host_node": 6, "device_node": 2, "measurement_node": 3},
        hidden_dim=16,
        target_node_type="flow_node",
        exclude_node_types=["host_node"],
    )
    for s, r, d in backbone.edge_types:
        assert s != "host_node", f"edge {(s, r, d)} should be excluded"
        assert d != "host_node", f"edge {(s, r, d)} should be excluded"


def test_forward_runs_with_host_excluded():
    """Even though HeteroData has host_node, forward should skip it cleanly."""
    hd = _build_toy_hetero()
    backbone = HeteroGNNBackbone(
        metadata=hd.metadata(),
        in_channels_dict={"flow_node": 8, "process_node": 4,
                          "host_node": 6, "device_node": 2, "measurement_node": 3},
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        target_node_type="flow_node",
        exclude_node_types=["host_node"],
    )
    logits, h_dict = backbone(hd)
    # Target logits returned for flow_node
    assert logits.shape == (10, 2)
    # h_dict must NOT contain host_node embeddings
    assert "host_node" not in h_dict


def test_target_node_in_exclude_raises():
    hd = _build_toy_hetero()
    with pytest.raises(ValueError, match="target_node_type"):
        HeteroGNNBackbone(
            metadata=hd.metadata(),
            in_channels_dict={"flow_node": 8, "process_node": 4,
                              "host_node": 6, "device_node": 2, "measurement_node": 3},
            hidden_dim=16,
            target_node_type="host_node",  # ⚠ excluded
            exclude_node_types=["host_node"],
        )


def test_exclude_all_types_raises():
    hd = _build_toy_hetero()
    all_types = ["flow_node", "process_node", "host_node",
                 "device_node", "measurement_node"]
    with pytest.raises(ValueError, match="All node types excluded"):
        HeteroGNNBackbone(
            metadata=hd.metadata(),
            in_channels_dict={t: 4 for t in all_types},
            hidden_dim=16,
            target_node_type="flow_node",
            exclude_node_types=all_types,
        )


def test_default_behaviour_unchanged():
    """Without exclude_node_types, backbone should behave exactly as before."""
    hd = _build_toy_hetero()
    backbone = HeteroGNNBackbone(
        metadata=hd.metadata(),
        in_channels_dict={"flow_node": 8, "process_node": 4,
                          "host_node": 6, "device_node": 2, "measurement_node": 3},
        hidden_dim=16,
        target_node_type="flow_node",
    )
    assert "host_node" in backbone.input_proj
    assert "host_node" in backbone.node_types
    assert backbone.excluded_types == []
    logits, h_dict = backbone(hd)
    assert logits.shape == (10, 2)
    # host_node should appear in h_dict since not excluded
    assert "host_node" in h_dict
