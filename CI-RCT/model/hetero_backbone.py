"""
HeteroGNNBackbone — Module 1 of CI-RCT.

Heterogeneous Graph Transformer (HGT) backbone for node classification
on heterogeneous graphs.  Dataset-agnostic: accepts any HeteroData
structure and returns per-type node embeddings plus target-type logits.

Reference: CI-RCT_Thesis_Plan.md § 5.2
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


class HeteroGNNBackbone(nn.Module):
    """
    HGT-based backbone for heterogeneous node classification.

    Architecture:
        Input projection (per node type) → HGT layers × num_layers → Classifier

    Args:
        metadata:          HeteroData.metadata() tuple (node_types, edge_types)
        in_channels_dict:  {node_type: feature_dim} or use -1 for lazy initialisation
        hidden_dim:        Hidden and output embedding dimension
        num_classes:       Number of classification classes for target_node_type
        num_heads:         Attention heads in each HGTConv layer
        num_layers:        Number of HGT message-passing layers
        target_node_type:  The node type whose logits are returned for classification
        dropout:           Dropout probability applied between HGT layers
    """

    def __init__(
        self,
        metadata: tuple,
        in_channels_dict: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        target_node_type: Optional[str] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.target_node_type = target_node_type or self.node_types[0]

        # Per-type input projection: maps raw features → hidden_dim
        # Use in_channels=-1 for lazy (deferred) initialisation if sizes unknown.
        in_ch_map = in_channels_dict or {ntype: -1 for ntype in self.node_types}
        self.input_proj = nn.ModuleDict(
            {ntype: Linear(in_ch_map.get(ntype, -1), hidden_dim) for ntype in self.node_types}
        )

        # HGT message-passing layers
        self.hgt_layers = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Classification head — raw logits, no sigmoid
        self.classifier = nn.Linear(hidden_dim, num_classes)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, data: HeteroData) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Run HGT forward pass on a HeteroData graph.

        Args:
            data: PyG HeteroData object

        Returns:
            logits:  Tensor [N_target, num_classes] — raw logits for target node type
            h_dict:  {node_type: Tensor [N, hidden_dim]} — post-HGT embeddings
        """
        # --- Initial feature projection ---
        x_dict: Dict[str, Tensor] = {}
        for ntype in self.node_types:
            if ntype not in data.node_types:
                continue
            x = data[ntype].x
            if x is None:
                continue
            x_dict[ntype] = self.act(self.input_proj[ntype](x))

        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in data.edge_types
            if hasattr(data[etype], "edge_index")
        }

        # --- HGT message passing ---
        # Note: PyG's HGTConv only emits destination node types in its output.
        # Source-only types (those that never appear as dst in any edge_type,
        # e.g. `device_node` in UNSW-MG24) would be silently dropped between
        # layers, causing the next layer's `k_dict[src]` lookup to KeyError.
        # We carry such types forward by merging the previous x_dict.
        for i, hgt_layer in enumerate(self.hgt_layers):
            x_dict_new = hgt_layer(x_dict, edge_index_dict)
            is_last = i == len(self.hgt_layers) - 1

            x_dict_merged: Dict[str, Tensor] = dict(x_dict)  # preserve source-only types
            for ntype, h in x_dict_new.items():
                if is_last:
                    x_dict_merged[ntype] = h
                else:
                    x_dict_merged[ntype] = self.dropout(self.act(h))
            x_dict = x_dict_merged

        h_dict: Dict[str, Tensor] = x_dict

        # --- Classification logits for target node type ---
        if self.target_node_type not in h_dict:
            raise RuntimeError(
                f"Target node type '{self.target_node_type}' not found in h_dict. "
                f"Available types: {list(h_dict.keys())}"
            )
        logits = self.classifier(h_dict[self.target_node_type])

        return logits, h_dict

    def get_embeddings(self, data: HeteroData) -> Dict[str, Tensor]:
        """Convenience wrapper returning only h_dict (no logits)."""
        _, h_dict = self.forward(data)
        return h_dict
