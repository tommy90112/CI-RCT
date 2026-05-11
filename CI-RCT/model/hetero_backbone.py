"""
HeteroGNNBackbone — Module 1 of CI-RCT.

Heterogeneous Graph Transformer (HGT) backbone for node classification
on heterogeneous graphs.  Dataset-agnostic: accepts any HeteroData
structure and returns per-type node embeddings plus target-type logits.

Reference: CI-RCT_Thesis_Plan.md § 5.2
"""
from typing import Dict, List, Optional, Tuple

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
        exclude_node_types: Node types to drop from message passing while
                            keeping them in the underlying HeteroData (so other
                            modules — e.g. RootCauseTracer — can still traverse
                            the full graph). DD-8 Fix 4 fairness setup uses
                            this to exclude host_node from detection while
                            retaining it for downstream tracing. Default: None.
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
        exclude_node_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        all_node_types, all_edge_types = metadata

        # Filter node and edge types for detection-time message passing.
        self.excluded_types: List[str] = sorted(set(exclude_node_types or []))
        excluded_set = set(self.excluded_types)
        node_types = [t for t in all_node_types if t not in excluded_set]
        edge_types = [
            (s, r, d) for (s, r, d) in all_edge_types
            if s not in excluded_set and d not in excluded_set
        ]

        if not node_types:
            raise ValueError(
                "All node types excluded — no nodes left for detection."
            )

        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.target_node_type = target_node_type or self.node_types[0]

        if self.target_node_type in excluded_set:
            raise ValueError(
                f"target_node_type {self.target_node_type!r} cannot be in "
                f"exclude_node_types {self.excluded_types}."
            )

        # Per-type input projection: maps raw features → hidden_dim.
        # Use in_channels=-1 for lazy (deferred) initialisation if sizes unknown.
        in_ch_map = in_channels_dict or {ntype: -1 for ntype in self.node_types}
        self.input_proj = nn.ModuleDict(
            {ntype: Linear(in_ch_map.get(ntype, -1), hidden_dim) for ntype in self.node_types}
        )

        # HGT message-passing layers — use filtered metadata so HGTConv only
        # allocates parameters for the included node/edge types.
        filtered_metadata = (self.node_types, self.edge_types)
        self.hgt_layers = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=filtered_metadata,
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
        # Excluded node types (e.g. host_node under DD-8 Fix 4) are silently
        # skipped here; their entries never appear in x_dict so they cannot
        # contribute to message passing.
        x_dict: Dict[str, Tensor] = {}
        for ntype in self.node_types:
            if ntype not in data.node_types:
                continue
            x = data[ntype].x
            if x is None:
                continue
            x_dict[ntype] = self.act(self.input_proj[ntype](x))

        # Filter edge_index_dict to only the (filtered) edge types HGTConv
        # was built for. Edges involving excluded node types are dropped.
        allowed_edges = set(self.edge_types)
        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in data.edge_types
            if etype in allowed_edges and hasattr(data[etype], "edge_index")
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
