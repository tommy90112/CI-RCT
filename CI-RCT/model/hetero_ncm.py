"""
HeteroNCM — Module 2b of CI-RCT.

Type-aware Neural Causal Model for heterogeneous graphs.

Computes the type-conditioned causal effect (do-calculus approximation):

    CE_τ(u → v) = P(ŷ | do(h_u = h_u_actual)) − P(ŷ | do(h_u = 0))

where P(ŷ | do(h_u = x)) is approximated by a per-edge-type MLP that maps
[h_source ‖ type_emb_source] → scalar probability.

The "do(h_u = 0)" baseline (null intervention) corresponds to removing the
source node's influence on the target, consistent with Pearl's do-calculus
edge-cutting operation.

Per-edge-type MLPs capture semantically distinct causal mechanisms.
Node type embeddings distinguish node roles in the graph ontology.

Standalone module — no CXGNN dependency.

Reference: CI-RCT_Thesis_Plan.md § 5.3.2
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model.typed_causal_graph import TypedCausalGraph


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    """
    Build a sigmoid-output MLP.

    Architecture: Linear → ReLU → [Linear → ReLU]*(L-1) → Linear → Sigmoid
    """
    layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers += [nn.Linear(hidden_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


class HeteroNCM(nn.Module):
    """
    Heterogeneous Neural Causal Model.

    A graph-level NCM: one set of per-edge-type MLPs is shared across all
    target nodes in the graph.  For each directed edge (u → v), CE(u → v)
    is computed using the edge-type MLP conditioned on u's embedding and
    u's type embedding.

    Args:
        node_emb_dim:      Dimension of node embeddings from HGT backbone
        all_node_types:    Sorted list of all node type strings
        all_edge_types:    Sorted list of all edge type strings
        node_type_emb_dim: Dimension of node-type embedding vectors
        ncm_h_size:        Hidden size of each per-edge-type MLP
        ncm_h_layers:      Number of hidden layers in each MLP
    """

    def __init__(
        self,
        node_emb_dim: int,
        all_node_types: List[str],
        all_edge_types: List[str],
        node_type_emb_dim: int = 16,
        ncm_h_size: int = 64,
        ncm_h_layers: int = 2,
    ) -> None:
        super().__init__()

        if not all_node_types:
            raise ValueError("all_node_types must not be empty.")
        if not all_edge_types:
            raise ValueError("all_edge_types must not be empty.")

        self.node_emb_dim = node_emb_dim
        self.node_type_emb_dim = node_type_emb_dim
        self.all_node_types: List[str] = sorted(all_node_types)
        self.all_edge_types: List[str] = sorted(all_edge_types)

        self.node_type_to_idx: Dict[str, int] = {
            t: i for i, t in enumerate(self.all_node_types)
        }

        # Learnable type embeddings
        self.type_embeddings = nn.Embedding(
            num_embeddings=len(self.all_node_types),
            embedding_dim=node_type_emb_dim,
        )

        # Per-edge-type MLP: input = [h_source (D) ‖ type_emb (T)] → P ∈ (0,1)
        ncm_input_dim = node_emb_dim + node_type_emb_dim
        self.edge_type_models = nn.ModuleDict(
            {
                etype: _build_mlp(ncm_input_dim, ncm_h_size, ncm_h_layers)
                for etype in self.all_edge_types
            }
        )

    # ── Core CE computation ───────────────────────────────────────────────────

    def compute_causal_effect(
        self,
        h_source: Tensor,
        edge_type: str,
        source_node_type: str,
    ) -> Tensor:
        """
        Compute CE_τ(source → target) via do-calculus approximation.

        CE = P(ŷ | do(h_u = h_actual)) − P(ŷ | do(h_u = 0))
           = sigmoid(MLP(h_source ‖ type_emb)) − sigmoid(MLP(0 ‖ type_emb))

        Args:
            h_source:         Source node embedding  [node_emb_dim]
            edge_type:        Edge type label (must be in all_edge_types)
            source_node_type: Node type of source

        Returns:
            Tensor: Scalar CE value ∈ (−1, 1)

        Raises:
            KeyError: if edge_type is not registered
        """
        if edge_type not in self.edge_type_models:
            raise KeyError(
                f"Edge type '{edge_type}' not registered. "
                f"Known types: {self.all_edge_types}"
            )

        type_idx = torch.tensor(
            self.node_type_to_idx.get(source_node_type, 0),
            dtype=torch.long,
            device=h_source.device,
        )
        type_emb = self.type_embeddings(type_idx)  # [node_type_emb_dim]

        # do(h_source = h_actual): observed value
        u_actual = torch.cat([h_source, type_emb], dim=-1)
        p_actual = self.edge_type_models[edge_type](u_actual).squeeze(-1)

        # do(h_source = 0): null intervention (cut incoming edge)
        u_null = torch.cat([torch.zeros_like(h_source), type_emb], dim=-1)
        p_null = self.edge_type_models[edge_type](u_null).squeeze(-1)

        return p_actual - p_null

    # ── Batch forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        flat_h: Dict[int, Tensor],
        causal_graph: TypedCausalGraph,
    ) -> Dict[Tuple[int, int], Tensor]:
        """
        Compute CE for all directed edges in the causal graph.

        Args:
            flat_h:        {global_node_id: embedding_tensor [D]}
            causal_graph:  TypedCausalGraph (uses edge_type_map and node_type)

        Returns:
            {(src, dst): CE_scalar_tensor}  — with gradient
        """
        causal_effects: Dict[Tuple[int, int], Tensor] = {}

        for (src, dst), edge_type in causal_graph.edge_type_map.items():
            if src not in flat_h or dst not in flat_h:
                continue
            if (src, dst) in causal_effects:
                continue

            src_type = causal_graph.node_type.get(src, self.all_node_types[0])
            causal_effects[(src, dst)] = self.compute_causal_effect(
                flat_h[src], edge_type, src_type
            )

        return causal_effects

    def supervised_ncm_loss(
        self,
        flat_h: Dict[int, Tensor],
        causal_graph: TypedCausalGraph,
        node_labels: "torch.Tensor",
        target_type_offset: int,
    ) -> "torch.Tensor":
        """
        Supervision loss for the NCM: train each edge-type MLP to predict
        the fraud probability of the *destination* node from the *source*
        node's embedding.

        For an edge (src → dst):
            p_actual = sigmoid(MLP(h_src ‖ type_emb_src))
            label    = y_dst  (1 if fraud, 0 if licit)
            loss     += BCE(p_actual, label)

        This gives NCM a directional signal: edges pointing to fraud nodes
        should yield high CE; edges to licit nodes should yield low CE.

        Args:
            flat_h:              {global_id: embedding [D]}
            causal_graph:        TypedCausalGraph
            node_labels:         Label tensor [N_target] (long, 0/1)
            target_type_offset:  Global ID offset for the target node type

        Returns:
            Scalar BCE loss tensor (0 if no valid edges found)
        """
        import torch.nn.functional as F

        losses: list = []
        n_labels = node_labels.size(0)
        device = next(self.parameters()).device

        for (src, dst), edge_type in causal_graph.edge_type_map.items():
            dst_local = dst - target_type_offset
            if dst_local < 0 or dst_local >= n_labels:
                continue
            if src not in flat_h:
                continue
            if edge_type not in self.edge_type_models:
                continue

            y = node_labels[dst_local].float().to(device)
            src_type = causal_graph.node_type.get(src, self.all_node_types[0])
            type_idx = torch.tensor(
                self.node_type_to_idx.get(src_type, 0),
                dtype=torch.long, device=device,
            )
            type_emb = self.type_embeddings(type_idx)
            u_actual = torch.cat([flat_h[src].to(device), type_emb], dim=-1)
            p_actual = self.edge_type_models[edge_type](u_actual).squeeze(-1)
            losses.append(F.binary_cross_entropy(p_actual, y))

        if not losses:
            return torch.zeros(1, device=device)
        return torch.stack(losses).mean()

    def detached_causal_effects(
        self,
        flat_h: Dict[int, Tensor],
        causal_graph: TypedCausalGraph,
    ) -> Dict[Tuple[int, int], float]:
        """
        Same as forward() but returns plain Python floats (no grad).
        Safe to pass to RootCauseTracer and Asymmetric Shapley.
        """
        with torch.no_grad():
            ce_tensors = self.forward(flat_h, causal_graph)
        return {k: v.item() for k, v in ce_tensors.items()}
