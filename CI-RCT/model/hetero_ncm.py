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
        baseline:          Null-intervention reference for CE = p_actual − p_null.
                           "zero" = do(h_u=0) (legacy, OOD baseline);
                           "type_mean" = do(h_u=E[h_type]) (in-distribution,
                           recentres CE so its sign is interpretable);
                           "marginal" = p_null=E[MLP(h)] over same-type sources
                           (true marginal do(h_u ~ P(h_type)); removes the
                           Jensen gap of type_mean, E[CE] over a type is 0).
    """

    def __init__(
        self,
        node_emb_dim: int,
        all_node_types: List[str],
        all_edge_types: List[str],
        node_type_emb_dim: int = 16,
        ncm_h_size: int = 64,
        ncm_h_layers: int = 2,
        baseline: str = "zero",
    ) -> None:
        super().__init__()

        if not all_node_types:
            raise ValueError("all_node_types must not be empty.")
        if not all_edge_types:
            raise ValueError("all_edge_types must not be empty.")
        if baseline not in ("zero", "type_mean", "marginal"):
            raise ValueError(
                f"baseline must be 'zero', 'type_mean' or 'marginal', "
                f"got {baseline!r}."
            )

        self.node_emb_dim = node_emb_dim
        self.node_type_emb_dim = node_type_emb_dim
        self.baseline = baseline
        # Diagnostic: labelled-edge count per edge type from the most recent
        # supervised_ncm_loss call. Zero-count types are exactly the MLPs that
        # never train and whose CE collapses to ~0 at eval time.
        self.last_supervision_counts: Dict[str, int] = {}
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
        baseline_h: Optional[Tensor] = None,
        baseline_p: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute CE_τ(source → target) via do-calculus approximation.

        CE = P(ŷ | do(h_u = h_actual)) − P(ŷ | do(h_u = h_base))
           = sigmoid(MLP(h_source ‖ type_emb)) − sigmoid(MLP(h_base ‖ type_emb))

        Args:
            h_source:         Source node embedding  [node_emb_dim]
            edge_type:        Edge type label (must be in all_edge_types)
            source_node_type: Node type of source
            baseline_h:       Null-intervention embedding h_base [node_emb_dim].
                              None → zero vector (do(h_u=0), legacy). Callers in
                              "type_mean" mode pass the source type's mean
                              embedding so CE is recentred at 0.
            baseline_p:       Precomputed p_null scalar. Takes precedence over
                              baseline_h. Callers in "marginal" mode pass the
                              mean MLP prediction over same-type sources
                              (E[MLP(h)], the marginal null intervention).

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

        # do(h_source = h_base): null intervention.
        # p_null precomputed (marginal, E[MLP(h)]) or evaluated at h_base =
        # 0 (legacy, OOD) / E[h_type] (type_mean, in-distribution).
        if baseline_p is not None:
            return p_actual - baseline_p

        h_base = torch.zeros_like(h_source) if baseline_h is None else baseline_h
        u_null = torch.cat([h_base, type_emb], dim=-1)
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

        # In "type_mean" mode, h_base for an edge is the mean embedding of the
        # source's node type — an in-distribution interventional baseline.
        type_baselines = (
            self._compute_type_baselines(flat_h, causal_graph)
            if self.baseline == "type_mean"
            else None
        )
        # In "marginal" mode, p_null for an edge type is the mean MLP
        # prediction over all same-type sources — E[MLP(h)], not MLP(E[h]).
        # Computed lazily, once per (edge_type, src_type).
        type_buckets = (
            self._bucket_embeddings_by_type(flat_h, causal_graph)
            if self.baseline == "marginal"
            else None
        )
        marginal_p_null: Dict[Tuple[str, str], Tensor] = {}

        for (src, dst), edge_type in causal_graph.edge_type_map.items():
            if src not in flat_h or dst not in flat_h:
                continue
            if (src, dst) in causal_effects:
                continue

            src_type = causal_graph.node_type.get(src, self.all_node_types[0])
            baseline_h = (
                type_baselines.get(src_type) if type_baselines is not None else None
            )
            baseline_p = None
            if type_buckets is not None:
                key = (edge_type, src_type)
                if key not in marginal_p_null:
                    marginal_p_null[key] = self._marginal_null_prediction(
                        type_buckets[src_type], edge_type, src_type
                    )
                baseline_p = marginal_p_null[key]
            causal_effects[(src, dst)] = self.compute_causal_effect(
                flat_h[src], edge_type, src_type, baseline_h, baseline_p
            )

        return causal_effects

    def _compute_type_baselines(
        self,
        flat_h: Dict[int, Tensor],
        causal_graph: TypedCausalGraph,
    ) -> Dict[str, Tensor]:
        """
        Per-node-type mean embedding E[h_type] over all nodes present in flat_h.

        Used as the in-distribution null-intervention baseline for "type_mean"
        mode: do(h_u = E[h_{type(u)}]) instead of do(h_u = 0).
        """
        buckets: Dict[str, List[Tensor]] = {}
        for node_id, h in flat_h.items():
            t = causal_graph.node_type.get(node_id, self.all_node_types[0])
            buckets.setdefault(t, []).append(h)
        return {t: torch.stack(hs).mean(dim=0) for t, hs in buckets.items()}

    def _bucket_embeddings_by_type(
        self,
        flat_h: Dict[int, Tensor],
        causal_graph: TypedCausalGraph,
    ) -> Dict[str, Tensor]:
        """Stack all embeddings per node type: {type: [N_type, D]}."""
        buckets: Dict[str, List[Tensor]] = {}
        for node_id, h in flat_h.items():
            t = causal_graph.node_type.get(node_id, self.all_node_types[0])
            buckets.setdefault(t, []).append(h)
        return {t: torch.stack(hs) for t, hs in buckets.items()}

    def _marginal_null_prediction(
        self,
        type_embs: Tensor,
        edge_type: str,
        source_node_type: str,
    ) -> Tensor:
        """
        p_null for "marginal" mode: E[MLP(h ‖ type_emb)] over all nodes of the
        source type — the marginal null intervention do(h_u ~ P(h_type)).

        Unlike MLP(E[h]) ("type_mean"), this commutes with the expectation, so
        CE averaged over all same-type sources of an edge type is exactly 0.

        Args:
            type_embs: Stacked source-type embeddings [N_type, node_emb_dim]

        Returns:
            Tensor: scalar mean prediction
        """
        type_idx = torch.tensor(
            self.node_type_to_idx.get(source_node_type, 0),
            dtype=torch.long,
            device=type_embs.device,
        )
        type_emb = self.type_embeddings(type_idx).expand(type_embs.size(0), -1)
        u = torch.cat([type_embs, type_emb], dim=-1)
        return self.edge_type_models[edge_type](u).mean()

    def supervised_ncm_loss(
        self,
        flat_h: Dict[int, Tensor],
        causal_graph: TypedCausalGraph,
        node_labels: "torch.Tensor",
        target_type_offset: int,
        wallet_labels: Optional["torch.Tensor"] = None,
        wallet_type_offset: int = 0,
        multi_task_labels: Optional[Dict[str, "torch.Tensor"]] = None,
        type_offsets: Optional[Dict[str, int]] = None,
        edge_balance: str = "none",
    ) -> "torch.Tensor":
        """
        Supervision loss for the NCM: train each edge-type MLP to predict
        the destination node's binary label from the source's embedding.

        For an edge (src → dst):
            p_actual = sigmoid(MLP(h_src ‖ type_emb_src))
            label    = y_dst  (1 if malicious/fraud, 0 if benign)
            loss     += BCE(p_actual, label)

        This gives NCM a directional signal: edges pointing to malicious
        nodes should yield high CE; edges to benign nodes should yield low CE.

        Four label-resolution paths (in priority order):

          1. **Multi-task** (`multi_task_labels` + `type_offsets` given):
             Look up each edge's destination type and use the corresponding
             per-type label tensor. This is the path used by UNSW-MG24
             where flow_node, process_node, and measurement_node are all
             labelled (DD-3). Edges whose dst_type has no label tensor are
             skipped, so e.g. host_node → process_node uses the process_node
             label, and host_node → flow_node uses the flow_node label.

          2. **Elliptic++ wallet→tx** (`wallet_labels` given):
             Special case where wallets carry their own (wallet_labels) and
             transactions carry node_labels. Falls back to this if (1) is
             not provided and the edge is wallet→transaction; supervised by
             the SOURCE wallet's label.

          2b. **Elliptic++ *→wallet** (`wallet_labels` given): any edge whose
             dst is a wallet (e.g. tx→wallet) is supervised by the DESTINATION
             wallet's label. Without this the tx→wallet MLP is never trained
             (dst=wallet is out of the target-tx label range in Path 3), so its
             CE collapses to ~0 and the tracer loses all directional signal on
             wallet-bound hops.

          3. **Single-task default**: assume dst is the target node type
             and look up node_labels[dst − target_type_offset]. Edges whose
             dst is not in the target type range are silently skipped.

        Args:
            flat_h:              {global_id: embedding [D]}
            causal_graph:        TypedCausalGraph
            node_labels:         Label tensor [N_target] (long, 0/1) for
                                 the primary target type — used in path (3).
            target_type_offset:  Global ID offset for the primary target type.
            wallet_labels:       Elliptic++ wallet labels (path 2 only).
            wallet_type_offset:  Global ID offset for the wallet type.
            multi_task_labels:   {node_type: label_tensor} for the multi-task
                                 supervision path (1). When provided together
                                 with `type_offsets`, every labelled dst type
                                 contributes BCE supervision for its incoming
                                 causal edges.
            type_offsets:        {node_type: global_offset} (path 1 lookup).
            edge_balance:        Per-edge-type loss re-weighting mode.
                                   - "none"    : plain mean over all BCE terms
                                                 (legacy behaviour; dominant
                                                 edge type drowns out rare ones)
                                   - "uniform" : each edge type contributes
                                                 equally regardless of count
                                                 (mean of per-type means)
                                   - "sqrt"    : inverse-sqrt-frequency weight
                                                 normalised to mean=1; rare
                                                 edges get strong but not
                                                 runaway gradient — recommended
                                                 default for highly imbalanced
                                                 hetero-graphs (DD-17)
                                   - "inverse" : 1/N_edges per type (aggressive,
                                                 usable as ablation comparator)

                                 See unsw_mg24_plan.md § DD-17 for the
                                 motivation: on MG24 `host→flow` had ~200k
                                 edges while `process→host` had ~1k, leading
                                 to NCM CE≈0.001 on sparse edges and tracer
                                 stuck at depth 1 in evaluation.

        Returns:
            Scalar BCE loss tensor (0 if no valid edges found)
        """
        import torch.nn.functional as F

        losses_by_type: Dict[str, list] = {et: [] for et in self.all_edge_types}
        n_labels = node_labels.size(0)
        device = next(self.parameters()).device
        use_multi_task = multi_task_labels is not None and type_offsets is not None

        for (src, dst), edge_type in causal_graph.edge_type_map.items():
            if src not in flat_h:
                continue
            if edge_type not in self.edge_type_models:
                continue

            src_type = causal_graph.node_type.get(src, self.all_node_types[0])
            dst_type = causal_graph.node_type.get(dst, self.all_node_types[0])

            y: Optional[Tensor] = None

            # Path 1: multi-task — supervise from dst's per-type label.
            if use_multi_task:
                label_tensor = multi_task_labels.get(dst_type)
                dst_off = type_offsets.get(dst_type)
                if label_tensor is None or dst_off is None:
                    continue
                dst_local = dst - dst_off
                if dst_local < 0 or dst_local >= label_tensor.size(0):
                    continue
                # Skip "unknown" labels (Elliptic++ class=3 convention).
                y_int = int(label_tensor[dst_local].item())
                if y_int not in (0, 1):
                    continue
                y = label_tensor[dst_local].float().to(device)

            # Path 2: Elliptic++ wallet→tx supervised by wallet (src) label.
            elif src_type == "wallet" and dst_type == "transaction":
                if wallet_labels is None:
                    continue
                src_local = src - wallet_type_offset
                if src_local < 0 or src_local >= wallet_labels.size(0):
                    continue
                if wallet_labels[src_local].item() not in (0, 1):
                    continue
                y = wallet_labels[src_local].float().to(device)

            # Path 2b: Elliptic++ *→wallet (e.g. tx→wallet) supervised by the
            # destination wallet's label. Without this the tx→wallet edge-type
            # MLP is never trained (dst=wallet falls out of the target-tx label
            # range in Path 3), leaving CE≈0 and the tracer with no directional
            # signal on every wallet-bound hop.
            elif dst_type == "wallet":
                if wallet_labels is None:
                    continue
                dst_local = dst - wallet_type_offset
                if dst_local < 0 or dst_local >= wallet_labels.size(0):
                    continue
                if wallet_labels[dst_local].item() not in (0, 1):
                    continue
                y = wallet_labels[dst_local].float().to(device)

            # Path 3: single-task default — dst is target type.
            else:
                dst_local = dst - target_type_offset
                if dst_local < 0 or dst_local >= n_labels:
                    continue
                y = node_labels[dst_local].float().to(device)

            type_idx = torch.tensor(
                self.node_type_to_idx.get(src_type, 0),
                dtype=torch.long, device=device,
            )
            type_emb = self.type_embeddings(type_idx)
            u_actual = torch.cat([flat_h[src].to(device), type_emb], dim=-1)
            p_actual = self.edge_type_models[edge_type](u_actual).squeeze(-1)
            losses_by_type[edge_type].append(F.binary_cross_entropy(p_actual, y))

        # Zero-count edge types are the MLPs that receive no gradient this
        # call — surfaced so undertrained CE (≈0 at eval) is visible upfront.
        self.last_supervision_counts = {
            et: len(ls) for et, ls in losses_by_type.items()
        }

        return self._aggregate_ncm_losses(losses_by_type, edge_balance, device)

    def _aggregate_ncm_losses(
        self,
        losses_by_type: Dict[str, list],
        edge_balance: str,
        device: "torch.device",
    ) -> "torch.Tensor":
        """Combine per-edge-type BCE terms into a single scalar loss.

        Centralised so the four `edge_balance` modes share the same
        zero-loss / numerical-normalisation guarantees.
        """
        # Drop edge types with no valid samples this batch.
        non_empty = {et: ls for et, ls in losses_by_type.items() if ls}
        if not non_empty:
            return torch.zeros(1, device=device)

        if edge_balance == "none":
            # Legacy: every BCE term weighted equally → dominant edge type wins.
            flat = [t for ls in non_empty.values() for t in ls]
            return torch.stack(flat).mean()

        # Per-edge-type aggregate (mean within each type).
        per_type_mean = {
            et: torch.stack(ls).mean() for et, ls in non_empty.items()
        }
        counts = {et: float(len(ls)) for et, ls in non_empty.items()}

        if edge_balance == "uniform":
            weights = {et: 1.0 for et in per_type_mean}
        elif edge_balance == "sqrt":
            raw = {et: 1.0 / (counts[et] ** 0.5) for et in per_type_mean}
            s = sum(raw.values()) / max(1, len(raw))  # mean of raw weights
            weights = {et: raw[et] / s for et in per_type_mean}  # mean=1
        elif edge_balance == "inverse":
            raw = {et: 1.0 / counts[et] for et in per_type_mean}
            s = sum(raw.values()) / max(1, len(raw))
            weights = {et: raw[et] / s for et in per_type_mean}  # mean=1
        else:
            raise ValueError(
                f"Unknown edge_balance mode: {edge_balance!r}. "
                f"Expected one of 'none' / 'uniform' / 'sqrt' / 'inverse'."
            )

        weighted = [weights[et] * per_type_mean[et] for et in per_type_mean]
        return torch.stack(weighted).mean()

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
