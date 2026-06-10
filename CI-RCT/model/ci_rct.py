"""
CI_RCT — Four-module causal intervention framework.

Architecture:
    Input: Directed Heterogeneous Graph G = (V, E, τ_v, τ_e, T)
      ↓
    [Module 1]  HeteroGNNBackbone (HGT)
                h_dict + logits
      ↓        ↓
    [Module 2]  Causal Intervention Engine          [Module 4 — training only]
                TypedCausalGraph                    CausalAdversarialGAN
                HeteroNCM → CE(u→v)                 Generator: camouflage nodes
                Asymmetric Causal Shapley → φ^asym  Discriminator = HeteroGNN
      ↓
    [Module 3]  RootCauseTracer
                Backward BFS on CE / φ scores
                → root cause node + causal chain

Joint loss (training):
    L_total = L_detection + λ1 · L_adversarial + λ2 · L_stability

    L_detection:  CrossEntropy on target node type
    L_adversarial: WGAN-GP generator loss (Module 4)
    L_stability:  ‖φ_t − φ_{t-1}‖² — Causal Shapley φ stability
                  ensures adversarial training does not destabilise explanations

Reference: CI-RCT_Thesis_Plan.md § 5.1–5.6
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from configs.config import CI_RCT_Config
from model.causal_adversarial_gan import CausalAdversarialGAN
from model.causal_shapley import compute_asymmetric_causal_shapley, compute_shapley_edge_scores
from model.hetero_backbone import HeteroGNNBackbone
from model.hetero_ncm import HeteroNCM
from model.root_cause_tracer import RootCauseTracer
from model.typed_causal_graph import TypedCausalGraph


class CI_RCT(nn.Module):
    """
    Causal Intervention-Based Root Cause Tracing framework.

    Args:
        config:            CI_RCT_Config dataclass (frozen)
        metadata:          HeteroData.metadata() — (node_types, edge_types)
        in_channels_dict:  {node_type: feature_dim}; None for lazy init
        node_feature_dim:  Feature dim of target node type (for GAN).
                           Required when use_gan=True.
        use_gan:           Enable Module 4 (CausalAdversarialGAN).
                           Set True for Elliptic++ training; False for
                           dataset-agnostic evaluation-only mode.
    """

    def __init__(
        self,
        config: CI_RCT_Config,
        metadata: tuple,
        in_channels_dict: Optional[Dict[str, int]] = None,
        node_feature_dim: Optional[int] = None,
        use_gan: bool = True,
        num_classes: int = 2,
        backbone_exclude_node_types: Optional[List[str]] = None,
        use_reconstruction: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.use_gan = use_gan
        self.use_reconstruction = use_reconstruction
        self.backbone_exclude_node_types: List[str] = list(
            backbone_exclude_node_types or []
        )

        node_types, edge_types = metadata
        self.node_types: List[str] = sorted(node_types)
        self.node_type_to_idx: Dict[str, int] = {
            t: i for i, t in enumerate(self.node_types)
        }

        # All edge types as "src__to__dst" strings for HeteroNCM
        self._edge_type_strs: List[str] = sorted(
            f"{s}__to__{d}" for s, _, d in edge_types
        )

        # ── Module 1: HGT Backbone ────────────────────────────────────────
        self.backbone = HeteroGNNBackbone(
            metadata=metadata,
            in_channels_dict=in_channels_dict,
            hidden_dim=config.hidden_dim,
            num_classes=num_classes,
            num_heads=config.num_heads,
            num_layers=config.num_hgt_layers,
            target_node_type=config.target_node_type,
            dropout=config.dropout,
            exclude_node_types=self.backbone_exclude_node_types,
        )

        # ── Module 2b: HeteroNCM ─────────────────────────────────────────
        self.hetero_ncm = HeteroNCM(
            node_emb_dim=config.hidden_dim,
            all_node_types=self.node_types,
            all_edge_types=self._edge_type_strs,
            node_type_emb_dim=config.node_type_emb_dim,
            ncm_h_size=config.ncm_h_size,
            ncm_h_layers=config.ncm_h_layers,
        )

        # ── Module 4: CausalAdversarialGAN ───────────────────────────────
        # Requires node_feature_dim of the target node type.
        if use_gan:
            if node_feature_dim is None:
                raise ValueError(
                    "node_feature_dim must be provided when use_gan=True."
                )
            self.causal_gan = CausalAdversarialGAN(
                node_feature_dim=node_feature_dim,
                type_emb_dim=config.node_type_emb_dim,
                hidden_dim=config.gan_hidden_dim,
                noise_std=config.noise_std,
                gp_weight=config.gp_weight,
            )
        else:
            self.causal_gan = None

        # ── GraphBEAN-style reconstruction decoders (Step 1, training only) ──
        # Self-supervision over ALL nodes (incl. the unlabeled majority the
        # supervised heads ignore): a per-type feature decoder reconstructs the
        # raw node features, and a bilinear edge decoder reconstructs the
        # bipartite cross-type adjacency (wallet↔transaction for Elliptic++).
        # Built only when use_reconstruction → no params / no effect otherwise.
        self.feature_decoders: Optional[nn.ModuleDict] = None
        self.edge_decoder: Optional[nn.Module] = None
        self._recon_edge_type: Optional[Tuple[str, str, str]] = None
        if use_reconstruction:
            if in_channels_dict is None:
                raise ValueError(
                    "in_channels_dict (per-type raw feature dims) is required "
                    "when use_reconstruction=True."
                )
            self.feature_decoders = nn.ModuleDict(
                {
                    ntype: nn.Sequential(
                        nn.Linear(config.hidden_dim, config.hidden_dim),
                        nn.ELU(),
                        nn.Linear(config.hidden_dim, int(raw_dim)),
                    )
                    for ntype, raw_dim in in_channels_dict.items()
                }
            )
            # Bipartite edge = the (deterministically chosen) cross-type edge
            # type (src != dst). For Elliptic++ this is wallet↔transaction; for
            # same-type-only graphs edge reconstruction is skipped.
            cross = sorted(et for et in edge_types if et[0] != et[2])
            self._recon_edge_type = cross[0] if cross else None
            if self._recon_edge_type is not None:
                self.edge_decoder = nn.Bilinear(
                    config.hidden_dim, config.hidden_dim, 1
                )

        # Buffer: previous-step φ values for stability loss (φ_{t-1})
        # Stored as plain dict — not a model parameter
        self._prev_phi: Optional[Dict[int, float]] = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, data: HeteroData) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Backbone forward pass.

        Returns:
            logits:  [N_target, 2] — raw classification logits
            h_dict:  {node_type: [N, hidden_dim]} — node embeddings
        """
        return self.backbone(data)

    # ── Causal effects ────────────────────────────────────────────────────────

    def compute_causal_effects(
        self,
        flat_h: Dict[int, Tensor],
        causal_graph: TypedCausalGraph,
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute all directed pairwise CE scores via HeteroNCM (no grad).

        Args:
            flat_h:       {global_node_id: embedding [D]}
            causal_graph: TypedCausalGraph for the local subgraph

        Returns:
            {(src, dst): CE_float}
        """
        return self.hetero_ncm.detached_causal_effects(flat_h, causal_graph)

    # ── Reconstruction self-supervision (Step 1) ───────────────────────────────

    def reconstruction_loss(
        self, data: HeteroData, h_dict: Dict[str, Tensor]
    ) -> Tensor:
        """
        GraphBEAN-style self-supervised reconstruction loss over ALL nodes.

            L_recon = mean_type MSE(decode(h) , x)        # feature reconstruction
                      + BCE(bilinear(h_src, h_dst) , 1/0) # bipartite edge recon

        Feature reconstruction trains EVERY node (incl. the unlabeled majority
        the supervised heads never touch); edge reconstruction trains the
        bipartite cross-type adjacency (wallet↔transaction) with equal-size
        negative sampling. Returns 0 if reconstruction is disabled.
        """
        device = data[self.config.target_node_type].x.device
        if not self.use_reconstruction or self.feature_decoders is None:
            return torch.zeros(1, device=device)

        # (a) feature reconstruction — averaged over node types present.
        feat_terms: List[Tensor] = []
        for ntype, dec in self.feature_decoders.items():
            if ntype not in h_dict or ntype not in data.node_types:
                continue
            x = getattr(data[ntype], "x", None)
            if x is None:
                continue
            feat_terms.append(F.mse_loss(dec(h_dict[ntype]), x))
        feat_loss = (
            torch.stack(feat_terms).mean()
            if feat_terms
            else torch.zeros(1, device=device)
        )

        # (b) bipartite edge reconstruction — positives + equal negatives.
        edge_loss = torch.zeros(1, device=device)
        et = self._recon_edge_type
        if (
            self.edge_decoder is not None
            and et is not None
            and et in data.edge_types
            and et[0] in h_dict
            and et[2] in h_dict
        ):
            ei = data[et].edge_index
            if ei is not None and ei.size(1) > 0:
                h_src, h_dst = h_dict[et[0]], h_dict[et[2]]
                n_pos = ei.size(1)
                pos = self.edge_decoder(h_src[ei[0]], h_dst[ei[1]]).squeeze(-1)
                neg_src = torch.randint(0, h_src.size(0), (n_pos,), device=device)
                neg_dst = torch.randint(0, h_dst.size(0), (n_pos,), device=device)
                neg = self.edge_decoder(h_src[neg_src], h_dst[neg_dst]).squeeze(-1)
                logits_e = torch.cat([pos, neg])
                targets = torch.cat(
                    [torch.ones_like(pos), torch.zeros_like(neg)]
                )
                edge_loss = F.binary_cross_entropy_with_logits(logits_e, targets)

        return feat_loss + edge_loss

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_total_loss(
        self,
        data: HeteroData,
        labels: Tensor,
        train_mask: Optional[Tensor] = None,
        causal_graph: Optional[TypedCausalGraph] = None,
        fraud_features: Optional[Tensor] = None,
        topo_order: Optional[List] = None,
        target_node: Optional[int] = None,
        is_critic_step: bool = True,
        class_weight: Optional[Tensor] = None,
        target_type_offset: Optional[int] = None,
        wallet_labels: Optional[Tensor] = None,
        wallet_type_offset: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute L_total = L_detection + λ1·L_adversarial + λ2·L_stability + λ3·L_ncm.

        Args:
            data:            Input HeteroData
            labels:          Ground-truth labels [N_target] (long)
            train_mask:      Boolean mask for training nodes; if provided,
                             detection loss is computed only on masked nodes
            causal_graph:    If provided, compute L_stability via φ values
            fraud_features:  Fraud node raw features for Generator (GAN step)
            topo_order:      TypedCausalGraph topological order (GAN step)
            target_node:     Target node id for φ stability computation
            is_critic_step:  True = update discriminator; False = update generator
            class_weight:    Optional [n_classes] tensor for imbalanced datasets
                             (e.g. [1.0, 44.0] for 1:44 licit/illicit ratio)

        Returns:
            (total_loss, detection_loss, adversarial_loss, stability_loss,
             ncm_loss, recon_loss)
        """
        logits, h_dict = self.forward(data)
        if train_mask is not None:
            detection_loss = F.cross_entropy(
                logits[train_mask], labels[train_mask], weight=class_weight
            )
        else:
            detection_loss = F.cross_entropy(logits, labels, weight=class_weight)

        adv_loss = torch.zeros(1, device=detection_loss.device)
        stability_loss = torch.zeros(1, device=detection_loss.device)

        flat_h = self._build_flat_h(h_dict)

        # ── L_adversarial (Module 4) ──────────────────────────────────────
        if self.use_gan and self.causal_gan is not None and fraud_features is not None:
            type_idx = torch.tensor(
                self.node_type_to_idx.get(self.config.target_node_type, 0),
                dtype=torch.long,
                device=fraud_features.device,
            )
            type_emb = self.hetero_ncm.type_embeddings(type_idx)
            fake_features, _ = self.causal_gan.generate(
                fraud_features, type_emb, topo_order or []
            )

            # Discriminator: uses backbone classifier head on raw features
            # We proxy discriminator_fn as: Linear(hidden_dim → 1) on fake features
            # For WGAN-GP we need D: R^D → R; use backbone.classifier
            def discriminator_fn(x: Tensor) -> Tensor:
                # Project from raw feature dim to hidden dim via input_proj
                ntype = self.config.target_node_type
                proj = self.backbone.input_proj[ntype]
                h = self.backbone.act(proj(x))
                logit = self.backbone.classifier(h)
                return logit[..., 1]  # fraud logit as score

            if is_critic_step:
                real_scores = discriminator_fn(fraud_features)
                adv_loss = self.causal_gan.discriminator_loss(
                    discriminator_fn,
                    real_scores,
                    fake_features,
                    fraud_features,
                    device=detection_loss.device,
                )
            else:
                adv_loss = self.causal_gan.generator_loss(discriminator_fn, fake_features)

        # ── L_stability (Module 2c) ───────────────────────────────────────
        # ‖φ_t − φ_{t-1}‖² — penalise Causal Shapley fluctuations
        if causal_graph is not None and target_node is not None:
            # Compute CE WITH gradients so L_stability backpropagates into NCM.
            ce_tensors = self.hetero_ncm.forward(flat_h, causal_graph)
            ce_scores = {k: v.item() for k, v in ce_tensors.items()}
            phi_current = compute_asymmetric_causal_shapley(
                ce_scores, causal_graph, target_node
            )

            parents = list(causal_graph.parents(target_node))
            n_parents = len(parents)
            if self._prev_phi is not None and phi_current and n_parents > 0:
                common_parents = set(phi_current.keys()) & set(self._prev_phi.keys())
                if common_parents:
                    diffs = []
                    for p in common_parents:
                        ce_t = ce_tensors.get(
                            (p, target_node),
                            torch.zeros(1, device=detection_loss.device),
                        )
                        phi_t = ce_t / n_parents
                        phi_prev = torch.tensor(
                            self._prev_phi[p], dtype=torch.float32,
                            device=detection_loss.device,
                        )
                        diffs.append((phi_t - phi_prev) ** 2)
                    stability_loss = torch.stack(diffs).mean()

            # Update φ buffer (detached — not part of computation graph)
            self._prev_phi = {k: v for k, v in phi_current.items()}

        # ── L_ncm (NCM supervision) ───────────────────────────────────────
        ncm_loss = torch.zeros(1, device=detection_loss.device)
        if causal_graph is not None and target_type_offset is not None:
            ncm_loss = self.hetero_ncm.supervised_ncm_loss(
                flat_h, causal_graph, labels, target_type_offset,
                wallet_labels=wallet_labels,
                wallet_type_offset=wallet_type_offset,
            )

        # ── L_recon (GraphBEAN-style self-supervision, Step 1) ────────────
        recon_loss = torch.zeros(1, device=detection_loss.device)
        if self.use_reconstruction:
            recon_loss = self.reconstruction_loss(data, h_dict)

        total_loss = (
            detection_loss
            + self.config.lambda_adversarial * adv_loss
            + self.config.lambda_stability * stability_loss
            + self.config.lambda_ncm * ncm_loss
            + self.config.lambda_recon * recon_loss
        )

        return total_loss, detection_loss, adv_loss, stability_loss, ncm_loss, recon_loss

    # ── Explanation ───────────────────────────────────────────────────────────

    def explain(
        self,
        data: HeteroData,
        target_node_id: int,
        causal_graph: TypedCausalGraph,
        top_k: int = 3,
    ) -> Dict:
        """
        Produce causal explanation + root cause trace for a target fraud node.

        Outputs:
            logits            — raw prediction logits for target node type
            prediction        — predicted class (0 = normal, 1 = fraud)
            causal_effects    — {(src, tgt): CE_float}
            phi               — Asymmetric Causal Shapley values {node_id: φ}
            edge_scores       — {(src, tgt): edge_score = φ_src × CE}
            root_cause        — identified root cause node ID
            causal_chain      — list [target → ... → root]
            root_cause_type   — node type of root cause node
            top_k_paths       — [(root, chain, score), ...]

        Args:
            data:           Input HeteroData
            target_node_id: Global node ID predicted as fraud
            causal_graph:   Pre-built TypedCausalGraph for local neighbourhood
            top_k:          Number of top-k paths to return
        """
        self.eval()
        with torch.no_grad():
            logits, h_dict = self.forward(data)
            flat_h = self._build_flat_h(h_dict)

        # CE scores from HeteroNCM
        causal_effects = self.compute_causal_effects(flat_h, causal_graph)

        # Asymmetric Causal Shapley φ values
        phi = compute_asymmetric_causal_shapley(causal_effects, causal_graph, target_node_id)

        # Edge-level Shapley scores for visualisation
        edge_scores = compute_shapley_edge_scores(
            phi, causal_effects, causal_graph, target_node_id
        )

        # Root cause tracing: use φ-weighted scores for greedy backward BFS
        phi_weighted_ce: Dict[Tuple[int, int], float] = {}
        for (src, dst), ce in causal_effects.items():
            phi_src = phi.get(src, 1.0)  # default 1.0 if not a direct parent
            phi_weighted_ce[(src, dst)] = abs(phi_src) * ce + ce  # hybrid score

        tracer = RootCauseTracer(
            causal_graph=causal_graph,
            max_hops=self.config.max_hops,
            threshold=self.config.ce_threshold,
        )
        root_cause, chain = tracer.trace_root_cause(target_node_id, phi_weighted_ce)
        top_k_paths = tracer.trace_top_k_paths(target_node_id, phi_weighted_ce, k=top_k)

        # Prediction score for target node
        target_logits = logits[target_node_id]
        prediction = int(target_logits.argmax().item())

        return {
            "logits": target_logits.tolist(),
            "prediction": prediction,
            "causal_effects": causal_effects,
            "phi": phi,
            "edge_scores": edge_scores,
            "root_cause": root_cause,
            "causal_chain": chain,
            "root_cause_type": causal_graph.node_type.get(root_cause, "unknown"),
            "top_k_paths": top_k_paths,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _build_flat_h(self, h_dict: Dict[str, Tensor]) -> Dict[int, Tensor]:
        """
        Convert type-keyed {node_type: [N, D]} to flat {global_id: [D]}.

        Global IDs are assigned by concatenating sorted node types:
            global_id = Σ_{type < current_type} N_type + local_idx
        """
        flat: Dict[int, Tensor] = {}
        offset = 0
        for ntype in self.node_types:
            if ntype not in h_dict:
                continue
            emb = h_dict[ntype]
            for local_idx in range(emb.size(0)):
                flat[offset + local_idx] = emb[local_idx]
            offset += emb.size(0)
        return flat

    def reset_phi_buffer(self) -> None:
        """Reset the φ stability buffer (call at epoch start)."""
        self._prev_phi = None

    # ── Checkpoint format ──────────────────────────────────────────────────
    # v2 checkpoints store the architecture metadata alongside the weights so
    # evaluate.py can rebuild a *matching* model regardless of its CLI
    # defaults. This removes the silent "train with 3 HGT layers, eval with 2
    # → extra layer stays randomly initialised (strict=False), AUC roughly
    # right but F1≈0" trap. Legacy bare-state_dict checkpoints still load via
    # the fallback branch in load_checkpoint().
    _CKPT_FORMAT = "ci_rct/v2"

    def arch_metadata(self) -> Dict[str, object]:
        """
        Architecture fields required to reconstruct this model from a
        checkpoint. Mirrors the values __init__ passes to the backbone / NCM
        constructors; kept here as the single source of truth so save and the
        evaluate-side rebuild never drift apart.
        """
        return {
            "hidden_dim": self.config.hidden_dim,
            "num_hgt_layers": self.config.num_hgt_layers,
            "num_heads": self.config.num_heads,
            "node_type_emb_dim": self.config.node_type_emb_dim,
            "dropout": self.config.dropout,
            "ncm_h_size": self.config.ncm_h_size,
            "ncm_h_layers": self.config.ncm_h_layers,
            "target_node_type": self.config.target_node_type,
            "use_gan": self.use_gan,
            "use_reconstruction": self.use_reconstruction,
            "backbone_exclude_node_types": list(self.backbone_exclude_node_types),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model weights plus the architecture metadata needed to rebuild
        the model at evaluation time (v2 format)."""
        torch.save(
            {
                "format": self._CKPT_FORMAT,
                "state_dict": self.state_dict(),
                "arch": self.arch_metadata(),
            },
            path,
        )

    def load_checkpoint(self, path: str, device: Optional[str] = None) -> None:
        """Load model weights. Accepts both the v2 dict format (with arch
        metadata) and the legacy bare-state_dict format."""
        map_location = device or self.config.device
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)

    @staticmethod
    def read_arch_metadata(
        path: str, device: str = "cpu"
    ) -> Optional[Dict[str, object]]:
        """
        Return the architecture metadata stored in a v2 checkpoint, or None
        for a legacy bare-state_dict checkpoint (or on any read failure).

        Lets evaluate.py construct a model whose layer count / hidden dim
        match the trained weights even when the CLI flags differ from the
        training recipe — see the v2-format note above.
        """
        try:
            state = torch.load(path, map_location=device)
        except Exception as e:  # unreadable / not a torch file
            print(f"  [checkpoint] could not read arch metadata: {e}")
            return None
        if isinstance(state, dict) and state.get("format") == CI_RCT._CKPT_FORMAT:
            arch = state.get("arch")
            return arch if isinstance(arch, dict) else None
        return None
