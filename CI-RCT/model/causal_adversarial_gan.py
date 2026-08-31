"""
CausalAdversarialGAN — Module 4 of CI-RCT.

Adversarial GAN for handling:
  - Gap 2: Class imbalance (fraud nodes ≪ normal nodes) — the generator
    augments the scarce fraud side with hard, hard-to-classify fraud-like
    samples near the real-fraud manifold.

Architecture
────────────
Generator:
  Conditioned on real fraud node features + type embedding (+ small
  Gaussian noise), generates hard fraud-like feature vectors that stay
  near the real-fraud distribution.
  NOTE: forward() also exposes an OPTIONAL edge generator restricted to
  topologically-prior upstream nodes, but the training pipeline
  (ci_rct.py) calls generate() WITHOUT upstream_features, so no
  edge/topology conditioning is exercised — only feature synthesis is used
  and the returned edge_probs are discarded.

Discriminator (reuses detector heads, NOT the full backbone):
  The WGAN score reuses only the backbone's input projection + classifier
  head — the HGT message-passing layers are skipped — applied to raw
  target-type features.  Adversarial training therefore strengthens the
  projection and classification layers shared with the fraud detector,
  not the full detector forward pass.

Training (WGAN-GP):
  Wasserstein distance with gradient penalty (Arjovsky et al., ICML 2017).
  Stable gradients for high-dimensional heterogeneous graph features.

Joint loss:
  L_total = L_detection + λ1 · L_adversarial + λ2 · L_causal
  where L_adversarial is the WGAN-GP objective.

Reference: CI-RCT_Thesis_Plan.md § 5.5, § 5.6
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Generator ─────────────────────────────────────────────────────────────────

class CausalAdversarialGenerator(nn.Module):
    """
    Hard fraud-like sample generator.

    Feature generation (the path actually used in training):
        Conditioned on real fraud node features + type embedding + small
        Gaussian noise, generates hard fraud-like feature vectors that
        stay near the real-fraud distribution.

    Structural constraint (DAG topology) — OPTIONAL, not exercised:
        forward() can also score connections to topologically-prior
        upstream nodes (nodes appearing *before* the sample in topological
        order), preserving temporal causal precedence.  This edge path
        only runs when upstream_features is supplied; the training
        pipeline never supplies it, so these edge_probs are placeholders
        and are discarded by the caller.

    Args:
        node_feature_dim: Raw feature dimension of fraud nodes
        type_emb_dim:     Node type embedding dimension
        hidden_dim:       Hidden dimension for both sub-networks
        noise_std:        Std of Gaussian noise injected for diversity
    """

    def __init__(
        self,
        node_feature_dim: int,
        type_emb_dim: int,
        hidden_dim: int = 128,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise_std = noise_std

        # Feature generator: fraud features + type emb → camouflaged features
        self.feature_gen = nn.Sequential(
            nn.Linear(node_feature_dim + type_emb_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, node_feature_dim),
            nn.Tanh(),  # normalise output to [-1, 1]
        )

        # Edge generator: decides connection probability to each upstream node
        # Input: [fake_node_feature ‖ upstream_node_feature]
        self.edge_gen = nn.Sequential(
            nn.Linear(node_feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        fraud_features: Tensor,
        type_emb: Tensor,
        topo_order: List,
        upstream_features: Optional[Dict[int, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        """
        Generate camouflaged fraud features + connection probabilities.

        Args:
            fraud_features:    Real fraud node feature(s)  [B, D] or [D]
            type_emb:          Node type embedding  [type_emb_dim] or [B, T]
            topo_order:        TypedCausalGraph.topological_order() result
            upstream_features: Optional {node_id: feature_tensor} for edge gen

        Returns:
            fake_features:  Camouflaged node features  [B, D]
            edge_probs:     {upstream_node_id: connection_probability_tensor}
        """
        if fraud_features.dim() == 1:
            fraud_features = fraud_features.unsqueeze(0)
        if type_emb.dim() == 1:
            type_emb = type_emb.unsqueeze(0).expand(fraud_features.size(0), -1)

        # Add Gaussian noise for generation diversity
        noise = torch.randn_like(fraud_features) * self.noise_std
        condition = torch.cat([fraud_features + noise, type_emb], dim=-1)
        fake_features = self.feature_gen(condition)  # [B, D]

        # Structural constraint: only connect to topologically earlier nodes
        # (nodes that appear before the fake node in causal ordering)
        valid_upstream_count = max(1, len(topo_order) // 2)
        valid_upstream = topo_order[:valid_upstream_count]

        edge_probs: Dict[int, Tensor] = {}
        if upstream_features is not None:
            for node_id in valid_upstream:
                if node_id not in upstream_features:
                    continue
                up_feat = upstream_features[node_id]
                if up_feat.dim() == 1:
                    up_feat = up_feat.unsqueeze(0).expand(fake_features.size(0), -1)
                pair = torch.cat([fake_features, up_feat], dim=-1)
                edge_probs[node_id] = self.edge_gen(pair)  # [B, 1]
        else:
            # Placeholder: uniform connection probabilities
            for node_id in valid_upstream:
                edge_probs[node_id] = torch.full(
                    (fake_features.size(0), 1), 0.5,
                    device=fake_features.device,
                )

        return fake_features, edge_probs


# ── WGAN-GP utilities ─────────────────────────────────────────────────────────

def compute_gradient_penalty(
    discriminator_fn,
    real_features: Tensor,
    fake_features: Tensor,
    device: torch.device,
    gp_weight: float = 10.0,
) -> Tensor:
    """
    Compute WGAN gradient penalty (Gulrajani et al., NeurIPS 2017).

    Enforces the Lipschitz constraint on the discriminator by penalising
    gradients that deviate from 1 at interpolated samples.

    Args:
        discriminator_fn: Callable that maps features → scalar scores
        real_features:    Real node features  [B, D]
        fake_features:    Generated features  [B, D]
        device:           Torch device
        gp_weight:        λ for gradient penalty (default: 10)

    Returns:
        Scalar gradient penalty tensor (with gradient for optimiser)
    """
    batch_size = real_features.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_features)

    interpolated = (alpha * real_features + (1 - alpha) * fake_features).requires_grad_(True)
    d_interpolated = discriminator_fn(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = gp_weight * ((gradient_norm - 1) ** 2).mean()
    return penalty


# ── CausalAdversarialGAN ──────────────────────────────────────────────────────

class CausalAdversarialGAN(nn.Module):
    """
    Full Causal Adversarial GAN wrapper.

    Manages the Generator and provides utilities for WGAN-GP training.
    The discriminator score reuses the backbone's input projection +
    classifier head (see discriminator_fn in ci_rct.py) — not the full
    HGT backbone — so adversarial play strengthens the projection and
    classification layers shared with the fraud detector.

    Semantic mapping:
        Real world:   Fraudster (attacker)   ← →  Detector (defender)
        CI-RCT:       Generator              ← →  Discriminator (reuses
                                                   detector proj + head)

    Args:
        node_feature_dim: Fraud node feature dimension
        type_emb_dim:     Node type embedding dimension
        hidden_dim:       Generator hidden dimension
        noise_std:        Generator noise standard deviation
        gp_weight:        WGAN gradient penalty coefficient λ
    """

    def __init__(
        self,
        node_feature_dim: int,
        type_emb_dim: int,
        hidden_dim: int = 128,
        noise_std: float = 0.05,
        gp_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.gp_weight = gp_weight

        self.generator = CausalAdversarialGenerator(
            node_feature_dim=node_feature_dim,
            type_emb_dim=type_emb_dim,
            hidden_dim=hidden_dim,
            noise_std=noise_std,
        )

    # ── Generator step ─────────────────────────────────────────────────────────

    def generate(
        self,
        fraud_features: Tensor,
        type_emb: Tensor,
        topo_order: List,
        upstream_features: Optional[Dict[int, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        """Convenience wrapper for Generator.forward()."""
        return self.generator(fraud_features, type_emb, topo_order, upstream_features)

    # ── Discriminator loss (WGAN-GP critic loss) ───────────────────────────────

    def discriminator_loss(
        self,
        discriminator_fn,
        real_scores: Tensor,
        fake_features: Tensor,
        real_features: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        WGAN-GP discriminator (critic) loss.

        L_D value = E[D(fake)] − E[D(real)] + GP

        NOTE on gradients: fake_scores is computed under torch.no_grad()
        (and on fake_features.detach()), so the E[D(fake)] term is a
        constant w.r.t. the critic parameters.  The critic update gradient
        therefore comes ONLY from −E[D(real)] (raising real-fraud scores)
        plus the gradient penalty — it does not push fake scores down.
        The reported loss value still equals the two-sided expression above.

        Args:
            discriminator_fn: Callable features → scores (for GP computation)
            real_scores:  D(real_features)  [B]  (here real = real fraud features)
            fake_features: Generated features [B, D]
            real_features: Real fraud features [B, D]
            device: torch device

        Returns:
            Scalar critic loss
        """
        fake_scores = discriminator_fn(fake_features.detach())
        wasserstein_dist = fake_scores.mean() - real_scores.mean()
        gp = compute_gradient_penalty(
            discriminator_fn, real_features, fake_features.detach(), device, self.gp_weight
        )
        return wasserstein_dist + gp

    # ── Generator loss (adversarial loss term in L_total) ──────────────────────

    def generator_loss(self, discriminator_fn, fake_features: Tensor) -> Tensor:
        """
        WGAN generator loss.

        L_G = −E[D(fake)]   (generator tries to maximise discriminator score)

        Args:
            discriminator_fn: Callable features → scores
            fake_features:    Generated features  [B, D]

        Returns:
            Scalar generator loss (to be added as L_adversarial in L_total)
        """
        fake_scores = discriminator_fn(fake_features)
        return -fake_scores.mean()
