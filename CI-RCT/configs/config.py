"""
CI-RCT hyperparameter configuration.

All model and training settings live here — nothing is hardcoded in model
files.  Treat config instances as immutable after construction (frozen=True).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CI_RCT_Config:
    # ── Dataset ───────────────────────────────────────────────────────────
    dataset: str = "elliptic++"
    data_root: str = "data"
    target_node_type: str = "author"
    hop_limit: int = 2            # BFS depth for causal graph construction
    node_limit: int = 500         # max nodes per causal subgraph

    # ── HGT Backbone ──────────────────────────────────────────────────────
    hidden_dim: int = 128
    num_heads: int = 4
    num_hgt_layers: int = 3
    dropout: float = 0.3

    # ── HeteroNCM ─────────────────────────────────────────────────────────
    node_type_emb_dim: int = 16   # dimension of type embedding vectors
    ncm_h_size: int = 64
    ncm_h_layers: int = 2
    # Null-intervention baseline for CE = p_actual − p_null.
    #   "zero"      : do(h_u = 0) zero vector. Legacy default; byte-identical to
    #                 the original NCM. The zero vector is out-of-distribution,
    #                 so p_null saturates (Elliptic++ wallet→tx: p_null≈0.97),
    #                 which pushes CE systematically negative and makes the SIGN
    #                 uninterpretable (see memory ce-null-baseline-artifact).
    #   "type_mean" : do(h_u = E[h_type]) — replace the source node with the mean
    #                 embedding of its node type (in-distribution interventional
    #                 baseline). Recentres CE at 0 so sign = promote(+)/suppress(−).
    #   "marginal"  : p_null = E[MLP(h)] over same-type sources — the marginal
    #                 null intervention do(h_u ~ P(h_type)). Removes type_mean's
    #                 Jensen gap (MLP(E[h]) ≠ E[MLP(h)]), so E[CE] over a type
    #                 is exactly 0. Inference-only, like type_mean: no retrain.
    ncm_baseline: str = "zero"

    # ── Root Cause Tracer ─────────────────────────────────────────────────
    max_hops: int = 5
    ce_threshold: float = 0.1
    top_k_paths: int = 3
    # Backward-search algorithm ablation. "greedy" (default) is byte-identical
    # to the legacy tracer; {dag_dp, dijkstra, bfs, dfs, beam} are comparison
    # arms (see model/tracer_strategies and tracer_ablation_plan.md).
    tracer_algorithm: str = "greedy"
    tracer_objective: str = "product"  # weighted-path objective: product|sum
    ce_eps: float = 1e-12              # clamp for -log|CE| in the product objective
    # Std of the Gaussian noise injected into node embeddings when measuring
    # φ-stability (input-perturbation robustness of the attribution) at eval time.
    phi_stability_noise_std: float = 0.01

    # ── CausalAdversarialGAN ──────────────────────────────────────────────
    gan_hidden_dim: int = 128     # Generator hidden dimension
    noise_std: float = 0.05       # Gaussian noise std for generation diversity
    gp_weight: float = 10.0       # WGAN-GP gradient penalty coefficient λ
    n_critic: int = 5             # Discriminator steps per Generator step

    # ── Joint Loss Weights ─────────────────────────────────────────────────
    # L_total = L_detection + λ1 · L_adversarial + λ2 · L_causal
    lambda_adversarial: float = 0.1   # λ1: weight of WGAN-GP adversarial loss
    lambda_stability: float = 0.5     # λ2: weight of Causal Shapley stability loss
    lambda_ncm: float = 0.3           # λ3: weight of NCM supervision (BCE) loss
    lambda_recon: float = 0.0         # λ4: GraphBEAN-style feature+edge reconstruction
                                      #     self-supervision (0 = OFF → byte-identical;
                                      #     enabled via --use_reconstruction for
                                      #     wallet/joint to train the unlabeled majority)

    # ── Training ──────────────────────────────────────────────────────────
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # ── Misc ──────────────────────────────────────────────────────────────
    device: str = "cuda"  # "cuda" or "cpu"
    eval_every: int = 10
    checkpoint_dir: str = "checkpoints"
    seed: int = 42

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
