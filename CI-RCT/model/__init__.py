"""
CI-RCT model package.

Exports all four modules and their primary classes.
Imports are lazy to avoid forcing torch_geometric at import time
for tests that only need pure-Python components.
"""
from model.causal_adversarial_gan import CausalAdversarialGAN, CausalAdversarialGenerator
from model.causal_shapley import compute_asymmetric_causal_shapley, compute_shapley_edge_scores
from model.ci_rct import CI_RCT
from model.hetero_backbone import HeteroGNNBackbone
from model.hetero_ncm import HeteroNCM
from model.root_cause_tracer import RootCauseTracer
from model.typed_causal_graph import TypedCausalGraph

__all__ = [
    "CI_RCT",
    "HeteroGNNBackbone",
    "TypedCausalGraph",
    "HeteroNCM",
    "compute_asymmetric_causal_shapley",
    "compute_shapley_edge_scores",
    "RootCauseTracer",
    "CausalAdversarialGAN",
    "CausalAdversarialGenerator",
]
