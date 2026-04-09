"""
Centralised sys.path injection for CXGNN base model imports.

All CI-RCT modules that need CXGNN classes (CausalGraph, NNModel)
must import this module first:

    from _cxgnn_path import register_cxgnn_path
    register_cxgnn_path()
    from causal import CausalGraph
    from alg1 import NNModel

This isolates the fragile path manipulation to one place.
"""
import sys
import os

_CXGNN_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "CXGNN", "model")
)


def register_cxgnn_path() -> None:
    """Insert CXGNN model directory into sys.path (idempotent)."""
    if _CXGNN_MODEL_PATH not in sys.path:
        sys.path.insert(0, _CXGNN_MODEL_PATH)
