"""Unit tests for evaluate.eval_classification_pooled (joint variant).

Locks in the per-head threshold-tuning fix: the two heads have different fraud
base rates, so a single shared argmax (0.5) cut lets the high-volume wallet
head over-predict and collapses the pooled fraud F1. Tuning a SEPARATE
threshold per head on its own val split must recover it — without touching the
already-calibrated transaction head.
"""
from __future__ import annotations

import math

import torch
from torch_geometric.data import HeteroData

from evaluate import eval_classification_pooled, _subset_match_meaningful


def _logits_for(prob1_per_node):
    """Build [N, 2] logits whose softmax class-1 prob equals each given value."""
    rows = [[0.0, math.log(p / (1.0 - p))] for p in prob1_per_node]
    return torch.tensor(rows, dtype=torch.float)


class _StubJoint:
    """Minimal model exposing the all_logits() contract used by the pooled eval."""

    def __init__(self, logits_by_type):
        self._logits_by_type = logits_by_type

    def eval(self):
        return self

    def all_logits(self, _data):
        return self._logits_by_type, {}


def _toy_data_and_model():
    """transaction head is well-separated; wallet head over-predicts at 0.5.

    Wallet licit nodes sit at prob1=0.6 (predicted fraud under argmax, but
    cleanly licit above a ~0.65 cut); wallet fraud nodes at 0.9.
    """
    data = HeteroData()
    # transaction: 4 nodes, clean separation (0.9 fraud / 0.1 licit)
    data["transaction"].y = torch.tensor([1, 1, 0, 0])
    tx_all = torch.ones(4, dtype=torch.bool)
    data["transaction"].val_mask = tx_all.clone()
    data["transaction"].test_mask = tx_all.clone()
    # wallet: 10 nodes, licit at 0.6 (argmax mislabels them fraud)
    data["wallet"].y = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    w_all = torch.ones(10, dtype=torch.bool)
    data["wallet"].val_mask = w_all.clone()
    data["wallet"].test_mask = w_all.clone()

    logits = {
        "transaction": _logits_for([0.9, 0.9, 0.1, 0.1]),
        "wallet": _logits_for([0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.6]),
    }
    return data, _StubJoint(logits)


def test_argmax_pooled_overpredicts():
    """Legacy argmax (no objective): the wallet head over-predicts fraud."""
    data, model = _toy_data_and_model()
    metrics, per_type_n, per_type_info = eval_classification_pooled(
        model, data, threshold_objective=None
    )
    assert per_type_n == {"transaction": 4, "wallet": 10}
    # Every wallet node crosses 0.5 → predicted fraud (5 false positives).
    assert per_type_info["wallet"]["pred_fraud_rate"] == 1.0
    assert per_type_info["wallet"]["threshold"] == 0.5
    # Transaction head is unaffected and perfect.
    assert per_type_info["transaction"]["fraud_f1"] == 1.0


def test_per_head_tuning_recovers_pooled_f1():
    """Per-head val-tuned thresholds fix the over-prediction and lift pooled F1."""
    data, model = _toy_data_and_model()
    argmax_m, _, _ = eval_classification_pooled(
        model, data, threshold_objective=None
    )
    tuned_m, _, tuned_info = eval_classification_pooled(
        model, data, threshold_objective="fraud_f1"
    )
    # The wallet head gets a cut above 0.5 and stops over-predicting.
    assert tuned_info["wallet"]["threshold"] > 0.5
    assert tuned_info["wallet"]["pred_fraud_rate"] < argmax_m["pred_fraud_rate"]
    # Pooled fraud F1 strictly improves (here to a perfect 1.0).
    assert tuned_m["fraud_f1"] > argmax_m["fraud_f1"]
    assert tuned_m["fraud_f1"] == 1.0
    # The already-calibrated transaction head is untouched.
    assert tuned_info["transaction"]["fraud_f1"] == 1.0


# ── Metric C: subset gt-match feasibility guard ─────────────────────────────

def test_subset_match_meaningful_for_precise_gt():
    """Strict-style GT (≤ chain length) → subset match is meaningful."""
    preds = [{1, 2, 3}, {4, 5, 6}, {7, 8}]      # chains of length 2-3
    gts = [{2}, {4, 5}, {7}]                      # 1-2 nodes, all fit
    meaningful, frac = _subset_match_meaningful(preds, gts)
    assert meaningful is True
    assert frac == 1.0


def test_subset_match_degenerate_for_broad_gt():
    """Extended-style GT (≫ chain length) → subset match is suppressed."""
    preds = [{1, 2}, {3, 4}, {5}]                 # short chains (1-2 nodes)
    gts = [set(range(100, 160)), set(range(200, 212)), set(range(300, 305))]
    meaningful, frac = _subset_match_meaningful(preds, gts)
    assert meaningful is False                    # no instance can fit its GT
    assert frac == 0.0


def test_subset_match_empty_inputs():
    """No non-empty (pred, gt) pairs → not meaningful, frac 0."""
    meaningful, frac = _subset_match_meaningful([set()], [set()])
    assert meaningful is False
    assert frac == 0.0
