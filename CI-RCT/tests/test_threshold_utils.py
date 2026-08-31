"""Unit tests for utils.threshold_utils and metrics.compute_fraud_f1."""
import numpy as np
import pytest
import torch

from utils.metrics import compute_fraud_f1
from utils.threshold_utils import (
    DEFAULT_GRID,
    predict_at_threshold,
    sweep_best_threshold,
)


def test_sweep_beats_default_cut_under_imbalance():
    # Fraud (label 1) scores cluster around 0.3; licit (0) around 0.1.
    # argmax (0.5) would call everything licit → fraud F1 = 0. A lower
    # threshold recovers the fraud class.
    rng = np.random.default_rng(0)
    fraud_scores = rng.uniform(0.25, 0.45, size=20)
    licit_scores = rng.uniform(0.0, 0.15, size=200)
    scores = np.concatenate([fraud_scores, licit_scores])
    y_true = np.concatenate([np.ones(20), np.zeros(200)]).astype(int)

    thr, val = sweep_best_threshold(scores, y_true, objective="fraud_f1")
    assert thr < 0.5                      # optimal cut is below the 0.5 default
    assert val == pytest.approx(1.0, abs=1e-6)   # cleanly separable → F1 = 1


def test_sweep_macro_objective_returns_grid_member_or_half():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    y_true = np.array([1, 1, 0, 0])
    thr, val = sweep_best_threshold(scores, y_true, objective="macro_f1")
    assert thr == 0.5 or thr in DEFAULT_GRID
    assert val == pytest.approx(1.0)


def test_predict_at_threshold():
    scores = [0.6, 0.4, 0.55, 0.49]
    np.testing.assert_array_equal(
        predict_at_threshold(scores, 0.5), np.array([1, 0, 1, 0])
    )


def test_invalid_objective_raises():
    with pytest.raises(ValueError):
        sweep_best_threshold([0.5], [1], objective="accuracy")


def test_mismatched_or_empty_inputs_raise():
    with pytest.raises(ValueError):
        sweep_best_threshold([], [])
    with pytest.raises(ValueError):
        sweep_best_threshold([0.5, 0.6], [1])


def test_compute_fraud_f1_matches_manual():
    # tp=2, fp=1, fn=1 → precision=2/3, recall=2/3, F1=2/3
    y_true = torch.tensor([1, 1, 1, 0])
    y_pred = torch.tensor([1, 1, 0, 1])
    assert compute_fraud_f1(y_true, y_pred) == pytest.approx(2 / 3, abs=1e-6)


def test_compute_fraud_f1_ignores_licit_perfection():
    # All licit correct, all fraud missed → fraud F1 = 0 (macro would be 0.5).
    y_true = torch.tensor([0, 0, 0, 1, 1])
    y_pred = torch.tensor([0, 0, 0, 0, 0])
    assert compute_fraud_f1(y_true, y_pred) == 0.0
