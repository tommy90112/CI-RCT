"""
Unit tests for utils/metrics.py.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import (
    compute_f1,
    compute_auc,
    compute_classification_metrics,
    explanation_accuracy,
    explanation_recall,
    groundtruth_match,
    compute_explanation_metrics,
    root_cause_precision,
    causal_chain_validity,
    mean_tracing_depth,
    compute_root_cause_metrics,
)


class TestClassificationMetrics:
    def test_perfect_f1(self):
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([0, 1, 0, 1])
        assert compute_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_random_f1_in_range(self):
        y_true = torch.tensor([0, 1, 1, 0])
        y_pred = torch.tensor([1, 0, 1, 0])
        f1 = compute_f1(y_true, y_pred)
        assert 0.0 <= f1 <= 1.0

    def test_auc_perfect(self):
        y_true = torch.tensor([0, 0, 1, 1])
        y_scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_scores) == pytest.approx(1.0)

    def test_auc_single_class_returns_zero(self):
        y_true = torch.tensor([1, 1, 1])
        y_scores = torch.tensor([0.8, 0.9, 0.7])
        assert compute_auc(y_true, y_scores) == 0.0

    def test_shape_mismatch_raises(self):
        y_true = torch.tensor([0, 1])
        y_pred = torch.tensor([0, 1, 0])
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_f1(y_true, y_pred)

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError):
            compute_f1([0, 1], torch.tensor([0, 1]))


class TestExplanationMetrics:
    def test_perfect_accuracy(self):
        assert explanation_accuracy({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)

    def test_partial_accuracy(self):
        assert explanation_accuracy({1, 2, 3, 4}, {1, 2}) == pytest.approx(0.5)

    def test_empty_pred_raises(self):
        with pytest.raises(ValueError, match="pred_nodes"):
            explanation_accuracy(set(), {1, 2})

    def test_perfect_recall(self):
        assert explanation_recall({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)

    def test_partial_recall(self):
        assert explanation_recall({1}, {1, 2}) == pytest.approx(0.5)

    def test_empty_gt_raises(self):
        with pytest.raises(ValueError, match="gt_nodes"):
            explanation_recall({1, 2}, set())

    def test_exact_match_true(self):
        assert groundtruth_match({1, 2, 3}, {1, 2, 3}) == 1

    def test_exact_match_false(self):
        assert groundtruth_match({1, 2}, {1, 2, 3}) == 0

    # ── subset mode (chain-vs-list explanations) ────────────────────────────
    def test_subset_match_when_gt_fully_recovered(self):
        # Chain contains the query tx (99) + intermediate + all GT wallets.
        chain, gt = {99, 7, 1, 2}, {1, 2}
        assert groundtruth_match(chain, gt, mode="subset") == 1
        # The CXGNN exact metric is 0 for the very same pair (this is the bug).
        assert groundtruth_match(chain, gt, mode="exact") == 0

    def test_subset_match_false_when_gt_partially_missed(self):
        assert groundtruth_match({99, 1}, {1, 2}, mode="subset") == 0

    def test_subset_empty_gt_is_not_a_match(self):
        # Empty GT must not count as a trivial subset hit.
        assert groundtruth_match({1, 2}, set(), mode="subset") == 0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            groundtruth_match({1}, {1}, mode="superset")

    def test_aggregate_subset_mode(self):
        # Both predictions hold the query node (90/91) + recover their GT.
        preds = [{90, 1, 2}, {91, 3}]
        gts = [{1, 2}, {3, 5}]   # 2nd misses wallet 5
        metrics = compute_explanation_metrics(preds, gts, gt_match_mode="subset")
        assert metrics["gt_match_mode"] == "subset"
        assert metrics["gt_match_accuracy"] == pytest.approx(0.5)  # 1 of 2

    def test_aggregate_metrics(self):
        preds = [{1, 2}, {3, 4}]
        gts = [{1, 2}, {3, 5}]
        metrics = compute_explanation_metrics(preds, gts)
        assert "explanation_accuracy" in metrics
        assert "explanation_recall" in metrics
        assert "gt_match_accuracy" in metrics
        assert metrics["gt_match_accuracy"] == pytest.approx(0.5)

    def test_aggregate_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_explanation_metrics([{1}], [{2}, {3}])

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError):
            compute_explanation_metrics([], [])


class TestRootCauseMetrics:
    fraud_set = {1, 3, 5}

    def test_root_cause_precision_correct(self):
        assert root_cause_precision(1, self.fraud_set) == 1

    def test_root_cause_precision_wrong(self):
        assert root_cause_precision(2, self.fraud_set) == 0

    def test_chain_validity_full(self):
        assert causal_chain_validity([1, 3, 5], self.fraud_set) == pytest.approx(1.0)

    def test_chain_validity_partial(self):
        assert causal_chain_validity([1, 2, 5], self.fraud_set) == pytest.approx(2 / 3)

    def test_chain_validity_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            causal_chain_validity([], self.fraud_set)

    def test_mean_tracing_depth(self):
        chains = [[0, 1, 2], [0, 1]]
        assert mean_tracing_depth(chains) == pytest.approx(1.5)

    def test_mean_tracing_depth_single_node(self):
        assert mean_tracing_depth([[0]]) == pytest.approx(0.0)

    def test_mean_tracing_depth_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mean_tracing_depth([])

    def test_compute_root_cause_metrics_full(self):
        roots = [1, 2]
        chains = [[0, 1], [0, 2, 3]]
        metrics = compute_root_cause_metrics(roots, chains, self.fraud_set)
        assert metrics["root_cause_precision"] == pytest.approx(0.5)
        assert "chain_validity" in metrics
        assert metrics["mean_tracing_depth"] == pytest.approx(1.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_root_cause_metrics([1], [[0, 1], [0, 2]], self.fraud_set)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_root_cause_metrics([], [], self.fraud_set)
