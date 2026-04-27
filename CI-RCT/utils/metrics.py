"""
Evaluation metrics for CI-RCT.

Three metric categories:
  1. Classification metrics:     F1-macro, AUC-ROC
  2. Explanation quality:        accuracy, recall, exact ground-truth match
  3. Root cause tracing quality: precision, chain validity, mean tracing depth

All functions are pure (no side effects, no mutation of inputs).
Inputs are validated and ValueError is raised on bad inputs.
"""
from typing import List, Set

import numpy as np
import torch
from torch import Tensor

try:
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError as exc:
    raise ImportError("scikit-learn is required for CI-RCT metrics.") from exc


# ── Classification ─────────────────────────────────────────────────────────────

def compute_f1(y_true: Tensor, y_pred: Tensor, average: str = "macro") -> float:
    """
    Macro F1 score for node classification.

    Args:
        y_true: Ground-truth labels [N]
        y_pred: Predicted labels [N]
        average: sklearn averaging strategy

    Returns:
        float: F1 score in [0, 1]
    """
    _validate_1d_tensors(y_true, y_pred, "compute_f1")
    return float(
        f1_score(y_true.numpy(), y_pred.numpy(), average=average, zero_division=0)
    )


def compute_auc(y_true: Tensor, y_scores: Tensor) -> float:
    """
    AUC-ROC score. Supports binary and multi-class (OvR macro).

    Args:
        y_true:   Ground-truth labels [N]
        y_scores: Class-1 probabilities [N] for binary,
                  or full probability matrix [N, C] for multi-class

    Returns:
        float: AUC in [0, 1]; 0.0 if only one class present.
    """
    if y_scores.dim() not in (1, 2):
        raise ValueError("compute_auc: y_scores must be 1-D or 2-D.")
    if y_true.dim() != 1:
        raise ValueError("compute_auc: y_true must be 1-D.")
    try:
        y_true_np = y_true.numpy()
        y_scores_np = y_scores.numpy()
        if y_scores.dim() == 2:
            result = roc_auc_score(
                y_true_np, y_scores_np, multi_class="ovr", average="macro"
            )
        else:
            result = roc_auc_score(y_true_np, y_scores_np)
        return 0.0 if result != result else float(result)  # guard against nan
    except ValueError:
        return 0.0  # single-class or unsupported edge case


def compute_classification_metrics(
    y_pred: Tensor,
    y_true: Tensor,
    y_scores: Tensor | None = None,
) -> dict:
    """
    Compute all classification metrics in one call.

    Args:
        y_pred:   Predicted class indices [N]
        y_true:   Ground-truth labels [N]
        y_scores: Class-1 probabilities [N] for binary, or [N, C] for multi-class

    Returns:
        dict: {'f1': float, 'auc': float}
    """
    metrics = {"f1": compute_f1(y_true, y_pred)}
    if y_scores is not None:
        metrics["auc"] = compute_auc(y_true, y_scores)
    else:
        metrics["auc"] = 0.0
    return metrics


# ── Explanation quality ────────────────────────────────────────────────────────

def explanation_accuracy(pred_nodes: Set, gt_nodes: Set) -> float:
    """
    Fraction of predicted explanation nodes that are in the ground truth.

    Args:
        pred_nodes: Predicted explanation node set
        gt_nodes:   Ground-truth causal node set

    Returns:
        float: |pred ∩ gt| / |pred|  in [0, 1]
    """
    _validate_sets(pred_nodes, gt_nodes, "explanation_accuracy")
    if not pred_nodes:
        raise ValueError("pred_nodes must not be empty.")
    return len(pred_nodes & gt_nodes) / len(pred_nodes)


def explanation_recall(pred_nodes: Set, gt_nodes: Set) -> float:
    """
    Fraction of ground-truth nodes covered by the predicted explanation.

    Args:
        pred_nodes: Predicted explanation node set
        gt_nodes:   Ground-truth causal node set

    Returns:
        float: |pred ∩ gt| / |gt|  in [0, 1]
    """
    _validate_sets(pred_nodes, gt_nodes, "explanation_recall")
    if not gt_nodes:
        raise ValueError("gt_nodes must not be empty.")
    return len(pred_nodes & gt_nodes) / len(gt_nodes)


def groundtruth_match(pred_nodes: Set, gt_nodes: Set) -> int:
    """
    Whether the predicted explanation exactly matches the ground truth.

    The strictest metric from the CXGNN paper (Groundtruth Match Accuracy).

    Returns:
        int: 1 if pred_nodes == gt_nodes, else 0
    """
    _validate_sets(pred_nodes, gt_nodes, "groundtruth_match")
    return int(pred_nodes == gt_nodes)


def compute_explanation_metrics(
    predicted_list: List[Set],
    ground_truth_list: List[Set],
) -> dict:
    """
    Aggregate explanation metrics over a list of (prediction, ground-truth) pairs.

    Args:
        predicted_list:    List of predicted node sets
        ground_truth_list: List of ground-truth node sets (same length)

    Returns:
        dict: {'explanation_accuracy': float, 'explanation_recall': float,
               'gt_match_accuracy': float}
    """
    if len(predicted_list) != len(ground_truth_list):
        raise ValueError(
            f"predicted_list and ground_truth_list must have equal length, "
            f"got {len(predicted_list)} vs {len(ground_truth_list)}."
        )
    if not predicted_list:
        raise ValueError("Input lists must not be empty.")

    accuracies = [
        explanation_accuracy(p, g)
        for p, g in zip(predicted_list, ground_truth_list)
        if p  # skip empty predictions
    ]
    recalls = [
        explanation_recall(p, g)
        for p, g in zip(predicted_list, ground_truth_list)
        if g  # skip empty ground truths
    ]
    gt_matches = [
        groundtruth_match(p, g)
        for p, g in zip(predicted_list, ground_truth_list)
    ]

    return {
        "explanation_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "explanation_recall": float(np.mean(recalls)) if recalls else 0.0,
        "gt_match_accuracy": float(np.mean(gt_matches)),
    }


# ── Root cause tracing quality ─────────────────────────────────────────────────

def root_cause_precision(pred_root, fraud_node_set: Set) -> int:
    """
    Whether the predicted root cause is truly a fraudulent node.

    Args:
        pred_root:      Predicted root cause node ID
        fraud_node_set: Set of ground-truth fraudulent node IDs

    Returns:
        int: 1 if pred_root ∈ fraud_node_set, else 0
    """
    if fraud_node_set is None:
        raise ValueError("fraud_node_set must not be None.")
    return int(pred_root in fraud_node_set)


def causal_chain_validity(causal_chain: List, fraud_node_set: Set) -> float:
    """
    Fraction of nodes in a single causal chain that are fraud-related.

    Args:
        causal_chain:   List of node IDs from target to root
        fraud_node_set: Set of ground-truth fraudulent node IDs

    Returns:
        float: |chain ∩ fraud| / |chain| in [0, 1]
    """
    if not causal_chain:
        raise ValueError("causal_chain must not be empty.")
    if fraud_node_set is None:
        raise ValueError("fraud_node_set must not be None.")
    fraud_in_chain = sum(1 for n in causal_chain if n in fraud_node_set)
    return fraud_in_chain / len(causal_chain)


def mean_tracing_depth(causal_chains: List[List]) -> float:
    """
    Average number of hops across multiple causal chains.

    Depth = len(chain) - 1 (number of edges, not nodes).

    Args:
        causal_chains: List of causal paths (each a list of node IDs)

    Returns:
        float: Mean depth across all chains
    """
    if not causal_chains:
        raise ValueError("causal_chains must not be empty.")
    depths = [max(len(chain) - 1, 0) for chain in causal_chains if chain]
    return float(np.mean(depths)) if depths else 0.0

def root_cause_hit_rate(causal_chains: List[List], fraud_node_set: Set) -> float:
    """
    Fraction of causal chains that contain AT LEAST ONE node from the
    fraud-related node set.

    This metric reflects the practical use case of root-cause tracing:
    investigators care whether *some* node in the traced chain is a known
    fraud-related entity, not whether *every* node is.  It complements:
      - root_cause_precision (only the chain's terminal node)
      - causal_chain_validity (proportion of fraud-related nodes per chain)

    Args:
        causal_chains:  List of node-ID lists (each a chain from target to root)
        fraud_node_set: Set of fraud-related global node IDs

    Returns:
        float: |chains with ≥1 fraud node| / |chains|
    """
    if not causal_chains:
        raise ValueError("causal_chains must not be empty.")
    if fraud_node_set is None:
        raise ValueError("fraud_node_set must not be None.")

    n_hits = sum(
        1 for chain in causal_chains
        if any(n in fraud_node_set for n in chain)
    )
    return n_hits / len(causal_chains)

def compute_root_cause_metrics(
    predicted_roots: List,
    causal_chains: List[List],
    fraud_node_set: Set,
) -> dict:
    """
    Aggregate all root cause tracing metrics.

    Args:
        predicted_roots: List of predicted root cause node IDs
        causal_chains:   Corresponding causal chains
        fraud_node_set:  Ground-truth fraudulent nodes

    Returns:
        dict: {'root_cause_precision': float, 'chain_validity': float,
               'mean_tracing_depth': float}
    """
    if len(predicted_roots) != len(causal_chains):
        raise ValueError(
            f"predicted_roots and causal_chains must have equal length, "
            f"got {len(predicted_roots)} vs {len(causal_chains)}."
        )
    if not predicted_roots:
        raise ValueError("predicted_roots must not be empty.")

    precisions = [root_cause_precision(r, fraud_node_set) for r in predicted_roots]
    validities = [
        causal_chain_validity(chain, fraud_node_set)
        for chain in causal_chains
        if chain
    ]

    return {
        "root_cause_precision": float(np.mean(precisions)),
        "root_cause_hit_rate":  root_cause_hit_rate(causal_chains, fraud_node_set),
        "chain_validity": float(np.mean(validities)) if validities else 0.0,
        "mean_tracing_depth": mean_tracing_depth(causal_chains),
    }


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_1d_tensors(a: Tensor, b: Tensor, caller: str) -> None:
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError(f"{caller}: inputs must be torch.Tensor.")
    if a.dim() != 1 or b.dim() != 1:
        raise ValueError(f"{caller}: inputs must be 1-D tensors.")
    if a.shape != b.shape:
        raise ValueError(
            f"{caller}: shape mismatch — got {a.shape} vs {b.shape}."
        )


def _validate_sets(a: Set, b: Set, caller: str) -> None:
    if not isinstance(a, set) or not isinstance(b, set):
        raise TypeError(f"{caller}: inputs must be Python sets.")
