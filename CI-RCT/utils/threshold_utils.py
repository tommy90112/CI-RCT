"""
threshold_utils.py — decision-threshold tuning for imbalanced binary detection.

argmax (i.e. a fixed 0.5 cut on the class-1 probability) is the implicit
operating point used by `argmax(dim=-1)`. Under class imbalance this is
rarely the F1-optimal point: the model can have good ranking (high AUC)
yet a poor argmax-F1 because the optimal cut sits far from 0.5.

These helpers sweep a threshold grid on a HELD-OUT split (validation) to
pick the operating point that maximises a chosen objective, then that same
threshold is applied to the test split. Selecting on val (never on test)
avoids leaking the test distribution into the threshold choice.

Pure functions — no mutation of inputs; inputs are read-only numpy/array-like.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# 0.05 … 0.95 inclusive, rounded to avoid float drift in printed thresholds.
DEFAULT_GRID = tuple(round(x, 4) for x in np.arange(0.05, 0.96, 0.05))

_VALID_OBJECTIVES = ("macro_f1", "fraud_f1")


def _f1_at(scores: np.ndarray, y_true: np.ndarray, thr: float, objective: str) -> float:
    from sklearn.metrics import f1_score

    preds = (scores > thr).astype(int)
    if objective == "fraud_f1":
        return float(f1_score(y_true, preds, average="binary", pos_label=1,
                              zero_division=0))
    return float(f1_score(y_true, preds, average="macro", zero_division=0))


def sweep_best_threshold(
    scores: Sequence[float],
    y_true: Sequence[int],
    objective: str = "macro_f1",
    grid: Sequence[float] | None = None,
) -> Tuple[float, float]:
    """
    Find the threshold in `grid` that maximises `objective` on (scores, y_true).

    Args:
        scores:    class-1 probabilities, shape [N].
        y_true:    ground-truth 0/1 labels, shape [N].
        objective: 'macro_f1' (default, matches the reported metric) or
                   'fraud_f1' (binary F1 of the positive/fraud class).
        grid:      iterable of candidate thresholds; defaults to 0.05…0.95.

    Returns:
        (best_threshold, best_objective_value). Falls back to (0.5, value@0.5)
        when no candidate beats it.

    Raises:
        ValueError: on unknown objective or empty / mismatched inputs.
    """
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}"
        )
    scores_arr = np.asarray(scores, dtype=float).ravel()
    y_arr = np.asarray(y_true, dtype=int).ravel()
    if scores_arr.size == 0 or scores_arr.shape != y_arr.shape:
        raise ValueError(
            f"scores and y_true must be non-empty and same length; "
            f"got {scores_arr.shape} vs {y_arr.shape}"
        )

    candidate_grid = DEFAULT_GRID if grid is None else tuple(grid)
    best_thr, best_val = 0.5, _f1_at(scores_arr, y_arr, 0.5, objective)
    for thr in candidate_grid:
        val = _f1_at(scores_arr, y_arr, float(thr), objective)
        if val > best_val:
            best_val, best_thr = val, float(thr)
    return best_thr, best_val


def predict_at_threshold(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Return 0/1 predictions: 1 where class-1 prob > threshold."""
    return (np.asarray(scores, dtype=float).ravel() > float(threshold)).astype(int)
