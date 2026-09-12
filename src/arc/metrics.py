"""ARC scoring: exact match, cell accuracy, pass@2."""

from __future__ import annotations

import numpy as np


def exact_match(pred: np.ndarray, target: np.ndarray) -> bool:
    pred = np.asarray(pred)
    target = np.asarray(target)
    return pred.shape == target.shape and np.array_equal(pred, target)


def shape_match(pred: np.ndarray, target: np.ndarray) -> bool:
    return tuple(np.asarray(pred).shape) == tuple(np.asarray(target).shape)


def cell_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        return 0.0
    return float((pred == target).mean())


def pass_at_2(preds: list[np.ndarray], target: np.ndarray) -> bool:
    return any(exact_match(p, target) for p in preds[:2])


def score_tasks(
    predictions: dict[str, list[list[np.ndarray]]],
    targets: dict[str, list[np.ndarray]],
) -> dict[str, float]:
    """predictions[task_id] = list over test inputs of up to 2 grids."""
    n = 0
    hits = 0
    shape_hits = 0
    cell_sum = 0.0
    for task_id, golds in targets.items():
        preds_task = predictions.get(task_id, [])
        for i, gold in enumerate(golds):
            n += 1
            cands = preds_task[i] if i < len(preds_task) else []
            if pass_at_2(cands, gold):
                hits += 1
            best = cands[0] if cands else np.zeros_like(gold)
            if shape_match(best, gold):
                shape_hits += 1
            cell_sum += cell_accuracy(best, gold)
    if n == 0:
        return {"n": 0, "pass_at_2": 0.0, "shape": 0.0, "cell": 0.0}
    return {
        "n": n,
        "pass_at_2": hits / n,
        "shape": shape_hits / n,
        "cell": cell_sum / n,
    }
