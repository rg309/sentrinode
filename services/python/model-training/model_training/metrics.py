from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class EvaluationMetrics:
    rmse: float
    mae: float
    breach_accuracy: float
    true_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "breach_accuracy": self.breach_accuracy,
            "true_positive_rate": self.true_positive_rate,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": int(self.true_positives),
            "true_negatives": int(self.true_negatives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
        }


def regression_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def regression_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - targets)))


def classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predicted_breach = predictions > threshold
    actual_breach = targets > threshold

    tp = np.logical_and(predicted_breach, actual_breach).sum()
    tn = np.logical_and(~predicted_breach, ~actual_breach).sum()
    fp = np.logical_and(predicted_breach, ~actual_breach).sum()
    fn = np.logical_and(~predicted_breach, actual_breach).sum()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    tpr = tp / max(tp + fn, 1)  # sensitivity
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "accuracy": float(accuracy),
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    latency_threshold: float,
) -> EvaluationMetrics:
    regression = {
        "rmse": regression_rmse(predictions, targets),
        "mae": regression_mae(predictions, targets),
    }
    classification = classification_metrics(
        predictions, targets, latency_threshold
    )
    return EvaluationMetrics(
        rmse=regression["rmse"],
        mae=regression["mae"],
        breach_accuracy=classification["accuracy"],
        true_positive_rate=classification["true_positive_rate"],
        false_positive_rate=classification["false_positive_rate"],
        false_negative_rate=classification["false_negative_rate"],
        precision=classification["precision"],
        recall=classification["recall"],
        f1_score=classification["f1_score"],
        true_positives=classification["tp"],
        true_negatives=classification["tn"],
        false_positives=classification["fp"],
        false_negatives=classification["fn"],
    )
