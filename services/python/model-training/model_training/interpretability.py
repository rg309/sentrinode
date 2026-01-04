from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class ShapResult:
    feature_importance: Dict[str, float]
    shap_values: np.ndarray


def compute_shap_values(
    model: torch.nn.Module,
    dataset,
    feature_names,
    sample_size: int = 32,
    background_size: int = 64,
) -> ShapResult:
    import shap  # Lazy import to keep dependency optional

    sequences = dataset.to_numpy()
    total_samples = sequences.shape[0]
    sample_size = min(sample_size, total_samples)
    background_size = min(background_size, total_samples)

    flat_sequences = sequences.reshape(total_samples, -1)
    background = flat_sequences[:background_size]
    sample = flat_sequences[-sample_size:]

    seq_len = dataset.sequence_length
    num_features = dataset.num_features

    device = next(model.parameters()).device

    def predict(flat_batch: np.ndarray) -> np.ndarray:
        seq_batch = flat_batch.reshape(-1, seq_len, num_features)
        tensor = torch.tensor(seq_batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = model(tensor).detach().cpu().numpy()
        return preds

    explainer = shap.KernelExplainer(
        predict, background, link="identity"
    )
    shap_values = explainer.shap_values(sample, nsamples="auto")

    # Aggregate absolute SHAP values across the time axis
    shap_matrix = np.abs(shap_values).reshape(-1, seq_len, num_features)
    per_feature = shap_matrix.mean(axis=(0, 1))
    importance = {
        feature_names[idx]: float(per_feature[idx]) for idx in range(num_features)
    }
    return ShapResult(feature_importance=importance, shap_values=shap_values)
