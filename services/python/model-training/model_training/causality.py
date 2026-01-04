from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import build_dataset_with_scaler, load_feature_store
from .interpretability import compute_shap_values
from .model import LatencyForecaster
from .train import evaluate_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SHAP causality scores for true positive events."
    )
    parser.add_argument("--feature-store-path", default="../feature_store")
    parser.add_argument(
        "--feature-columns",
        default="cpu_rolling_avg,path_depth,fan_out",
    )
    parser.add_argument("--target-column", default="target_latency_p99")
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--latency-threshold", type=float, default=500.0)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--scaler-path", required=True)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-ts", help="YYYY-MM-DD filter start", default=None)
    parser.add_argument("--end-ts", help="YYYY-MM-DD filter end", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--background-size", type=int, default=64)
    parser.add_argument(
        "--output",
        default="artifacts/causality_report.json",
        help="Path for causality diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cols = [col.strip() for col in args.feature_columns.split(",")]
    df = load_feature_store(args.feature_store_path)
    if args.start_ts:
        df = df[df["timestamp"] >= pd.Timestamp(args.start_ts, tz="UTC")]
    if args.end_ts:
        df = df[df["timestamp"] < pd.Timestamp(args.end_ts, tz="UTC")]
    if df.empty:
        raise RuntimeError("No data left after applying time filters.")

    scaler = torch.load(args.scaler_path)
    dataset = build_dataset_with_scaler(
        df,
        feature_cols,
        args.target_column,
        args.sequence_length,
        scaler,
        args.latency_threshold,
    )
    if len(dataset) == 0:
        raise RuntimeError("No sequences available for causality analysis.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = LatencyForecaster(
        input_size=dataset.num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)

    predictions, targets = evaluate_loader(model, loader, device)
    tp_mask = np.logical_and(
        predictions > args.latency_threshold, targets > args.latency_threshold
    )
    tp_indices = np.where(tp_mask)[0]
    if tp_indices.size == 0:
        print("No true positive events found in the provided window.")
        return
    tp_subset = dataset.subset(tp_indices[: args.max_samples])

    shap_result = compute_shap_values(
        model,
        tp_subset,
        feature_names=feature_cols,
        sample_size=min(len(tp_subset), args.max_samples),
        background_size=min(len(tp_subset), args.background_size),
    )
    shap_matrix = np.abs(shap_result.shap_values).reshape(
        -1, tp_subset.sequence_length, tp_subset.num_features
    )
    shap_averaged = shap_matrix.mean(axis=1)

    report: List[dict] = []
    for idx in range(len(tp_subset)):
        feature_scores = shap_averaged[idx]
        top_idx = int(np.argmax(feature_scores))
        metadata = tp_subset.metadata(idx)
        report.append(
            {
                "service": metadata["service"],
                "timestamp": metadata["timestamp"],
                "top_feature": feature_cols[top_idx],
                "top_score": float(feature_scores[top_idx]),
                "all_features": {
                    feature_cols[i]: float(feature_scores[i])
                    for i in range(len(feature_cols))
                },
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "true_positive_samples": len(tp_subset),
                "reports": report,
            },
            indent=2,
        )
    )
    print(f"Causality report written to {output_path}")


if __name__ == "__main__":
    main()
