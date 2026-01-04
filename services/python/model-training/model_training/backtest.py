from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
import pandas as pd

from .data import (
    load_feature_store,
    prepare_time_series_split,
)
from .metrics import evaluate_predictions
from .model import LatencyForecaster
from .train import evaluate_loader, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-series backtesting for the latency forecaster."
    )
    parser.add_argument(
        "--feature-store-path",
        default="../feature_store",
        help="Directory containing feature parquet files.",
    )
    parser.add_argument(
        "--feature-columns",
        default="cpu_rolling_avg,path_depth,fan_out",
        help="Comma separated list of feature columns.",
    )
    parser.add_argument(
        "--target-column",
        default="target_latency_p99",
        help="Column representing future P99 latency.",
    )
    parser.add_argument("--sequence-length", type=int, default=6)
    parser.add_argument("--latency-threshold", type=float, default=500.0)
    parser.add_argument("--train-window-days", type=int, default=21)
    parser.add_argument("--val-window-days", type=int, default=7)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--min-val-rows", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="artifacts/backtest.json",
        help="Path to write fold metrics.",
    )
    return parser.parse_args()


def sliding_windows(
    df,
    train_days: int,
    val_days: int,
    min_train_rows: int,
    min_val_rows: int,
) -> List[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    windows = []
    train_start = start
    train_delta = timedelta(days=train_days)
    val_delta = timedelta(days=val_days)
    while True:
        train_end = train_start + train_delta
        val_end = train_end + val_delta
        if val_end > end:
            break
        train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
        val_mask = (df["timestamp"] >= train_end) & (df["timestamp"] < val_end)
        if train_mask.sum() < min_train_rows or val_mask.sum() < min_val_rows:
            train_start += val_delta
            continue
        windows.append((train_start, train_end, val_end))
        train_start += val_delta
    return windows


def run_fold(
    fold_idx: int,
    df,
    window,
    feature_cols,
    target_col,
    args,
) -> dict:
    train_start, train_end, val_end = window
    val_start = train_end
    train_df = df[
        (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
    ]
    val_df = df[(df["timestamp"] >= val_start) & (df["timestamp"] < val_end)]

    try:
        train_dataset, val_dataset, scaler = prepare_time_series_split(
            train_df,
            val_df,
            feature_cols,
            target_col,
            args.sequence_length,
            args.latency_threshold,
        )
    except ValueError as exc:
        print(f"[Fold {fold_idx}] Skipping due to insufficient data: {exc}")
        return None

    if len(train_dataset) < 1 or len(val_dataset) < 1:
        raise ValueError("Not enough sequences built for fold {fold_idx}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    device = torch.device(args.device)
    model = LatencyForecaster(
        input_size=train_dataset.num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model, _ = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    val_predictions, val_targets = evaluate_loader(model, val_loader, device)
    metrics = evaluate_predictions(
        val_predictions, val_targets, args.latency_threshold
    )
    print(
        f"[Fold {fold_idx}] {train_start.date()}-{val_end.date()} "
        f"RMSE={metrics.rmse:.2f} MAE={metrics.mae:.2f} "
        f"FPR={metrics.false_positive_rate:.4f} "
        f"FNR={metrics.false_negative_rate:.4f}"
    )
    return {
        "fold": fold_idx,
        "train_window": {
            "start": train_start.isoformat(),
            "end": train_end.isoformat(),
        },
        "validation_window": {
            "start": val_start.isoformat(),
            "end": val_end.isoformat(),
        },
        "metrics": metrics.as_dict(),
    }


def main() -> None:
    args = parse_args()
    feature_cols = [col.strip() for col in args.feature_columns.split(",")]
    df = load_feature_store(args.feature_store_path).sort_values("timestamp")

    windows = sliding_windows(
        df,
        args.train_window_days,
        args.val_window_days,
        args.min_train_rows,
        args.min_val_rows,
    )
    if not windows:
        raise RuntimeError("No valid backtesting windows found.")

    results = []
    for idx, window in enumerate(windows, start=1):
        fold_result = run_fold(
            idx,
            df,
            window,
            feature_cols,
            args.target_column,
            args,
        )
        if fold_result:
            results.append(fold_result)

    if not results:
        raise RuntimeError("All backtesting windows were skipped; not enough sequences.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Backtest summary written to {output_path}")


if __name__ == "__main__":
    main()
