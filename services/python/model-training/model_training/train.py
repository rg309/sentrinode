from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import create_datasets, load_feature_store
from .interpretability import compute_shap_values
from .metrics import evaluate_predictions
from .model import LatencyForecaster
from .trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate an LSTM latency forecaster."
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
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=6,
        help="Number of timesteps per training sample.",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=500.0,
        help="Threshold (ms) for breach classification metrics.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for model checkpoints and metrics.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cuda or cpu).",
    )
    return parser.parse_args()


def evaluate_loader(model, loader, device: torch.device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for sequences, batch_targets, _ in loader:
            sequences = sequences.to(device)
            preds = model(sequences).detach().cpu().numpy()
            predictions.append(preds)
            targets.append(batch_targets.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    args = parse_args()
    feature_columns = [col.strip() for col in args.feature_columns.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_store(args.feature_store_path)
    train_dataset, val_dataset, test_dataset, scaler = create_datasets(
        df,
        feature_columns,
        args.target_column,
        args.sequence_length,
        latency_threshold=args.latency_threshold,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device(args.device)
    model = LatencyForecaster(
        input_size=train_dataset.num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    val_predictions, val_targets = evaluate_loader(model, val_loader, device)
    test_predictions, test_targets = evaluate_loader(model, test_loader, device)

    val_metrics = evaluate_predictions(
        val_predictions, val_targets, args.latency_threshold
    )
    test_metrics = evaluate_predictions(
        test_predictions, test_targets, args.latency_threshold
    )

    shap_result = compute_shap_values(
        model,
        test_dataset,
        feature_names=feature_columns,
        sample_size=32,
        background_size=64,
    )

    torch.save(model.state_dict(), output_dir / "latency_forecaster.pt")
    torch.save(scaler, output_dir / "scaler.pt")

    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
            },
            indent=2,
        )
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "validation": val_metrics.as_dict(),
                "test": test_metrics.as_dict(),
            },
            indent=2,
        )
    )

    shap_path = output_dir / "feature_importance.json"
    shap_path.write_text(
        json.dumps(
            {
                "feature_importance": shap_result.feature_importance,
            },
            indent=2,
        )
    )

    print("Validation metrics:", json.dumps(val_metrics.as_dict(), indent=2))
    print("Test metrics:", json.dumps(test_metrics.as_dict(), indent=2))
    print("Top feature attributions:", shap_result.feature_importance)


if __name__ == "__main__":
    main()
