# Latency Forecasting Model Training

Phase 2 focuses on training and validating the latency predictor built on top of the Feature Store. This module provides a PyTorch-based LSTM pipeline plus explainability artifacts (SHAP) to surface causality scores for SRE workflows.

## Highlights
- **Model choice:** LSTM/GRU-style recurrent net (implemented with a configurable LSTM) to capture temporal dependencies across per-service telemetry sequences.
- **Data prep:** Loads Parquet feature sets produced by the Feature Aggregator, performs normalization, and builds rolling sequences of configurable length.
- **Targets:** `target_latency_p99` at a future horizon becomes the regression target; a latency threshold (default 500ms) turns the task into a breach/no-breach classification for business metrics.
- **Metrics:** Reports RMSE/MAE plus breach accuracy, TPR, False Positive/Negative Rates, and raw confusion-matrix counts so SRE alert fatigue can be tracked directly.
- **Explainability:** Runs SHAP KernelExplainer on validation windows to highlight which features (CPU, Path Depth, Fan-Out, etc.) contributed to a predicted spike, yielding a per-feature causality score. Additional tooling inspects causality per true-positive event.
- **Backtesting:** A dedicated CLI (`backtest.py`) performs strict time-series cross-validation (e.g., train Weeks 1-3 → validate Week 4) to guarantee no lookahead bias.

## Setup
```
cd model-training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Ensure the Feature Aggregator has produced Parquet files (default `../feature_store`). Adjust `--feature-store-path` if needed.

## Running Training
```
python train.py \
  --feature-store-path ../feature_store \
  --feature-columns cpu_rolling_avg,path_depth,fan_out \
  --sequence-length 8 \
  --latency-threshold 500 \
  --epochs 30 \
  --output-dir artifacts
```

Key arguments:
- `--sequence-length`: number of minutes of history per sample.
- `--latency-threshold`: breach definition for the classification-oriented metrics.
- `--hidden-size`, `--num-layers`, `--dropout`: tune the LSTM capacity.
- `--device`: `cuda` or `cpu`.

Outputs (default `model-training/artifacts/`):
- `latency_forecaster.pt`: trained PyTorch weights.
- `scaler.pt`: StandardScaler fitted on training data.
- `metrics.json`: validation + test metrics (RMSE, MAE, FPR, FNR, confusion counts).
- `training_history.json`: per-epoch losses.
- `feature_importance.json`: SHAP-derived average importance per feature (your causality signal).

## Time-Series Backtesting
Use `backtest.py` to roll training and validation windows forward in time (default: 21-day train, 7-day validation). This mirrors Week 1-3 ➜ Week 4 style evaluations.
```
python backtest.py \
  --feature-store-path ../feature_store \
  --train-window-days 21 \
  --val-window-days 7 \
  --output artifacts/backtest.json
```
Each fold retrains a fresh model, evaluates RMSE/MAE plus FPR/FNR, and writes the metrics with the exact window boundaries. Inspect `artifacts/backtest.json` or feed it into dashboards to prove statistical accuracy on unseen data.

## Causality Score Validation
After a chaos run (where you know the injected fault), load the saved checkpoint and scaler to inspect SHAP scores only for true positives:
```
python analyze_causality.py \
  --model-checkpoint artifacts/latency_forecaster.pt \
  --scaler-path artifacts/scaler.pt \
  --feature-store-path ../feature_store \
  --start-ts 2024-05-01 --end-ts 2024-05-02 \
  --max-samples 10 \
  --output artifacts/causality_report.json
```
The report lists each TP’s service/timestamp plus the top contributing feature. Cross-check that the highlighted feature matches the injected root cause (e.g., Inventory CPU saturation) to validate the system’s causality signal.

## Validation Playbook
1. **Numerical accuracy (RMSE/MAE):** Confirm model predictions are tight versus observed future latency.
2. **Business metric:** Inspect the breach accuracy/TPR along with `true_positives`/`false_negatives` to ensure real incidents are captured.
3. **False alarms:** Keep `false_positive_rate` under ~1% per the SRE requirement. Adjust the latency threshold or reweight the loss if alerts fire too often.
4. **Root-cause insights:** Compare chaos timelines to `feature_importance.json` and the per-TP `causality_report.json` to prove the model points at the injected dependency (CPU, queue depth, path depth, etc.).

Future improvements can explore GRU variants, multitask heads for joint regression/classification, or integrating graph embeddings before the recurrent layer.
