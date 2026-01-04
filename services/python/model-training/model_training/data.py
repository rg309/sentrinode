from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class SequenceDataset(Dataset):
    sequences: torch.Tensor
    targets: torch.Tensor
    breach_labels: torch.Tensor
    services: List[str]
    timestamps: List[str]

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],
            self.targets[idx],
            self.breach_labels[idx],
        )

    @property
    def num_features(self) -> int:
        return self.sequences.size(-1)

    @property
    def sequence_length(self) -> int:
        return self.sequences.size(1)

    def to_numpy(self) -> np.ndarray:
        return self.sequences.cpu().numpy()

    def metadata(self, idx: int) -> dict:
        return {
            "service": self.services[idx],
            "timestamp": self.timestamps[idx],
        }

    def subset(self, indices: Sequence[int]) -> "SequenceDataset":
        index_list = list(indices)
        seq = self.sequences[index_list]
        tgt = self.targets[index_list]
        breach = self.breach_labels[index_list]
        services = [self.services[i] for i in index_list]
        timestamps = [self.timestamps[i] for i in index_list]
        return SequenceDataset(
            sequences=seq,
            targets=tgt,
            breach_labels=breach,
            services=services,
            timestamps=timestamps,
        )


def load_feature_store(path: str | Path) -> pd.DataFrame:
    base = Path(path)
    frames: List[pd.DataFrame] = []
    for parquet in sorted(base.glob("*.parquet")):
        frame = pd.read_parquet(parquet)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {base}")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    services: List[str] = []
    timestamps: List[str] = []
    grouped = df.groupby("service", sort=False)
    for service_name, service_frame in grouped:
        service_frame = service_frame.sort_values("timestamp")
        feat = service_frame[feature_cols].to_numpy(dtype=np.float32)
        tgt = service_frame[target_col].to_numpy(dtype=np.float32)
        if len(service_frame) < sequence_length:
            continue
        for idx in range(sequence_length - 1, len(service_frame)):
            window = feat[idx - sequence_length + 1 : idx + 1]
            sequences.append(window)
            targets.append(float(tgt[idx]))
            services.append(str(service_name))
            timestamps.append(
                pd.Timestamp(service_frame["timestamp"].iloc[idx]).isoformat()
            )
    if not sequences:
        raise ValueError(
            "Insufficient rows to build sequences. "
            f"Received {len(df)} rows, need >= sequence_length per service."
        )
    return (
        np.stack(sequences),
        np.array(targets, dtype=np.float32),
        services,
        timestamps,
    )


def _apply_scaler(sequences: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, seq_len, num_feat = sequences.shape
    flat = sequences.reshape(-1, num_feat)
    scaled = scaler.transform(flat)
    return scaled.reshape(n, seq_len, num_feat)


def create_datasets(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    latency_threshold: float,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 7,
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset, StandardScaler]:
    sequences, targets, services, timestamps = _build_sequences(
        df, feature_cols, target_col, sequence_length
    )
    services_arr = np.array(services)
    timestamps_arr = np.array(timestamps)

    indices = np.arange(len(sequences))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, shuffle=True
    )
    val_relative = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_relative, random_state=random_state, shuffle=True
    )

    scaler = StandardScaler()
    scaler.fit(sequences[train_idx].reshape(-1, sequences.shape[-1]))

    def build_dataset(idxs: np.ndarray) -> SequenceDataset:
        scaled_sequences = _apply_scaler(sequences[idxs], scaler)
        seq_tensor = torch.tensor(scaled_sequences, dtype=torch.float32)
        target_tensor = torch.tensor(targets[idxs], dtype=torch.float32)
        breach_tensor = (target_tensor > latency_threshold).float()
        service_subset = services_arr[idxs].tolist()
        timestamp_subset = timestamps_arr[idxs].tolist()
        return SequenceDataset(
            sequences=seq_tensor,
            targets=target_tensor,
            breach_labels=breach_tensor,
            services=[str(item) for item in service_subset],
            timestamps=[str(item) for item in timestamp_subset],
        )

    return (
        build_dataset(train_idx),
        build_dataset(val_idx),
        build_dataset(test_idx),
        scaler,
    )


def prepare_time_series_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    latency_threshold: float,
) -> Tuple[SequenceDataset, SequenceDataset, StandardScaler]:
    train_seq, train_targets, train_services, train_ts = _build_sequences(
        train_df, feature_cols, target_col, sequence_length
    )
    val_seq, val_targets, val_services, val_ts = _build_sequences(
        val_df, feature_cols, target_col, sequence_length
    )
    scaler = StandardScaler()
    scaler.fit(train_seq.reshape(-1, train_seq.shape[-1]))

    def build(
        sequences: np.ndarray,
        targets: np.ndarray,
        services: List[str],
        timestamps: List[str],
    ) -> SequenceDataset:
        scaled = _apply_scaler(sequences, scaler)
        seq_tensor = torch.tensor(scaled, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        breach_tensor = (target_tensor > latency_threshold).float()
        return SequenceDataset(
            sequences=seq_tensor,
            targets=target_tensor,
            breach_labels=breach_tensor,
            services=list(map(str, services)),
            timestamps=list(map(str, timestamps)),
        )

    return (
        build(train_seq, train_targets, train_services, train_ts),
        build(val_seq, val_targets, val_services, val_ts),
        scaler,
    )


def build_dataset_with_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    scaler: StandardScaler,
    latency_threshold: float,
) -> SequenceDataset:
    sequences, targets, services, timestamps = _build_sequences(
        df, feature_cols, target_col, sequence_length
    )
    scaled = _apply_scaler(sequences, scaler)
    seq_tensor = torch.tensor(scaled, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    breach_tensor = (target_tensor > latency_threshold).float()
    return SequenceDataset(
        sequences=seq_tensor,
        targets=target_tensor,
        breach_labels=breach_tensor,
        services=list(map(str, services)),
        timestamps=list(map(str, timestamps)),
    )
