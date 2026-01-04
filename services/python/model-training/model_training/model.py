from __future__ import annotations

import torch
from torch import nn


class LatencyForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(sequences)
        final_state = lstm_out[:, -1, :]
        regression = self.regressor(final_state)
        return regression.squeeze(-1)
