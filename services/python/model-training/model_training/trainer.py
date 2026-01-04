from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    train_loss: Dict[int, float]
    val_loss: Dict[int, float]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[nn.Module, TrainingHistory]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.to(device)
    best_model_state = None
    best_val_loss = float("inf")
    history = TrainingHistory(train_loss={}, val_loss={})

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        for sequences, targets, _ in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * sequences.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for sequences, targets, _ in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                epoch_val_loss += loss.item() * sequences.size(0)
        epoch_val_loss /= len(val_loader.dataset)

        history.train_loss[epoch] = epoch_train_loss
        history.val_loss[epoch] = epoch_val_loss

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history
