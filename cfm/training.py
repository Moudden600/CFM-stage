"""Training utilities for conditional mean models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import MeanRegressorMLP, TransformerRegressor


@dataclass
class MeanModelConfig:
    seq_len: int
    control_dim: int
    static_dim: int
    model_type: str = "mlp"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    n_epochs: int = 200
    device: str = "cpu"


class MeanModelTrainer:
    def __init__(self, config: MeanModelConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if config.model_type == "mlp":
            self.model: nn.Module = MeanRegressorMLP(
                seq_len=config.seq_len,
                control_dim=config.control_dim,
                static_dim=config.static_dim,
            )
        elif config.model_type == "transformer":
            self.model = TransformerRegressor(
                seq_len=config.seq_len,
                control_dim=config.control_dim,
                static_dim=config.static_dim,
            )
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        for key in batch:
            batch[key] = batch[key].to(self.device)
        preds = self.model(batch["dynamic_controls"], batch["static_controls"])
        loss = F.mse_loss(preds, batch["targets"])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        losses: List[float] = []
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(self.device)
            preds = self.model(batch["dynamic_controls"], batch["static_controls"])
            loss = F.mse_loss(preds, batch["targets"])
            losses.append(loss.item())
        return float(torch.tensor(losses).mean().item())

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> Dict[str, List[float]]:
        history = {"train": [], "val": []}
        for _ in range(self.config.n_epochs):
            epoch_losses = []
            for batch in train_loader:
                loss = self.step(batch)
                epoch_losses.append(loss)
            train_loss = float(torch.tensor(epoch_losses).mean().item())
            history["train"].append(train_loss)
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history["val"].append(val_loss)
            else:
                history["val"].append(float("nan"))
        return history

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        preds = []
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(self.device)
            pred = self.model(batch["dynamic_controls"], batch["static_controls"])
            preds.append(pred.cpu())
        return torch.cat(preds, dim=0)
