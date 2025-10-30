"""Conditional flow matching utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import ConditionalVelocityMLP


@dataclass
class FlowMatchingConfig:
    seq_len: int
    control_dim: int
    static_dim: int
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 200
    batch_size: int = 128
    device: str = "cpu"
    warmup_steps: int = 0


class FlowMatchingTrainer:
    def __init__(self, config: FlowMatchingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = ConditionalVelocityMLP(
            seq_len=config.seq_len,
            control_dim=config.control_dim,
            static_dim=config.static_dim,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.global_step = 0

    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        for key in batch:
            batch[key] = batch[key].to(self.device)

        targets = batch["targets"]
        batch_size = targets.size(0)
        t = torch.rand(batch_size, device=self.device)
        base = torch.randn_like(targets)
        x_t = (1 - t).unsqueeze(-1) * base + t.unsqueeze(-1) * targets
        velocity_target = targets - base

        pred = self.model(
            x_t=x_t,
            t=t,
            dynamic_controls=batch["dynamic_controls"],
            static_controls=batch["static_controls"],
        )
        loss = F.mse_loss(pred, velocity_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        return loss.item()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        losses: List[float] = []
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].to(self.device)
            targets = batch["targets"]
            batch_size = targets.size(0)
            t = torch.rand(batch_size, device=self.device)
            base = torch.randn_like(targets)
            x_t = (1 - t).unsqueeze(-1) * base + t.unsqueeze(-1) * targets
            velocity_target = targets - base

            pred = self.model(
                x_t=x_t,
                t=t,
                dynamic_controls=batch["dynamic_controls"],
                static_controls=batch["static_controls"],
            )
            loss = F.mse_loss(pred, velocity_target)
            losses.append(loss.item())
        return float(torch.tensor(losses).mean().item())

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> Dict[str, List[float]]:
        history = {"train": [], "val": []}
        for epoch in range(self.config.n_epochs):
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
    def sample(
        self,
        dynamic_controls: torch.Tensor,
        static_controls: torch.Tensor,
        n_steps: int = 30,
        base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples conditioned on the controls."""

        self.model.eval()
        dynamic_controls = dynamic_controls.to(self.device)
        static_controls = static_controls.to(self.device)
        batch_size, seq_len, _ = dynamic_controls.shape
        if base is None:
            x = torch.randn(batch_size, seq_len, device=self.device)
        else:
            x = base.to(self.device)

        times = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)
        for i in range(n_steps):
            t = torch.full((batch_size,), times[i], device=self.device)
            velocity = self.model(
                x_t=x,
                t=t,
                dynamic_controls=dynamic_controls,
                static_controls=static_controls,
            )
            dt = times[i + 1] - times[i]
            x = x + dt * velocity
        return x
