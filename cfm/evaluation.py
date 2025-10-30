"""Evaluation helpers for the synthetic flow matching benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .training import MeanModelTrainer
from .flow_matching import FlowMatchingTrainer


@dataclass
class DataEfficiencyResult:
    subset_size: int
    train_loss: float
    val_loss: float
    test_mse: float


def compute_mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return float(torch.mean((preds - targets) ** 2).item())


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in keys
    }


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def evaluate_mean_model(trainer: MeanModelTrainer, dataloader: DataLoader) -> float:
    preds = trainer.predict(dataloader)
    targets = torch.cat([batch["targets"] for batch in dataloader.dataset])  # type: ignore[arg-type]
    return compute_mse(preds, targets)


def evaluate_flow_matching(
    trainer: FlowMatchingTrainer,
    dataloader: DataLoader,
    n_samples: int = 8,
) -> Dict[str, float]:
    metrics = {"mse": 0.0, "mae": 0.0}
    total = 0
    for batch in dataloader:
        dynamic = batch["dynamic_controls"]
        static = batch["static_controls"]
        targets = batch["targets"]
        generated = trainer.sample(dynamic, static, n_steps=40)
        metrics["mse"] += torch.sum((generated - targets) ** 2).item()
        metrics["mae"] += torch.sum(torch.abs(generated - targets)).item()
        total += targets.numel()
    metrics["mse"] /= total
    metrics["mae"] /= total
    return metrics


def _make_subset(dataset: Dataset, subset_size: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices.tolist())


def data_efficiency_curve(
    subset_sizes: Iterable[int],
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    trainer_factory: Callable[[int], MeanModelTrainer],
    batch_size: int,
    seed: int = 0,
) -> List[DataEfficiencyResult]:
    results: List[DataEfficiencyResult] = []
    for subset_size in subset_sizes:
        subset = _make_subset(train_dataset, subset_size, seed + subset_size)
        train_loader = make_dataloader(subset, batch_size=batch_size, shuffle=True)
        val_loader = make_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = make_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

        trainer = trainer_factory(subset_size)
        history = trainer.fit(train_loader, val_loader)
        test_preds = trainer.predict(test_loader)
        test_targets = torch.cat([batch["targets"] for batch in test_loader.dataset])  # type: ignore[arg-type]
        test_mse = compute_mse(test_preds, test_targets)
        results.append(
            DataEfficiencyResult(
                subset_size=subset_size,
                train_loss=history["train"][-1],
                val_loss=history["val"][-1],
                test_mse=test_mse,
            )
        )
    return results
