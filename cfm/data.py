"""Synthetic dataset generation utilities for flow matching experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class SyntheticDatasetConfig:
    """Configuration for :class:`SyntheticControlDataset`.

    Attributes
    ----------
    n_samples:
        Number of sequences to generate.
    seq_len:
        Length of each time-series sample.
    control_dim:
        Dimensionality of the time-varying control signal.
    static_dim:
        Dimensionality of the static control vector (per sequence).
    noise_std:
        Standard deviation of the observation noise.
    nonlinear_strength:
        Scale of the tanh-based nonlinearity that perturbs the latent state.
    regime_change_prob:
        Probability (per time step) of a regime switch that alters the dynamics.
    regime_scale:
        Magnitude of the jump applied at regime switches.
    seed:
        Optional random seed for reproducibility.
    """

    n_samples: int = 1024
    seq_len: int = 32
    control_dim: int = 2
    static_dim: int = 2
    noise_std: float = 0.15
    nonlinear_strength: float = 0.3
    regime_change_prob: float = 0.1
    regime_scale: float = 0.6
    seed: Optional[int] = None


class SyntheticControlDataset(Dataset):
    """Synthetic time-series with controllable statistics.

    Each sample contains a time-series target with exogenous controls. The
    dynamics are governed by a stochastic autoregressive process with optional
    regime changes, static features, and nonlinear interactions.
    """

    def __init__(self, config: SyntheticDatasetConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        (self.dynamic_controls,
         self.static_controls,
         self.targets,
         self.regime_ids) = self._generate()

    def _generate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        dyn_controls = np.zeros((cfg.n_samples, cfg.seq_len, cfg.control_dim), dtype=np.float32)
        static_controls = np.zeros((cfg.n_samples, cfg.static_dim), dtype=np.float32)
        targets = np.zeros((cfg.n_samples, cfg.seq_len), dtype=np.float32)
        regimes = np.zeros((cfg.n_samples, cfg.seq_len), dtype=np.float32)

        for idx in range(cfg.n_samples):
            static = self._rng.normal(0.0, 1.0, size=(cfg.static_dim,)).astype(np.float32)
            static_controls[idx] = static

            # Sample sample-specific linear weights for controls and biases.
            base_a = 0.4 + 0.4 * self._rng.uniform()
            base_b = self._rng.normal(0.0, 0.8, size=(cfg.control_dim,))
            bias = 0.1 * static.sum()

            # Generate dynamic control trajectories.
            controls = self._rng.normal(0.0, 1.0, size=(cfg.seq_len, cfg.control_dim)).astype(np.float32)
            dyn_controls[idx] = controls

            state = self._rng.normal(0.0, 1.0)
            regime_state = 0.0
            for t in range(cfg.seq_len):
                if self._rng.uniform() < cfg.regime_change_prob:
                    regime_state += self._rng.normal(0.0, cfg.regime_scale)
                regimes[idx, t] = regime_state

                effective_a = np.tanh(base_a + 0.1 * regime_state)
                control_effect = float(np.dot(controls[t], base_b))
                nonlinear_term = cfg.nonlinear_strength * np.tanh(state + 0.5 * bias)
                noise = self._rng.normal(0.0, cfg.noise_std * (1.0 + 0.2 * abs(regime_state)))

                state = effective_a * state + 0.6 * control_effect + nonlinear_term + bias + noise
                targets[idx, t] = state

        return (
            torch.from_numpy(dyn_controls),
            torch.from_numpy(static_controls),
            torch.from_numpy(targets),
            torch.from_numpy(regimes),
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "dynamic_controls": self.dynamic_controls[idx],
            "static_controls": self.static_controls[idx],
            "targets": self.targets[idx],
            "regimes": self.regime_ids[idx],
        }


def train_val_test_split(
    dataset: SyntheticControlDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[Subset, Subset, Subset]:
    """Split the dataset into train/val/test subsets."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )
