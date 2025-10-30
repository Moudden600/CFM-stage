"""Synthetic flow matching benchmark package."""

from .data import SyntheticControlDataset, SyntheticDatasetConfig, train_val_test_split
from .models import (
    MeanRegressorMLP,
    TransformerRegressor,
    ConditionalVelocityMLP,
    count_parameters,
)
from .training import MeanModelTrainer, MeanModelConfig
from .flow_matching import FlowMatchingTrainer, FlowMatchingConfig
from .evaluation import (
    DataEfficiencyResult,
    compute_mse,
    make_dataloader,
    data_efficiency_curve,
    evaluate_flow_matching,
)
from . import utils

__all__ = [
    "SyntheticControlDataset",
    "SyntheticDatasetConfig",
    "train_val_test_split",
    "MeanRegressorMLP",
    "TransformerRegressor",
    "ConditionalVelocityMLP",
    "count_parameters",
    "MeanModelTrainer",
    "MeanModelConfig",
    "FlowMatchingTrainer",
    "FlowMatchingConfig",
    "DataEfficiencyResult",
    "compute_mse",
    "make_dataloader",
    "data_efficiency_curve",
    "evaluate_flow_matching",
    "utils",
]
