"""Model definitions for synthetic flow matching experiments."""
from __future__ import annotations

import torch
from torch import nn


class MeanRegressorMLP(nn.Module):
    """Predicts the conditional mean of the target trajectory with an MLP."""

    def __init__(
        self,
        seq_len: int,
        control_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = seq_len * control_dim + static_dim
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, seq_len))
        self.net = nn.Sequential(*layers)

    def forward(self, dynamic_controls: torch.Tensor, static_controls: torch.Tensor) -> torch.Tensor:
        batch_size = dynamic_controls.size(0)
        features = torch.cat(
            [dynamic_controls.view(batch_size, -1), static_controls], dim=-1
        )
        return self.net(features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    """Transformer encoder that outputs a conditional mean trajectory."""

    def __init__(
        self,
        seq_len: int,
        control_dim: int,
        static_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.control_embed = nn.Linear(control_dim, d_model)
        self.static_embed = nn.Linear(static_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, dynamic_controls: torch.Tensor, static_controls: torch.Tensor) -> torch.Tensor:
        static_token = self.static_embed(static_controls).unsqueeze(1).expand(-1, self.seq_len, -1)
        control_tokens = self.control_embed(dynamic_controls)
        tokens = control_tokens + static_token
        tokens = self.positional_encoding(tokens)
        encoded = self.encoder(tokens)
        preds = self.output_proj(encoded).squeeze(-1)
        return preds


class ConditionalVelocityMLP(nn.Module):
    """Velocity field network used for conditional flow matching."""

    def __init__(
        self,
        seq_len: int,
        control_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = seq_len + seq_len * control_dim + static_dim + 1  # +1 for time
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, seq_len))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dynamic_controls: torch.Tensor,
        static_controls: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_t.size(0)
        features = torch.cat(
            [
                x_t.view(batch_size, -1),
                dynamic_controls.view(batch_size, -1),
                static_controls,
                t.view(batch_size, 1),
            ],
            dim=-1,
        )
        velocity = self.net(features)
        return velocity


def count_parameters(model: nn.Module) -> int:
    """Utility to count trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
