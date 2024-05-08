import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        hidden_activation_fn: nn.Module = nn.ReLU(True),
    ):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            hidden_activation_fn,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            hidden_activation_fn,
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x) -> torch.Tensor:
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_residual_layers: int,
        hidden_activation_fn: nn.Module = nn.ReLU(),
    ):
        super(ResidualStack, self).__init__()
        self.n_residual_layers = n_residual_layers
        self.layers = nn.ModuleList(
            [Residual(in_channels, hidden_dim) for _ in range(self.n_residual_layers)]
        )
        self.hidden_activation_fn = hidden_activation_fn

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.hidden_activation_fn(x)
