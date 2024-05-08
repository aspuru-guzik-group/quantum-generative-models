import torch
from torch import nn

from .residual import ResidualStack


class Encoder(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        hidden_activation_fn: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.embedding = nn.Embedding()
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=hidden_dim,
            hidden_dim=residual_layer_hidden_dim,
            n_residual_layers=n_residual_layers,
            hidden_activation_fn=hidden_activation_fn,
        )
        self.hidden_activation_fn = hidden_activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._conv_1(inputs)
        x = self.hidden_activation_fn(x)

        x = self._conv_2(x)
        x = self.hidden_activation_fn(x)

        x = self._conv_3(x)
        return self._residual_stack(x)
