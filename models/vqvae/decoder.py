import torch
from torch import nn

from .residual import ResidualStack


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_residual_layers: int,
        residual_layer_hidden_dim: int,
        hidden_activation_fn: nn.Module = nn.ReLU(),
    ):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
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

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=hidden_dim // 2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.hidden_activation_fn = hidden_activation_fn

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = self.hidden_activation_fn(x)

        return self._conv_trans_2(x)
