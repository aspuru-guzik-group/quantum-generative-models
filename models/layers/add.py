import torch
from torch import nn


class Add(nn.Module):
    """Add two tensors."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y