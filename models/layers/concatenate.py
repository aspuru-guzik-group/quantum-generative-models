from typing import Sequence

import torch
from torch import nn


class Concatenate(nn.Module):
    """A layer that concatenates multiple tensor into a single tensor along a given dimention"""

    def __init__(self, dim: int = -1) -> None:
        """
        Args:
            dim (int, optional): dimension along which to concatenate tensors. Defaults to -1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.concat(tensors, dim=self.dim)
