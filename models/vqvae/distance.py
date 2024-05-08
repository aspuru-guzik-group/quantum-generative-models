import typing as t
from abc import ABC, abstractmethod

import torch
from torch import nn


class _CodebookDistance(ABC):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.measure(x, y)

    def measure(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EuclidianCodebookDistance(_CodebookDistance):
    def measure(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes a pairwise distance between every vector in tensor x and y.

        Args:
            x (torch.Tensor): tensor with shape (n_x, vector_dim)
            y (torch.Tensor): tensor with shape (n_y, vector_dim)

        Returns:
            torch.Tensor: tensor with pairwise distance between every vector in x and y of shape
                (n_x, n_y) where the ij-th entry is the distance between vector x_i and y_j.
        """
        x_sq = torch.sum(x**2, dim=1, keepdim=True)
        y_sq = torch.sum(y**2, dim=1)
        x_y = torch.matmul(x, y.t())

        # when we sum x_sq and y_sq we get a (n_x, n_y) matrix
        return x_sq + y_sq - 2 * x_y
