from typing import Any

import torch
from einops import repeat
from torch import nn


class Repeat(nn.Module):
    """Repeat a tensor following a pattern.
    Wraps around the einops.repeat function, but can be used where a torch.nn.Module is expected.
    """

    def __init__(self, pattern: str, **axes_lengths: Any) -> None:
        """Initialize the layer.

        Args:
            pattern (str): pattern used to repeat the tensor.
            axes_lengths (Any): keyword arguments specifying the length of the axes used in the pattern.
        """
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def __repr__(self) -> str:
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ", {}={}".format(axis, length)
        return "{}({})".format(self.__class__.__name__, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return repeat(x, pattern=self.pattern, **self.axes_lengths)
