import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding layer for use with Transformer, or similar, architectures. This module
    allows to add positional information, to sequences that otherwise carry no positional information.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        """Initialize positional encoding layer.

        Args:
            d_model (int): Dimensionality of the hidden layers of the transformer.
            dropout (float, optional): Rate at which to randomly drop neurons. Defaults to 0.1.
            max_len (int, optional): Hyperparameter for adjusting the time-period of the sinusoidal functions. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # expect x to have shape (B, L, D)
        x = x + self.positional_encoding[:, : x.size(1), :]  # type: ignore
        return self.dropout(x)
