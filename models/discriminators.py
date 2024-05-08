import typing as t

import torch
from einops.layers.torch import Rearrange
from orquestra.qml.models.adversarial.th import Discriminator
from torch import nn


# TODO: _DiscriminatorModule is implemented in a weird way, we might not need it
class _GRUGANDiscriminator(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int,
        n_layers: int = 1,
        bidirectional: bool = True,
    ):
        """Creates an instance of a GRU-based discriminator component for an adversarial-style network.

        Args:
            n_features (int): number of features in the data tensors.
            seq_len (int): the length of the sequences the generator should generate.
            hidden_dim (int): the dimensionality of the deep layer of the discriminator.
            n_layers (int, optional): number of GRU layers. Defaults to 1.
            bidirectional (bool, optional): whether to read the data left-to-right only, or LTR and RTL directions. Defaults to True.
        """
        super().__init__()
        _D = 2 if bidirectional else 1

        # NOTE: after passing a sample through n_layers GRUs we keep the final hidden state,
        # flatten it and use that to perform classification. Hidden state contains a "summarized" view input samples
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.rearrange = Rearrange(
            "o b l -> b (o l)"
        )  # this moves batch dim to index 0, and flattens other dims
        self.latent_linear = nn.Linear(_D * hidden_dim * n_layers, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.classifier_linear = nn.Linear(hidden_dim, 1)
        # NOTE: no activation here because activation is applied by AssociativeDiscriminator class

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(tensor)

        # hidden state should capture a representation of the input samples
        out = self.rearrange(h)
        out = self.latent_linear(out)
        out = self.leaky_relu(out)
        out = self.classifier_linear(out)

        return out


class GRUGANDiscriminator(Discriminator):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int,
        n_layers: int = 1,
        bidirectional: bool = True,
    ):
        """Creates an instance of a GRU-based discriminator component for an adversarial-style network.

        Args:
            n_features (int): number of features in the data tensors.
            seq_len (int): the length of the sequences the generator should generate.
            hidden_dim (int): the dimensionality of the deep layer of the discriminator.
            n_layers (int, optional): number of GRU layers. Defaults to 1.
            bidirectional (bool, optional): whether to read the data left-to-right only, or LTR and RTL directions. Defaults to True.
        """
        model = _GRUGANDiscriminator(
            n_features, seq_len, hidden_dim, n_layers, bidirectional
        )
        super().__init__(model)
