import typing as t

import torch
from torch import nn


class GRUGANGenerator(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        latent_dim: int,
        n_layers: int = 1,
        bidirectional: bool = True,
    ):
        """Creates an instance of a GRU-based generator component for an adversarial-style network.

        Args:
            n_features (int): number of features in the data tensors.
            seq_len (int): the length of the sequences the generator should generate.
            latent_dim (int): the dimensionality of the noise vector that will be fed to the generator.
            n_layers (int, optional): number of GRU layers. Defaults to 1.
            bidirectional (bool, optional): whether to read the data left-to-right only, or LTR and RTL directions. Defaults to True.
        """
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.n_layers = n_layers
        self._D = 2 if bidirectional else 1  # this is needed for computing output dims

        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # latent linear is responsible for ensuring that as we generate the output sequence element by element, all dims match up
        self.latent_linear = nn.Linear(self._D * latent_dim, latent_dim)
        self.leaky_relu = nn.LeakyReLU()

        # final linear layer that maps from latent space to space with dim `n_features`
        self.output_linear = nn.Linear(latent_dim, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = noise.size(0)

        # TODO: it might not be as simple as _D * n_layers, since that feeds the same noise to every layer
        # noise has shape (b, latent_dim), we want (self._D * self.n_layers, b, latent_dim)
        noise = noise.repeat((self._D * self.n_layers, 1, 1))

        hidden_state = noise

        # will be (b, seq_len, latent_dim), however must use cat instead of slicing as inplace operations cause issues
        generated_sequence = torch.zeros(0)
        inputs = torch.zeros((batch_size, 1, self.latent_dim))

        # NOTE: instead of feeding updated sequence to RNN, we feed previous result together with previous hidden state
        # which should be more or less the same thing due to weight sharing
        for i in range(self.seq_len):
            # NOTE: this passes a single element through all GRU layers in one go
            # (b, 1, self._D * latent_dim), (self._D * self.n_layers, b, latent_dim)
            output_sequence, hidden_state = self.rnn(inputs, hidden_state)

            # (b, 1, self._D * latent_dim) -> (b, 1, latent_dim)
            output_sequence = self.leaky_relu(self.latent_linear(output_sequence))

            generated_sequence = torch.cat((generated_sequence, output_sequence), dim=1)

            inputs = output_sequence

        # # TODO: do we want to apply LeakyReLU in the loop or do a MLP block here
        generated_sequence = self.leaky_relu(generated_sequence)
        generated_sequence = self.output_linear(generated_sequence)

        return self.sigmoid(generated_sequence)
