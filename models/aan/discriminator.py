from typing import Optional, Tuple

import torch
from einops.layers.torch import Reduce
from orquestra.qml.losses.th import LossFunction
from orquestra.qml.models.adversarial.th import (
    AssociativeDiscriminator,
    AssociativeDiscriminatorWrapper,
)
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn
from torch.nn import functional as F


class _LSTMDiscriminator(nn.Module):
    def __init__(self, recurrent_net: torch.nn.LSTM, embedding_layer: nn.Embedding):
        """LSTM-based GAN Discriminator.

        Args:
            recurrent_net (torch.nn.LSTM): built LSTM.
            embedding_layer (typing.Optional[nn.Embedding], optional): token embedding layer.
        """
        super().__init__()
        self.n_features = recurrent_net.input_size
        self.latent_dim = recurrent_net.hidden_size
        self._D = (
            2 if recurrent_net.bidirectional else 1
        )  # this is needed for computing output dims

        self.embedding_layer = embedding_layer

        self.recurrent_net = recurrent_net
        self.latent_linear = nn.Linear(self._D * self.latent_dim, 1)

        # note that a real/fake label is outputted for every step, hence vote to determine label for whole sample
        self.voting_layer = Reduce("b s l -> b l", reduction="mean")

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len = inputs.shape

        embedded_inputs = self.embedding_layer(inputs)
        generated_sequence, hidden_state = self.recurrent_net(embedded_inputs)

        generated_sequence = self.latent_linear(generated_sequence)

        return self.voting_layer(generated_sequence), hidden_state


class LSTMAANDiscriminator(AssociativeDiscriminator):
    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        padding_idx: Optional[int] = None,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
    ) -> None:
        embedding_layer = nn.Embedding(
            n_embeddings, embedding_dim, padding_idx=padding_idx
        )
        lstm = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )

        _model = _LSTMDiscriminator(lstm, embedding_layer)
        optimizer = optimizer_config.optimizer(_model.parameters())
        super().__init__(
            _model,
            activation_fn=nn.Sigmoid(),
            input_size=(()),
            optimizer=optimizer,
            loss_fn=F.binary_cross_entropy_with_logits,
        )

    def __call__(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, hidden_state = self._model(data)
        return logits, hidden_state
