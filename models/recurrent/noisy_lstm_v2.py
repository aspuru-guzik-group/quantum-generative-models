from typing import List, Optional, Sequence, Tuple

import torch
from orquestra.qml.api import Batch, TorchGenerativeModel, TrainResult
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn
from torch.distributions import Categorical

from ..layers import Concatenate
from .config import NoisyLSTMv2Config


class _Model(nn.Module):
    def __init__(
        self,
        prior_sample_dim: int,
        lstm: nn.LSTM,
        n_embeddings: int,
        embedding_dim: int,
        output_dim: int,
        projection_activation_fn: nn.Module = nn.Identity(),
        output_activation: nn.Module = nn.Identity(),
        padding_token_index: int = 0,
    ) -> None:
        super().__init__()

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.n_directions: int = 2 if lstm.bidirectional else 1
        self.n_layers = lstm.num_layers
        self.hidden_size = lstm.hidden_size

        prior_samples_size = prior_sample_dim
        lstm_hidden_size = lstm.hidden_size

        # we will split projected samples to create hidden_state_0 and cell_state_0
        self.embedding = nn.Embedding(n_embeddings, embedding_dim, padding_token_index)
        self.linear_projection = nn.Linear(prior_samples_size, lstm_hidden_size)
        self.linear_projection_activation = projection_activation_fn
        self.concatenate = Concatenate(dim=-1)
        self.recurrent_net = lstm
        self.output_classifier = nn.Linear(
            self.n_directions * lstm_hidden_size, output_dim
        )
        self.output_activation = output_activation

    def forward(
        self,
        inputs: torch.Tensor,
        prior_samples: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Completes a forward pass through the model.

        Args:
            inputs (torch.Tensor): input tensor of sequences of integers, with shape (batch_size, sequence_length)
            prior_samples (torch.Tensor): samples from a prior, with shape (batch_size, sample_dimension).
            hidden_state (Optional[Tuple[torch.Tensor, torch.Tensor]]): tuple of tensors comprising the initial hidden and cell states.
                Each tensor should have shape (n_directions * n_layers, batch_size, hidden_size).
                Note: n_directtions is either 1 or 2 for a uni- and bi- directional LSTMs respectively, while n_layers is the number
                of stacked LSTM layers.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: _description_
        """

        batch_size, sequence_length = inputs.shape
        _, sample_dimension = prior_samples.shape

        h0, c0 = hidden_state

        # projected_samples[n_layers, B, 2 * hidden_size]
        projected_samples = self.linear_projection(prior_samples)
        projected_samples = self.linear_projection_activation(projected_samples)

        # add projected samples to c0 and h0
        h0 += projected_samples
        c0 += projected_samples

        # embed input sequences (2D -> 3D), shape: (batch_size, seq_len, embedding_dim)
        embedded_inputs = self.embedding(inputs)

        lstm_output, (h1, c1) = self.recurrent_net.forward(embedded_inputs, (h0, c0))

        class_logits = self.output_classifier(lstm_output)

        return self.output_activation(class_logits), (h1, c1)


class NoisyLSTMv2(TorchGenerativeModel):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        sos_token_index: int,
        prior_sample_dim: int,
        padding_token_index: Optional[int] = None,
        projection_activation_fn: nn.Module = nn.Identity(),
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        loss_key: str = "loss",
        model_identifier: str = "noisy-lstm-v2",
    ) -> None:
        super().__init__()
        recurrent_net = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=False,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.sos_token_index = sos_token_index
        self.padding_token_index = padding_token_index
        self._input_size = (seq_len,)

        self._model: _Model = _Model(
            prior_sample_dim,
            recurrent_net,
            n_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            output_dim=vocab_size,
            padding_token_index=padding_token_index,
            projection_activation_fn=projection_activation_fn,
        )

        self.optimizer = optimizer_config.optimizer(self._model.parameters())
        self.loss_fn = nn.NLLLoss()
        self.loss_key = loss_key
        self.model_identifier = model_identifier

    def __call__(
        self,
        inputs: torch.Tensor,
        prior_samples: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # outputs[0] -> (batch_size, sequence_length, vocab_size)
        # outputs[1] -> (h1, c1) -> (n_directions * n_layers, batch_size, hidden_size)
        return self._model.forward(inputs, prior_samples, hidden_state)

    def _make_xo(self, n_samples: int) -> torch.Tensor:
        return torch.full((n_samples, 1), self.sos_token_index).to(self._device)

    def _make_initial_hidden_state(
        self, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates initial hidden state and cell state."""
        h0 = self._make_h0(n_samples).to(self._device)
        c0 = self._make_c0(n_samples).to(self._device)
        return h0, c0

    def _make_c0(self, batch_size: int) -> torch.Tensor:
        """Returns a tensor of zeros to act as initial cell to an RNN (usually LSTM).
        Args:
            batch_size (int): size of the batch to ensure shape match.
        Returns:
            torch.Tensor: initial cell state, either all zeros or sampled from unit Gaussian distribution.
        """
        memory_vec_shape = (
            self._model.n_directions * self._model.n_layers,
            batch_size,
            self._model.hidden_size,
        )
        return torch.zeros(size=memory_vec_shape).to(self._device)

    def _make_h0(self, batch_size: int) -> torch.Tensor:
        """Generates a tensor of all zeros to act as the initial hidden state to the RNN.
        Args:
            batch_size (int): size of the batch to ensure shape match.
        Returns:
            torch.Tensor: initial hidden state, either all zeros or sampled from unit Gaussian distribution.
        """
        memory_vec_shape = (
            self._model.n_directions * self._model.n_layers,
            batch_size,
            self._model.hidden_size,
        )

        return torch.zeros(size=memory_vec_shape)

    def get_token_embeddings(self, token_indices: Sequence[int]) -> torch.Tensor:
        input_tensor = torch.tensor(token_indices).view((-1, 1))
        with torch.no_grad():
            embeddings = self._model.embedding_layer(input_tensor)
        return embeddings

    def train_on_batch(self, batch: Batch) -> TrainResult:
        return self._train_on_batch(batch.convert_to_torch(self._device))

    def _train_on_batch(self, batch: Batch[torch.Tensor]) -> TrainResult:
        data = batch.data
        noise = batch.targets

        if noise is None:
            raise ValueError("Recieved None for hidden state.")

        self.set_train_state()
        self.optimizer.zero_grad()

        if len(data.size()) != 2:
            raise ValueError(
                f"Expected 2D tensor as input, but got {len(data.size())}D tensor."
            )

        batch_size, seq_len = data.shape

        # first element will be the <START> token
        # append to that everything but last element of sequence
        x0 = self._make_xo(batch_size).to(data.device)
        model_inputs = torch.concat((x0, data[:, : seq_len - 1]), dim=1).long()

        hidden_sate = self._make_initial_hidden_state(batch.batch_size)

        # output -> (batch_size, sequence_length, vocab_size)
        generated_sequence, *_ = self.__call__(model_inputs, noise, hidden_sate)

        # permute to fit shape needed by NLL loss function
        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
        generated_sequence_t = generated_sequence.permute(0, 2, 1)
        loss = self.loss_fn(generated_sequence_t, data.long())  # (B, L)

        loss.backward()
        self.optimizer.step()

        return {self.loss_key: loss.detach()}

    def _generate_w_probs(
        self, prior_samples: torch.Tensor, random_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.
        Args:
            prior_samples (torch.Tensor): samples from prior, with shape (n_samples, sample_dimensions).
            random_seed (Optional[int], optional): an optional random seed for reproducibility. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the raw generated sequences and the associated probabilities.
        """
        random_generator = None
        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        n_samples = prior_samples.shape[0]

        inputs = self._make_xo(n_samples).to(self._device)  # (batch_size, 1)
        hidden_state = self._make_initial_hidden_state(n_samples)
        outputs = torch.zeros((n_samples, self.seq_len)).to(self._device)
        seq_probabilities = torch.ones((n_samples, 1)).to(
            self._device
        )  # TODO: update probs

        with torch.no_grad():
            for index in range(0, self.seq_len):
                # class_logit_sequence -> [batch_size, 1, vocab_size]
                # hidden_state -> (h1, c1) -> [n_directions * n_layers, batch_size, hidden_size]
                class_logit_sequence, hidden_state = self.__call__(
                    inputs, prior_samples, hidden_state
                )

                # create a distribution for easier sampling, recall that default output activation is Identity
                # for that reason we get back logits
                cat_distribution = Categorical(logits=class_logit_sequence.squeeze(1))

                # outputs -> (batch_size, )
                sampled_token_indices = cat_distribution.sample()
                outputs[:, index] = sampled_token_indices

                # inputs -> (batch_size, 1)
                inputs = sampled_token_indices.unsqueeze(1)

        return outputs, seq_probabilities

    @property
    def _models(self) -> List[nn.Module]:
        return [self._model]

    @property
    def sample_size(self) -> Tuple[int, ...]:
        h0 = self._make_h0(1)
        c0 = self._make_c0(1)
        hidden_state = torch.cat([h0, c0], dim=2)

        return tuple(self.generate(1, hidden_state).shape[1:])

    @property
    def input_size(self) -> Tuple[int, ...]:
        return self._input_size

    @property
    def config(self) -> NoisyLSTMv2Config:
        """Returns model configuration."""
        d = {
            "vocab_size": self.vocab_size,
            "n_embeddings": self._model.n_embeddings,
            "embedding_dim": self._model.embedding_dim,
            "latent_dim": self._model.hidden_size,
            "n_layers": self._model.n_layers,
            "dropout": self._model.recurrent_net.dropout,
            "bidirectional": self._model.n_directions,
            "padding_token_index": self.padding_token_index,
        }
        config = NoisyLSTMv2Config(**d)
        return config

    def _generate(
        self, n_samples: int, random_seed: Optional[int] = None
    ) -> torch.Tensor:
        return super()._generate(n_samples, random_seed)

    def generate(
        self, prior_samples: torch.Tensor, random_seed: Optional[int] = None
    ) -> torch.Tensor:  # type: ignore
        try:
            n_samples, sample_dimension = prior_samples.shape
        except Exception as e:
            raise ValueError(
                f"Prior samples must be a 2-D tensor with shape (n_samples, sample_dimension). Got {prior_samples.shape}"
            ) from e

        generated_sequences, probs = self._generate_w_probs(prior_samples, random_seed)
        return generated_sequences
