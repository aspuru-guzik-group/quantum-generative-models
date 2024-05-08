from dataclasses import dataclass
from typing import List, Optional, Tuple
from warnings import warn

import torch
from orquestra.qml.api import (Batch, TorchGenerativeModel, TrainResult,
                               convert_to_torch)
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn
from torch.distributions import Categorical

from ..layers import Concatenate
from .config import NoisyLSTMv2Config

# TODO: initialize sample differently


@dataclass
class NoisyLSTMv3Config(NoisyLSTMv2Config):
    pass


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
        lstm_input_dim = lstm.input_size
        lstm_hidden_size = lstm.hidden_size

        self.projection_dim = lstm_input_dim - embedding_dim
        self.embedding = nn.Embedding(n_embeddings, embedding_dim, padding_token_index)
        self.linear_projection = nn.Linear(prior_samples_size, self.projection_dim)
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
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.

        Args:
            inputs (torch.Tensor): input sequence of integers, where each integer corresponds to a token in a corpus. Shape: (B, SL).
            prior_samples (torch.Tensor): samples from a prior distribution. Shape: (B, ND), where <B> is the batch size and <ND> is the dimension of a single sample.
            hidden_state (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): initial hidden state. Defaults to None.
                Expected shape (for each element of tuple): (B, D*n_layers, H), where <B> is the batch size and <H> is the hidden size of the LSTM,
                and <D> is 1 if the LSTM is unidirectional, and 2 if it is bidirectional.
                This is different to the default shape of the hidden state returned by the LSTM, which is (D*n_layers, B, H). The transposed
                variation is compatible with DataParallel, which splits along dimension 0. Hidden state will be transposed before being
                passed to the model (swap dimensions 0 and 1).
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: sequence of class logits, and the final hidden state.
        """
        # noise shape: (B, ND)
        # inputs shape: (B, SL)
        batch_size, sequence_length = inputs.shape
        _, noise_dim = prior_samples.shape

        # can only concat similar tensors so we first expand to 3D, then repeat to match shape of input
        # noise shape: (B, SL, ND)
        prior_samples = prior_samples.view((batch_size, 1, noise_dim)).expand(
            (batch_size, sequence_length, noise_dim)
        )

        # (B, SL, ProjDim)
        prior_samples = self.linear_projection(prior_samples)
        prior_samples = self.linear_projection_activation(prior_samples)

        # (B, SL, EmbDim)
        outputs = self.embedding(inputs)

        # (B, SL, ProjDim + EmbDim), note that ProjDim + EmbDim = LSTM Input Dim
        outputs = self.concatenate(outputs, prior_samples)

        # do the transpose as explained in the docstring if necessary
        if hidden_state is not None:
            (
                h,
                c,
            ) = hidden_state  # (batch_size, D*n_layers, hidden_size), (D*n_layers, batch_size, hidden_size)
            h = h.transpose_(0, 1).contiguous()  # (D*n_layers, batch_size, hidden_size)
            c = c.transpose_(0, 1).contiguous()  # (D*n_layers, batch_size, hidden_size)
            hidden_state = (
                h,
                c,
            )  # (D*n_layers, batch_size, hidden_size), (D*n_layers, batch_size, hidden_size)

        outputs, hidden_state = self.recurrent_net(outputs, hidden_state)

        # transpose hidden state again
        h: torch.Tensor
        c: torch.Tensor
        (
            h,
            c,
        ) = hidden_state  # (D*n_layers, batch_size, hidden_size), (D*n_layers, batch_size, hidden_size)
        h = h.transpose_(0, 1).contiguous()  # (batch_size, D*n_layers, hidden_size)
        c = c.transpose_(0, 1).contiguous()  # (batch_size, D*n_layers, hidden_size)
        hidden_state = (
            h,
            c,
        )  # (batch_size, D*n_layers, hidden_size), (batch_size, D*n_layers, hidden_size)

        # these will be class logits
        outputs = self.output_classifier(outputs)

        return self.output_activation(outputs), hidden_state


class NoisyLSTMv3(TorchGenerativeModel):
    """Implements a "noisy" LSTM version 3.
    This model accepts a sample tensor from a prior (such as an RBM or QCBM) and
    concatenates that tensor with the (embedded) input sequence. The same sample
    is concatenated to each step in the sequence.

    Example:
        Given an input sequence <i(t)>, with length <L>, and shape (B, L), where <B> is
            the size of a batch consisting of integers, corresponding to tokens in a corpus.
        Given a batch of samples from a prior, with shape (B, D), where <D> is the dimension of
            an individual sample.
        The input sequence <i(t)> will be embedded into a 3D tensor of shape (B, L, ED), <ED> is the embedding
            dimension (a hyper-parameter).
        This embedded sequence will then be concatenated with the the prior samples to form a new sequence, <u(t)>,
            such that for a given step t_n, u(t_n) = [i(t_n), prior_sample]. The resultant sequence <u(t)> will have shape (B, L, ED + D).
        This sequence <u(t)> will then be passed through the model.

    Summary:
        p ==> |PRIOR| : (batch_size, prior_sample_dim) -> |LINEAR PROJECTION| : (B, ProjDim) ->
        i(t) ==> |INPUT| : (B, L) -> |EMBEDDING|: (B, L, EmdDim) ->
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        sos_token_index: int,
        prior_sample_dim: int,
        padding_token_index: Optional[int] = None,
        prior_sample_projection_dim: int = 64,
        projection_activation_fn: nn.Module = nn.Identity(),
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        loss_key: str = "loss",
        model_identifier: str = "noisy-lstm-v3",
        do_greedy_sampling: bool = False,
        sampling_temperature: float = 1.0,
    ) -> None:
        """Initializes the model.

        Args:
            vocab_size (int): the number of unique tokens in the vocabulary.
            seq_len (int): the maximum sequence length.
            sos_token_index (int): the index of the start-of-sentence (SOS) token.
            prior_sample_dim (int): the dimension of the samples generated by the prior.
            padding_token_index (Optional[int], optional): index of the padding token. Defaults to None.
            prior_sample_projection_dim (int, optional): output dimension of a linear layer that will be used to project the prior samples. Defaults to 64.
            projection_activation_fn (nn.Module, optional): activation function applied after the linear layer projecting prior samples. Defaults to nn.Identity().
            embedding_dim (int, optional): the dimension of each token embedding vector. Defaults to 64.
            hidden_dim (int, optional): the dimension of the hidden layers of the LSTM. Defaults to 128.
            n_layers (int, optional): number of layers in the LSTM. Defaults to 1.
            dropout (float, optional): dropout applied between consecutive LSTM layers. Defaults to 0.0.
                Note that this will only have an effect if there are more than 1 stacked LSTM layers.
            optimizer_config (TorchOptimizerConfig, optional): configuration of the optimizer. Defaults to AdamConfig().
            loss_key (str, optional): key which will contain the model's training loss. Defaults to "loss".
            model_identifier (str, optional): a string to identify the model. Defaults to "noisy-lstm-v3".
            do_greedy_sampling (bool, optional): whether to use greedy sampling. This will always select
                the token with the highest probability at each step when generating samples. Defaults to False.
            sampling_temperature (float, optional): temperature to use when generating samples. This parameter
                will have no effect if <greedy_sampling> is set to True. Higher value will cause the sampling
                distribution to be "flatter", i.e. more uniform, while a lower value will emphasize the most
                probable tokens. Defaults to 1.0.
        """
        super().__init__()
        lstm = nn.LSTM(
            input_size=embedding_dim + prior_sample_projection_dim,
            hidden_size=hidden_dim,
            bidirectional=False,
            batch_first=True,  # using False helps alleviate headaches with DataParallel and hidden state shapes
            num_layers=n_layers,
            dropout=dropout,
        )

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.sos_token_index = sos_token_index
        self.padding_token_index = padding_token_index
        self._input_size = (seq_len,)
        self.prior_sample_dim = prior_sample_dim

        if (
            not isinstance(sampling_temperature, (float, int))
            or sampling_temperature <= 0.0
        ):
            raise ValueError(
                f"Sampling temperature must be a positive number, got {sampling_temperature}"
            )
        self._sampling_temperature = float(sampling_temperature)

        if not isinstance(do_greedy_sampling, bool):
            raise ValueError(
                f"Greedy sampling must be a boolean, got {do_greedy_sampling}"
            )

        if do_greedy_sampling and sampling_temperature != 1.0:
            warn("Sampling temperature will be ignored when greedy sampling is enabled")

        self._do_greedy_sampling = do_greedy_sampling

        # model-specific attributes cannot be accessed when wrapped with DataParallel hence we
        # set them at the level of the interface
        self.n_directions: int = 2 if lstm.bidirectional else 1
        self.n_layers = lstm.num_layers
        self.hidden_size = lstm.hidden_size
        self.n_embeddings = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.projection_dim = prior_sample_projection_dim

        self._model: _Model = _Model(
            prior_sample_dim=prior_sample_dim,
            lstm=lstm,
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
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # outputs[0] -> (batch_size, sequence_length, vocab_size)
        # outputs[1] -> hidden state

        sequences, hidden_state = self._model(inputs, prior_samples, hidden_state)
        return sequences, hidden_state

    def _make_xo(self, n_samples: int) -> torch.Tensor:
        # next: try with noise instead of
        return torch.full((n_samples, 1), self.sos_token_index).to(self._device)

    def _make_initial_hidden_state(
        self, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates initial hidden state and cell state."""
        h0 = self._make_h0(n_samples)
        c0 = self._make_c0(n_samples)
        return h0, c0

    def _make_c0(self, batch_size: int) -> torch.Tensor:
        """Returns a tensor of zeros to act as initial cell to an RNN (usually LSTM).
        Args:
            batch_size (int): size of the batch to ensure shape match.
        Returns:
            torch.Tensor: initial cell state, either all zeros or sampled from unit Gaussian distribution.
        """
        memory_vec_shape = (
            batch_size,
            self.n_directions * self.n_layers,
            self.hidden_size,
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
            batch_size,
            self.n_directions * self.n_layers,
            self.hidden_size,
        )

        return torch.zeros(size=memory_vec_shape).to(self._device)

    def train_on_batch(self, batch: Batch) -> TrainResult:
        return self._train_on_batch(batch.convert_to_torch(self._device))

    def _train_on_batch(self, batch: Batch[torch.Tensor]) -> TrainResult:
        # data = batch.data
        # prior_samples = batch.targets

        # data tensor should be of type Long because we are working with discrete tokens
        batch.data = batch.data.long()

        if batch.targets is None:
            raise ValueError("No `targets` present in batch.")

        self.set_train_state()
        self.optimizer.zero_grad()

        if len(batch.data.size()) != 2:
            raise ValueError(
                f"Expected 2D tensor as input, but got {len(batch.data.size())}D tensor."
            )

        batch_size, seq_len = batch.data.shape

        # first element will be the <START> token
        # append to that everything but last element of sequence
        x0 = self._make_xo(batch_size)
        inputs = torch.concat((x0, batch.data[:, : seq_len - 1]), dim=1).long()

        # output -> (batch_size, sequence_length, vocab_size)
        # we utilize the `targets` field as the initial hidden state (noise) for the LSTM
        # outputs are the class logits
        outputs, *_ = self.__call__(inputs, batch.targets)

        # apply log softmax to convert logits to log probabilities
        outputs = nn.LogSoftmax(-1)(outputs)

        # permute to fit shape needed by NLL loss function
        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
        outputs = outputs.permute(0, 2, 1)

        loss = self.loss_fn(outputs, batch.data)  # (B, L)

        loss.backward()
        self.optimizer.step()

        train_result = {self.loss_key: loss.item()}

        # explicitly delete tensors to free up memory
        del loss, inputs, x0, outputs

        return train_result

    def _generate_w_probs(
        self,
        n_samples: int,
        prior_samples: torch.Tensor,
        random_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.

        Args:
            n_samples (int): then number of samples to generate.
            random_seed (Optional[int], optional): an optional random seed for reproducibility. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the raw generated sequences and the associated probabilities.
        """
        random_generator = None

        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        inputs = self._make_xo(n_samples)  # (batch_size, 1)
        hidden_state: Tuple[
            torch.Tensor, torch.Tensor
        ] = self._make_initial_hidden_state(n_samples)
        outputs = torch.zeros((n_samples, self.seq_len)).to(self._device)
        seq_probabilities = torch.ones((n_samples, 1))  # TODO: update probs

        with torch.no_grad():
            for index in range(0, self.seq_len):
                # class_logit_sequence -> (batch_size, 1, vocab_size)
                # hidden_state -> (batch_size, D*n_layers, hidden_size), (batch_size, D*n_layers, hidden_size)
                class_logit_sequence, hidden_state = self.__call__(
                    inputs, prior_samples, hidden_state
                )

                # select element with highest probability
                # or sample from the distribution
                if self._do_greedy_sampling:
                    # sampled_token_indices -> (batch_size, )
                    sampled_token_indices = torch.argmax(
                        class_logit_sequence.squeeze(1), dim=-1
                    )
                else:
                    cat_distribution = Categorical(
                        logits=class_logit_sequence.squeeze(1)
                        / self.sampling_temperature
                    )

                    # sampled_token_indices -> (batch_size, )
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
        prior_samples = torch.zeros((1, self.prior_sample_dim)).to(self._device)
        generated_samples = self.generate(prior_samples)
        return tuple(generated_samples.shape[1:])

    @property
    def input_size(self) -> Tuple[int, ...]:
        return self._input_size

    @property
    def config(self) -> NoisyLSTMv3Config:
        """Returns model configuration."""
        d = {
            "name": self.model_identifier,
            "vocab_size": self.vocab_size,
            "projection_dim": self.projection_dim,
            "n_embeddings": self.n_embeddings,
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "bidirectional": self.n_directions > 1,
            "padding_token_index": self.padding_token_index,
        }
        config = NoisyLSTMv3Config(**d)
        return config

    @property
    def do_greedy_sampling(self) -> bool:
        return self._do_greedy_sampling

    @do_greedy_sampling.setter
    def do_greedy_sampling(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Greedy sampling must be a boolean. Got {type(value)}.")

        self._do_greedy_sampling = bool(value)

    @property
    def sampling_temperature(self) -> float:
        return self._sampling_temperature

    @sampling_temperature.setter
    def sampling_temperature(self, value: float) -> None:
        if not isinstance(value, (float, int)) or value <= 0.0:
            raise ValueError(
                f"Sampling temperature must be a positive number, got {value}"
            )

        if value <= 0:
            raise ValueError(f"Sampling temperature must be positive. Got {value}.")

        self._sampling_temperature = value

    def _generate(
        self, n_samples: int, random_seed: Optional[int] = None
    ) -> torch.Tensor:
        return super()._generate(n_samples, random_seed)

    def generate(
        self, prior_samples: torch.Tensor, random_seed: Optional[int] = None
    ) -> torch.Tensor:  # type: ignore
        n_samples = prior_samples.shape[0]
        prior_samples = convert_to_torch(prior_samples).to(self._device)
        generated_sequences, probs = self._generate_w_probs(
            n_samples, prior_samples, random_seed
        )
        return generated_sequences

    def enable_greedy_sampling(self) -> None:
        """Enables greedy sampling."""
        self._do_greedy_sampling = True

    def disable_greedy_sampling(self) -> None:
        """Disables greedy sampling."""
        self._do_greedy_sampling = False
