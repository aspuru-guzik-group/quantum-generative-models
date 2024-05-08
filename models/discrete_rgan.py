import typing as t

import orquestra.qml.models.adversarial.th as adversarial
import torch
from einops.layers.torch import Reduce
from orquestra.qml.api import GenerativeModel
from orquestra.qml.models.samplers.th import GaussianSampler, MultiDimGaussianSampler
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn
from torch.nn import RNNBase
from torch.nn import functional as F


# TODO feed the model a vector of embedding_dim in the beginning
# TODO: investigate training generator and discriminator seperately by detaching tensors
class _DiscreteLSTMGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        recurrent_net: RNNBase,
        padding_index: t.Optional[int] = None,
        softmax_temperature: float = 1.0,
    ):
        """Creates an instance of a GRU-based generator component for an adversarial-style network.
        Args:
            vocab_size (int): number of unique tokens in the training corpus.
            recurrent_net (RNNBase): the underlying recurrent neural network to use in the RGAN generator.
                E.g. GRU, LSTM, or a custom rnn-type model.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = recurrent_net.input_size
        self.latent_dim = recurrent_net.hidden_size
        self._D = (
            2 if recurrent_net.bidirectional else 1
        )  # this is needed for computing output dims
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_index,
        )
        self.recurrent_net = recurrent_net
        self.latent_linear = nn.Linear(self._D * self.latent_dim, self.vocab_size)
        self.softmax_temperature = softmax_temperature
        self.gumbel_softmax = F.gumbel_softmax
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs.shape) == 2:
            final_outputs, _ = self._forward_2d(inputs.unsqueeze(1))
        elif len(inputs.shape) == 3:
            final_outputs, _ = self._forward(inputs)
        else:
            raise Exception("Invalid input.")

        return final_outputs

    def _forward_2d(
        self, inputs: torch.Tensor, verbose: bool = False
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        final_outputs = torch.zeros(0)
        final_probabilities = torch.ones((batch_size, 1))
        hidden_state = None
        for i in range(self.seq_len):
            # (batch_size, 1, _D * latent_dim) and (h_n or (h_n, c_n))
            output, hidden_state = self.recurrent_net(inputs, hidden_state)

            # use Gumbel Softmax to sample from categorical distr without destroying gradients
            # shape (batch-size, 1, vocab_size) -> (batch_size, vocab_size)
            selected_tokens_oh = self.gumbel_softmax(
                self.latent_linear(output), hard=True, tau=self.softmax_temperature
            ).squeeze(1)

            # stack the OH encoded vectors
            final_outputs = torch.concat(
                (final_outputs, selected_tokens_oh.unsqueeze(1)), dim=1
            )

            # this gets the embeddings of the tokens at the selected indices
            inputs = selected_tokens_oh.matmul(self.token_embedding.weight.data)
            inputs = inputs.unsqueeze(1)

        # recall that these are sequences of integers
        return final_outputs, final_probabilities

    def _forward(self, noise: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """The model receives as input a 3D tensor of noise of shape (batch_size, seq_len, noise_dim).
        This passes through the RNN, which produces a sequence of shape (batch_size, seq_len, _D * hidden_dim).
        A linear layer then projects that to a 3D tensor of shape (batch_size, seq_len, vocab_size).
        These are raw logits, which we then apply a Gumble Softmax function to, obtaining a 3D tensor corresponding
        to one-hot encoded output sequences. These should still be differentiable.
        """
        # noise has shape (batch_size, seq_len, noise_dim)
        # rnn_output has shape (batch_size, seq_len, _D * hidden_dim)
        rnn_output, *_ = self.recurrent_net(noise)

        # logits has shape (batch_size, seq_len, vocab_size)
        logits = self.latent_linear(rnn_output)

        oh_generated_sequence = self.gumbel_softmax(
            logits, hard=True, tau=self.softmax_temperature
        )

        return oh_generated_sequence, torch.zeros(0)


class _DiscreteLSTMDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        recurrent_net: RNNBase,
        padding_token_index: t.Optional[int] = None,
    ):
        """Creates an instance of a GRU-based generator component for an adversarial-style network.
        Args:
            vocab_size (int): number of unique tokens in the training corpus.
            recurrent_net (RNNBase): the underlying recurrent neural network to use in the RGAN discriminator.
                E.g. GRU, LSTM, or a custom rnn-type model.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dimension = recurrent_net.input_size
        self.latent_dim = recurrent_net.hidden_size
        self._D = (
            2 if recurrent_net.bidirectional else 1
        )  # this is needed for computing output dims

        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dimension, padding_idx=padding_token_index
        )
        self.recurrent_net = recurrent_net
        self.latent_linear = nn.Linear(self._D * self.latent_dim, 1)

        # note that a real/fake label is outputted for every step, hence vote to determine label for whole sample
        self.voting_layer = Reduce("b s l -> b l", reduction="mean")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data is one-hot encoded so
        # embedded_seq = self.embedding(data.argmax(dim=2).long())

        # GRU/RNN output two tensors, but LSTM 3 so we just toss everything but first output
        generated_sequence, *_ = self.recurrent_net(data)
        generated_sequence = self.latent_linear(generated_sequence)

        return self.voting_layer(generated_sequence)


class DiscreteLSTMGenerator(adversarial.Generator):
    def __init__(
        self,
        noise_dim: int,
        sequence_length: int,
        vocab_size: int,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        padding_token_index: t.Optional[int] = None,
        softmax_temperature: float = 1.0,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
    ) -> None:
        """Recurrent, LSTM-based generator for the R(C)GAN.

        Args:
            noise_dim (int): dimension of the noise that will be fed to the generator.
            sequence_length (int): length of sequences in the dataset.
            vocab_size (int): number of unique tokens in the training corpus.
            output_activation (nn.Module): activation function to apply to outputs of the generator.
            hidden_dim (int, optional): latent dimension of the model. Defaults to 128.
            n_layers (int, optional): number of layers in the LSTM. Defaults to 1.
            dropout (float, optional): dropout rate. Defaults to 0.0.
            bidirectional (bool, optional): whether to make the LSTM bidirectional. Defaults to False.
            optimizer_config (TorchOptimizerConfig, optional): configuration of the optimizer to use during training. Defaults to AdamConfig().
        """
        recurrent_net = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        # input should be just the index of the <START> character
        input_shape = (sequence_length, noise_dim)
        model = _DiscreteLSTMGenerator(
            vocab_size,
            sequence_length,
            recurrent_net,
            padding_token_index,
            softmax_temperature=softmax_temperature,
        )
        optimizer = optimizer_config.optimizer(model.parameters())
        super().__init__(model, input_shape, optimizer, loss_fn=nn.BCEWithLogitsLoss())

    def generate_with_probs(
        self, inputs: torch.Tensor, verbose: bool = False
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        output_seq, output_probs = self.model._forward(inputs, verbose)
        return output_seq, output_probs


class DiscreteLSTMDiscriminator(adversarial.Discriminator):
    def __init__(
        self,
        sequence_length: int,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        padding_token_index: t.Optional[int] = None,
    ) -> None:
        """Recurrent, LSTM-based discriminator for the R(C)GAN.

        Args:
            sequence_length (int): length of sequences in the dataset.
            vocab_size (int): number of unique tokens in the training corpus.
            embedding_dim (int): the dimension of the dense embeddings used for tokens.
            hidden_dim (int, optional): latent dimension of the model. Defaults to 128.
            n_layers (int, optional): number of layers in the LSTM. Defaults to 1.
            dropout (float, optional): dropout rate. Defaults to 0.0.
            bidirectional (bool, optional): whether to make the LSTM bidirectional. Defaults to False.
            optimizer_config (TorchOptimizerConfig, optional): configuration of the optimizer to use during training. Defaults to AdamConfig().
        """
        recurrent_net = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        # NOTE: this is the shape of the inputs to the model, not the RNN layer
        input_shape = (sequence_length, vocab_size)
        model = _DiscreteLSTMDiscriminator(
            vocab_size, recurrent_net, padding_token_index=padding_token_index
        )
        optimizer = optimizer_config.optimizer(model.parameters())
        super().__init__(
            model, nn.Sigmoid(), input_shape, optimizer, loss_fn=nn.BCEWithLogitsLoss()
        )


class DiscreteRGAN(adversarial.GAN):
    def __init__(
        self,
        noise_dim: int,
        vocab_size: int,
        sequence_len: int,
        generator_hidden_dim: int = 128,
        gen_n_layers: int = 1,
        gen_bidirectional: bool = False,
        gen_optimizer_config: TorchOptimizerConfig = AdamConfig(),
        disc_hidden_dim: int = 128,
        disc_n_layers: int = 1,
        disc_bidirectional: bool = False,
        disc_optimizer_config: TorchOptimizerConfig = AdamConfig(),
        disc_embedding_dim: int = 64,
        padding_token_index: t.Optional[int] = None,
        use_soft_labels: bool = True,
        label_flip_probability: float = 0.03,
        dropout: float = 0.3,
        softmax_temperature: float = 1.0,
    ) -> None:
        generator = DiscreteLSTMGenerator(
            noise_dim,
            sequence_len,
            vocab_size,
            generator_hidden_dim,
            gen_n_layers,
            dropout,
            gen_bidirectional,
            padding_token_index,
            softmax_temperature,
            gen_optimizer_config,
        )
        discriminator = DiscreteLSTMDiscriminator(
            sequence_len,
            vocab_size,
            disc_embedding_dim,
            disc_hidden_dim,
            disc_n_layers,
            dropout,
            disc_bidirectional,
            disc_optimizer_config,
            padding_token_index=padding_token_index,
        )
        prior = MultiDimGaussianSampler((sequence_len, noise_dim))
        super().__init__(
            generator, discriminator, prior, use_soft_labels, label_flip_probability
        )
