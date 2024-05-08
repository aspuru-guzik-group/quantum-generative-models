from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from orquestra.qml import api
from orquestra.qml.api import TrainResult
from orquestra.qml.models.adversarial.th import (
    QCGAN,
    Discriminator,
    Generator,
    LSTMGenerator,
    TorchAdversarialGenerativeModel,
)
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn
from torch.distributions import Categorical

_DataSelector = Callable[
    [TorchAdversarialGenerativeModel], Tuple[np.ndarray, np.ndarray]
]


def default_conv_block(latent_dim: int) -> List[nn.Module]:
    return [nn.Conv1d(latent_dim, 128, 3), nn.Conv1d(128, 64, 3), nn.Conv1d(64, 32, 3)]


def default_prior_training_data_selector(
    n_prior_training_shots: int = 1000, batch_size: int = 128
) -> _DataSelector:
    """Default prior training data selector function wrapper.
    Returns a function that takes as input a TorchAdversarialGenerativeModel and returns a tuple of
    prior samples and their corresponding probabilities.

    Args:
        n_prior_training_shots (int, optional): number of prior training shots. Defaults to 1000.
        batch_size (int, optional): number of samples in each batch. The total number of batches will be adjusted such that
            the total number of samples is equal to `prior_training_shots`. Defaults to 128.
    """

    def f(model: TorchAdversarialGenerativeModel) -> Tuple[np.ndarray, np.ndarray]:
        batches = [batch_size] * (n_prior_training_shots // batch_size)
        if sum(batches) < n_prior_training_shots:
            batches.append(n_prior_training_shots - sum(batches))

        all_prior_samples: np.ndarray = np.empty(0)
        all_losses: np.ndarray = np.empty(0)
        for idx, sampling_batch_size in enumerate(batches):
            prior_samples = model.prior.generate(sampling_batch_size)

            # generate synthetic data
            synthetic_data = model.generator.generate(prior_samples)

            # infer with discriminator to get loss
            loss_values = (
                model.discriminator.infer(synthetic_data).detach().numpy().squeeze()
            )

            if idx == 0:
                all_prior_samples = prior_samples.detach().numpy()
                all_losses = loss_values
            else:
                all_prior_samples = np.concatenate(
                    [all_prior_samples, prior_samples], axis=0
                )
                all_losses = np.concatenate([all_losses, loss_values], axis=0)

        # create target distribution to match the Distribution interfaces
        # TODO (Brian+Manuel) we currently have no guarantee from the prior, what bitstrings are being mapped to
        # i.e. 0&1 vs. -1&1. We will default to expecting 0&1 for now, and revisit in another PR issue
        # that will circumvent the need to re-normalize the quantum samples by using
        # True&False and requiring the user to set choices in a higher level class (AAN or Trainer classes)
        unique_samples, unique_probs = QCGAN.target_from_losses(
            np.array(all_prior_samples), np.array(all_losses)
        )
        return unique_samples, unique_probs

    return f


def sample_from_mol_logits(logits: torch.FloatTensor) -> torch.LongTensor:
    """Given a tensor of raw logits, applies a softmax to get a probability distribution over the vocabulary,
    and then samples from that distribution to generate a sequence of tokens.

    Args:
        logits (torch.FloatTensor): raw logits outputs of the MolGenerator. Shape: (batch_size, sequence_length, vocab_size)

    Returns:
        torch.LongTensor: sampled tokens. Shape: (batch_size, sequence_length)
    """
    distr = Categorical(logits=logits)
    return distr.sample().long()


class _GeneratorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        recurrent_net: nn.RNNBase,
        latent_activation: nn.Module,
        output_activation: nn.Module,
        embedding_layer: Optional[nn.Module] = nn.Identity(),
    ):
        """Creates an instance of a GRU-based generator component for an adversarial-style network.
        Args:
            vocab_size (int): number of tokens in vocabulary.
                This value will be used as the output dimension of the last trainable layer in the model.
            recurrent_net (RNNBase): pre-configured, recurrent network to use.
            latent_activation (nn.Module): activation function of the hidden layers of the model.
            output_activation (nn.Module): activation function applied to the raw outputs of the model.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_input_dim = recurrent_net.input_size
        self.latent_dim = recurrent_net.hidden_size
        self._D = (
            2 if recurrent_net.bidirectional else 1
        )  # this is needed for computing output dims

        self.embedding_layer = embedding_layer
        self.recurrent_net = recurrent_net
        self.latent_activation = latent_activation
        self.latent_linear = nn.Linear(self._D * self.latent_dim, self.vocab_size)
        self.output_activation = output_activation

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # by default embedding layer is Identity
        # otherwise, the input shape has to be compatible with the specified embedding layer
        output = self.embedding_layer(inputs)

        # GRU/RNN output two tensors, but LSTM 3 so we just toss everything but first output
        generated_sequence, *_ = self.recurrent_net(output)
        generated_sequence = self.latent_activation(generated_sequence)
        generated_sequence = self.latent_linear(generated_sequence)

        return self.output_activation(generated_sequence)


class _DiscriminatorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        latent_dim: int,
        sequence_length: int,
        conv_block: List[nn.Module],
        padding_idx: Optional[int] = None,
        latent_activation_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.padding_idx = padding_idx
        self.act = latent_activation_fn

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        # samples from dataset will be embedded and projected to latent space
        # (pre-sampling) samples from generator will be projected to latent space
        # samples from generator must have vocav_size dimensionality to avoid any additional trainable layers in Generator
        self.real_sample_latent_projection = nn.Linear(embedding_dim, latent_dim)
        self.fake_sample_latent_projection = nn.Linear(vocab_size, latent_dim)

        # convert to ModuleList so that the layers are registered as submodules
        self.conv_block = nn.ModuleList(conv_block)

        # this value depends on the parameters specified at initialization
        # to avoid hardcoding a value we compute it
        flat_dim = self._compute_pre_flat_shape()[-1]

        self.flatten = nn.Flatten()
        self.pre_classifier = nn.Linear(in_features=flat_dim, out_features=flat_dim)
        self.classifier = nn.Linear(in_features=flat_dim, out_features=1)

    def _compute_pre_flat_shape(self) -> Tuple[int, ...]:
        dummy_input = torch.zeros((1, self.latent_dim, self.sequence_length))
        # (B, latent_dim, seq_len) -> (B, ?, ?)
        output = dummy_input
        for layer in self.conv_block:
            output = layer(output)

        # (B, ?, ?) -> (B, ?)
        flat: torch.Tensor = nn.Flatten()(output)
        return flat.shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # a real sample will be a tensor of shape (B, seq_len) so we need to perform embedding
        if inputs.ndim == 2:
            inputs = self.embedding(
                inputs.long()
            )  # (B, seq_len) -> (B, seq_len, embedding_dim)
            inputs = self.real_sample_latent_projection(
                inputs
            )  # (B, seq_len, embedding_dim) -> (B, seq_len, latent_dim)

        # a fake sample (from generator) will be a tensor of shape (B, seq_len, embedding_dim) so we don't need to do anything
        elif inputs.ndim == 3:
            inputs = self.fake_sample_latent_projection(
                inputs
            )  # (B, seq_len, vocab_size) -> (B, seq_len, latent_dim)

        else:
            raise ValueError(
                "Input has unexpected number of dimensions. Must be 2 for 'real' samples or 3 for 'synthetic' samples."
            )

        # to use 1D Conv layers we need shape (B, channels, seq_length)
        inputs = inputs.permute(
            (0, 2, 1)
        ).contiguous()  # (B, seq_len, latent_dim) -> (B, latent_dim, seq_len)
        output = inputs

        # (B, latent_dim, seq_len) -> (B, ?, ?)
        for layer in self.conv_block:
            output = self.act(layer(output))

        # (B, ?, ?) -> (B, ?)
        output = self.flatten(output)

        # (B, ?) -> (B, ?)
        output = self.act(self.pre_classifier(output))

        # (B, ?) -> (B, 1)
        return self.classifier(output)


class MolDiscriminator(Discriminator):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        sequence_length: int,
        embedding_dim: int = 64,
        latent_activation_fn: torch.nn.Module = nn.ReLU(),
        padding_index: Optional[int] = None,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        conv_block: Optional[List[nn.Module]] = None,
        loss_key: str = "discriminator_loss",
    ) -> None:
        """Initializes a 1D convolutional discriminator for use with the molecular drug discovery GAN.

        Args:
            vocab_size (int): number of unique tokens in the vocabulary.
            latent_dim (int): dimensionality of the latent space of the model
            sequence_length (int): length of input molecular sequences.
                Note that all sequences in the dataset must be of the same length.
            embedding_dim (int, optional): dimension of embedding space. Defaults to 64.
            latent_activation_fn (torch.nn.Module, optional): activation function applied in model's latent space. Defaults to nn.ReLU().
            padding_index (Optional[int], optional): index of padding token. Defaults to None.
            optimizer_config (TorchOptimizerConfig, optional): configuration for optimizer. Defaults to AdamConfig().
            conv_block (Optional[List[nn.Module]], optional): layers that will make up the convolutional block of the model.
                The first layer of the block must be a layer with `in_channels` equal to `latent_dim`.
                If left unspecified then the default 3-layer block consisting of
                Conv1d(latend_dim, 64, 3) -> Conv1d(64, 128, 3) -> Conv1d(128, 256, 3) will be used.
                Please note that a `latent_activation_fn` will be applied between each layer in the block (including the default).
                The final arrangement of the convolutional block may look like this:
                Conv1d(latent_dim, 64, 3) -> Relu() -> Conv1d(64, 128, 3) -> Relu() -> Conv1d(128, 256, 3) -> Relu()
            loss_key (str, optional): model loss will be returned under this key. Defaults to "discriminator_loss".
        """
        if conv_block is None:
            conv_block = default_conv_block(latent_dim)

        first_layer: nn.Conv1d = conv_block[0]
        if first_layer.in_channels != latent_dim:
            raise ValueError(
                "The first layer in the convolutional block must have `in_channels` equal to `latent_dim`."
            )

        model = _DiscriminatorModel(
            vocab_size,
            embedding_dim,
            latent_dim,
            sequence_length,
            conv_block,
            padding_index,
            latent_activation_fn,
        )

        optimizer = optimizer_config.optimizer(model.parameters())
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        inference_act_fn = torch.nn.Sigmoid()
        input_size = ((sequence_length, latent_dim),)
        super().__init__(
            model, inference_act_fn, input_size, optimizer, loss_fn, loss_key
        )


class MolGenerator(LSTMGenerator):
    """Generator for MolGAN.
    Wraps around LSTMGenerator for convenience.
    """

    def __init__(
        self,
        noise_dim: int,
        sequence_length: int,
        vocab_size: int,
        output_activation: nn.Module,
        latent_activation: nn.Module = nn.ReLU(),
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = False,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
    ) -> None:
        """Initializes a LSTM based generator for use with the molecular drug discovery GAN.

        Args:
            noise_dim (int): dimension of noise vector generated by sampler and fed as input to generator.
            sequence_length (int): length of input molecular sequences.
            vocab_size (int): number of unique tokens in the vocabulary.
            output_activation (nn.Module): activation function applied to output of generator.
            latent_activation (nn.Module, optional): activation function applied in model's latent space. Defaults to nn.ReLU().
            hidden_dim (int, optional): dimension of the latent space of the LSTM. Defaults to 128.
            n_layers (int, optional): number of layers in the LSTM. Defaults to 1.
            dropout (float, optional): dropout between LSTM layers. Defaults to 0.
            bidirectional (bool, optional): whether to use bi-directional LSTM. Defaults to False.
            optimizer_config (TorchOptimizerConfig, optional): optimizer configuration. Defaults to AdamConfig().
        """
        super().__init__(
            noise_dim,
            sequence_length,
            n_features=vocab_size,
            output_activation=output_activation,
            latent_activation=latent_activation,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            optimizer_config=optimizer_config,
        )


class MolQCGAN(TorchAdversarialGenerativeModel):
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        prior: api.GenerativeModel,
        use_soft_labels: bool = True,
        label_flip_probability: float = 0.03,
        prior_training_data_selector: _DataSelector = default_prior_training_data_selector(),
        model_identifier: str = "MolGAN"
    ):
        """Initialize Molecular QCGAN model for molecular drug discovery.

        Args:
            generator (Generator): component that generates fake data
            discriminator (Discriminator): component that discriminates between real and fake data
            prior (GenerativeModel): component that creates initial distribution to be fed into the generator
                Priors can be trained based on the classified quality of generated samples
            use_soft_labels (bool, optional): whether to add noise to 1s and 0s labels to convert them from `hard` into `soft` labels.
            label_flip_probability (float, optional): the probability of an arbitrary label being flipped from 1 to a 0 or 0 to a 1. This
                has been empirically shown to improve training of GANs.
            prior_training_data_selector (DataSelector, optional): function that selects the data to be used for training the prior.
        """
        super().__init__(
            generator, discriminator, prior, use_soft_labels, label_flip_probability
        )
        self.prior_training_data_selector = prior_training_data_selector
        self.model_identifier = model_identifier

    def _train_prior(self) -> api.TrainResult:
        """Perform one training update on prior"""
        samples, probs = self.prior_training_data_selector(self)
        return self.prior.train_on_batch(
            api.Batch(data=torch.tensor(samples), probs=probs)
        )

    def train_prior(self) -> TrainResult:
        return self._train_prior()
