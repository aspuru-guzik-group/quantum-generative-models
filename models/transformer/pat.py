import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from orquestra.qml.api import Batch, TorchGenerativeModel, TrainResult
from orquestra.qml.functional.layers.th import PositionalEncoding
from orquestra.qml.functional.layers.th.core import Repeat
from orquestra.qml.losses.th import LossFunction
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn
from torchinfo import summary
from torchinfo.model_statistics import ModelStatistics

from ..layers import Concatenate


@dataclass
class PATConfig:
    model_name: str
    prior_dim: int
    hidden_dim: int
    dim_feed_forward: int
    n_attention_heads: int
    n_encoder_layers: int
    hidden_activation_fn_name: str
    output_activation_fn_name: str
    dropout: float

    _custom_attributes = []

    def as_dict(self) -> Dict[str, Any]:
        fields = {
            "prior_dim": self.prior_dim,
            "hidden_dim": self.hidden_dim,
            "dim_feed_forward": self.dim_feed_forward,
            "n_attention_heads": self.n_attention_heads,
            "n_encoder_layers": self.n_encoder_layers,
            "hidden_activation_fn_name": self.hidden_activation_fn_name,
            "output_activation_fn_name": self.output_activation_fn_name,
            "dropout": self.dropout,
        }

        for attr in self._custom_attributes:
            fields[attr] = getattr(self, attr)

        return fields

    def _set_fields(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self._custom_attributes.append(key)

    def get_field(self, field_name: str) -> Any:
        return getattr(self, field_name)


class _DiscretePATModel(nn.Module):
    """Prior Assisted Transformer Model for discrete data.
    Implements a encoder-only transformer model that takes as input
    a sequence of values and samples from a prior distribution.
    """

    def __init__(
        self,
        n_tokens: int,
        prior_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dim_feed_forward: int,
        n_attention_heads: int,
        n_encoder_layers: int,
        hidden_activation_fn: nn.Module,
        output_activation_fn: nn.Module,
        dropout: float = 0.0,
        padding_token_index: Optional[int] = None,
    ) -> None:
        """Initialize Prior Assisted Transformer Model for discrete data.

        Args:
            n_tokens (int): number of tokens in the vocabulary.
            prior_dim (int): dimensionality of the samples from the prior distribution.
            embedding_dim (int): dimension to use for token embeddings.
            hidden_dim (int): dimension to use for hidden layers of the model.
            dim_feed_forward (int): dimension to use for feed forward layers of the model.
            n_attention_heads (int): number of attention heads to use in the model.
            n_encoder_layers (int): number of layers in the encoder stack.
            hidden_activation_fn (nn.Module): activation function applied inside the feed forward layers.
            output_activation_fn (nn.Module): activation function applied to the raw outputs of the model.
            dropout (float, optional): fraction of nodes to "drop out". Defaults to 0.0.
            padding_token_index (int, optional): index of the padding token in the vocabulary.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            n_tokens, embedding_dim, padding_idx=padding_token_index
        )
        self.concatenate = Concatenate()

        self.pre_pe_projection = nn.Linear(embedding_dim + prior_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            n_attention_heads,
            dim_feed_forward,
            dropout,
            batch_first=True,
            activation=hidden_activation_fn,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.output_head = nn.Linear(hidden_dim, n_tokens)
        self.output_activation_fn = output_activation_fn

    def forward(
        self,
        x: torch.LongTensor,
        prior_samples: torch.FloatTensor,
        attn_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs forward pass through the model.

        Args:
            x (torch.LongTensor): input sequence of token indices with shape (batch_size, seq_len) and dtype:int64
            prior_samples (torch.FloatTensor): samples from a prior distribution with shape (batch_size, sample_dim) and dtype:float32.
            attn_mask (Optional[torch.Tensor], optional): if specified, a 2D or 3D mask of shape (seq_len,) or
                to use in self-attention layers. This mask will ensure that the model cannot "cheat" by observing future tokens.
            src_key_padding_mask (Optional[torch.Tensor], optional): if specified, a mask of shape :math:`(N, S)`
                indicating which elements within ``key`` to ignore for the purpose of attention (i.e. treat as "padding").
        Returns:
            torch.Tensor: output logits with shape (batch_size, seq_len, n_tokens) and dtype:float32.
        """
        if x.dtype != torch.long:
            raise ValueError(
                f"Expected input sequence to be of dtype:torch.long, but got {x.dtype} instead. Please convert to torch.long before passing to the model."
            )

        if prior_samples.dtype != torch.float32:
            raise ValueError(
                f"Expected prior samples to be of dtype:torch.float32, but got {prior_samples.dtype} instead. Please convert to torch.float32 by calling before passing to the model."
            )

        # x: (batch_size, seq_len), prior_samples: (batch_size, sample_dim)

        # x_emb: (batch_size, seq_len, embedding_dim)
        x_emb = self.embedding(x)

        # x_emb: (batch_size, seq_len, sample_dim + embedding_dim)
        repeater = Repeat("b o d -> b (repeat o) d", repeat=x_emb.shape[1])
        x_emb = self.concatenate([x_emb, repeater(prior_samples.unsqueeze(1))])

        # x_emb: (batch_size, seq_len, hidden_dim)
        x_emb = self.pre_pe_projection(x_emb)

        # x_emb: (batch_size, seq_len, hidden_dim)
        x_emb = self.positional_encoding(x_emb)

        # x: (batch_size, seq_len, n_tokens)
        x = self.encoder(
            x_emb, src_key_padding_mask=src_key_padding_mask, mask=attn_mask
        )
        x = self.output_head(x)

        return self.output_activation_fn(x)


class MolPAT(TorchGenerativeModel):
    """Molecular Prior-Assisted Transformer Model."""

    _repr_fields = []

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        prior_dim: int,
        start_token_index: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dim_feed_forward: int = 128,
        n_attention_heads: int = 2,
        n_encoder_layers: int = 1,
        hidden_activation_fn: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        padding_token_index: int = 0,
        loss_key: str = "loss",
        model_identifier: str = "MolPAT",
    ) -> None:
        """Initializes the model.

        Args:
            vocab_size (int): then number of tokens in the vocabulary.
            seq_len (int): the length of the input sequence.
            prior_dim (int): the dimensionality of the prior samples.
            start_token_index (int): the index of the start token in the vocabulary.
            embedding_dim (int, optional): the dimension of the token embeddings. Defaults to 64.
            hidden_dim (int, optional): the dimension of the hidden layers in the model. Defaults to 128.
            dim_feed_forward (int, optional): the dimension of the feed forward layer in the model that comes at the end of each encoder block. Defaults to 128.
            n_attention_heads (int, optional): the number of attention heads. Defaults to 2.
            n_encoder_layers (int, optional): the number of encoder layers. Defaults to 1.
            hidden_activation_fn (nn.Module, optional): the activation function to use in the hidden layers. Defaults to nn.ReLU().
            dropout (float, optional): the dropout probability. Defaults to 0.0.
            optimizer_config (TorchOptimizerConfig, optional): configuration for the optimizer. Defaults to AdamConfig().
            padding_token_index (int, optional): index of the padding token. Defaults to 0.
            loss_key (str, optional): the key to use for the loss in the output dictionary of the `train_on_batch` method. Defaults to "loss".
            model_identifier (str, optional): the identifier for the model. Defaults to "MolPAT".
        """
        super().__init__()
        output_activation_fn = nn.LogSoftmax(dim=-1)
        loss_fn: LossFunction = nn.functional.nll_loss
        self._model = _DiscretePATModel(
            vocab_size,
            prior_dim,
            embedding_dim,
            hidden_dim,
            dim_feed_forward,
            n_attention_heads,
            n_encoder_layers,
            hidden_activation_fn,
            output_activation_fn,
            dropout,
            padding_token_index,
        )
        self.optimizer = optimizer_config.optimizer(self._model.parameters())
        self.loss_fn = loss_fn
        self.loss_key = loss_key
        self.start_token_index = start_token_index
        self.padding_token_index = padding_token_index
        self.seq_len = seq_len
        self.model_identifier = model_identifier

        self._config = PATConfig(
            model_name="MolPAT",
            prior_dim=prior_dim,
            hidden_dim=hidden_dim,
            dim_feed_forward=dim_feed_forward,
            n_attention_heads=n_attention_heads,
            n_encoder_layers=n_encoder_layers,
            hidden_activation_fn_name=hidden_activation_fn.__class__.__name__,
            output_activation_fn_name=output_activation_fn.__class__.__name__,
            dropout=dropout,
        )
        self._config._set_fields(
            start_token_index=start_token_index,
            padding_token_index=padding_token_index,
            seq_len=seq_len,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

    def __call__(
        self,
        data: torch.Tensor,
        prior_samples: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes a forward pass through the model and returns an output sequence.

        Args:
            data (torch.Tensor): input sequence of token indices with shape (batch_size, seq_len) and dtype:int64.
            prior_samples (torch.Tensor): samples from a prior distribution with shape (batch_size, sample_dim) and dtype:float32.
            attn_mask (torch.Tensor, optional): a "look-ahead" mask to prevent attention over future tokens.
            padding_mask (torch.Tensor, optional): a mask to prevent attention over padding tokens.
        """
        return self._model(data, prior_samples, attn_mask, padding_mask)

    def _raise_error_if_invalid_train_batch(self, batch: Batch[torch.Tensor]) -> None:
        """Performs a sanity check on the batch to ensure it is valid for training, and raises an error if not.

        Args:
            batch (Batch[torch.Tensor]): a batch of data to be used for training.
        """
        if batch.targets is None:
            raise ValueError(
                "Expected targets to be provided for training. Please set samples from prior distribution as targets."
            )

        if len(batch.data.size()) != 2:
            raise ValueError(
                f"Expected 2D tensor as input, but got {len(batch.data.size())}D tensor."
            )

        if batch.data.dtype != torch.long:
            raise ValueError(
                f"Expected input sequence to be of dtype:torch.long, but got {batch.data.dtype} instead. Please convert to torch.long before passing to the model."
            )

        if batch.targets.dtype != torch.float32:
            raise ValueError(
                f"Expected prior samples to be of dtype:torch.float32, but got {batch.targets.dtype} instead. Please convert to torch.float32 by calling before passing to the model."
            )

    def _make_x0(self, n_samples: int) -> torch.LongTensor:
        """Returns a tensor to be used as the initial input to the model."""
        return torch.full(
            (n_samples, 1),
            self.start_token_index,
            dtype=torch.long,
            device=self._device,
        )

    def _train_on_batch(self, batch: Batch[torch.Tensor]) -> TrainResult:
        """Trains the model on a single batch of data and returns the associated loss."""
        self._raise_error_if_invalid_train_batch(batch)

        self.set_train_state()
        self.optimizer.zero_grad()

        data: torch.LongTensor = batch.data
        prior_samples: torch.FloatTensor = batch.targets

        batch_size, seq_len = batch.data.shape

        # first element will be the <START> token
        # append to that everything but last element of sequence
        x0: torch.LongTensor = self._make_x0(batch_size)
        model_inputs = torch.concat((x0, batch.data[:, : seq_len - 1]), dim=1)

        pad_mask = self.make_padding_mask(model_inputs)
        attn_mask = self.make_lookahead_mask(model_inputs)

        # using __call__ because otherwise I don't see signature hints in VSCODE
        # generated_sequence -> (batch_size, sequence_length, vocab_size)
        generated_sequence = self.__call__(
            model_inputs, prior_samples, attn_mask, pad_mask
        )

        # permute to fit shape needed by NLL loss function
        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
        generated_sequence_t = generated_sequence.permute(0, 2, 1)
        loss = self.loss_fn(generated_sequence_t, data, batch.probs)  # (B, L)
        # print(type(generated_sequence_t))
        # loss = self.loss_fn(generated_sequence_t, batch.data)  # (B, L)
        # sum -w_i*log(x_i.dot(y_i)) = -log(product(w_i*x_i.dot(y_i)))
        loss = loss.sum()

        loss.backward()
        self.optimizer.step()

        return {self.loss_key: loss.item()}

    def generate(
        self, prior_samples: torch.Tensor, random_seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generates samples.

        Args:
            prior_samples (torch.Tensor): samples from a prior distribution with shape (batch_size, sample_dim) and dtype:float32.
            random_seed (Optional[int], optional): random seed for reproducibility. Defaults to None.

        Returns:
            torch.Tensor: generated samples with shape (batch_size, sequence_length) and dtype:float32.
        """
        return self._generate(prior_samples, random_seed)

    def _generate(
        self, prior_samples: torch.Tensor, random_seed: Optional[int] = None
    ) -> torch.Tensor:
        seq, _ = self._generate_w_probs(prior_samples, random_seed)
        return seq

    def _generate_w_probs(
        self, prior_samples: torch.Tensor, random_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.
        """
        n_samples = prior_samples.shape[0]

        prior_samples = prior_samples.to(self._device)
        x0 = self._make_x0(n_samples)  # (batch_size, 1)
        outputs = torch.zeros(
            (n_samples, self.seq_len), device=self._device, dtype=torch.long
        )

        with torch.no_grad():
            for index in range(0, self.seq_len):
                # inputs -> (batch_size, index - 1)
                inputs = outputs[:, :index]

                # inputs -> (batch_size, index)
                inputs = torch.concat((x0, inputs), dim=1)

                # log_prob_dist -> (batch_size, l, vocab_size)
                attention_mask = self.make_lookahead_mask(inputs)
                log_prob_distribution = self(
                    inputs, prior_samples, attn_mask=attention_mask
                )

                # output is passed through LogSoftmax so take exponent
                # prob_dist -> (batch_size, vocab_size)
                prob_dist = torch.exp(log_prob_distribution[:, -1, :]).squeeze(1)

                # sampled_token_indices -> (batch_size, 1)
                sampled_token_indices = torch.multinomial(prob_dist, 1)
                outputs[:, index] = sampled_token_indices.squeeze(1).long()

        # dummy tensor stands in for the conditional probabilities
        return outputs, torch.empty(0)

    def make_lookahead_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a look-ahead mask to be used in the self-attention layers, where future tokens are masked out.
        Positions with ``True`` values are masked out and are not allowed to be attended to.
        Positions with ``False`` values are allowed to be attended to.

        Args:
            x (torch.Tensor): input sequence of token indices with shape (batch_size, seq_len) and dtype:int64

        Outputs:
            torch.Tensor: a mask of shape (seq_len, seq_len) to use in self-attention layers.

        Example:
        >>> _make_attn_mask(3)
        >>> tensor([[False, True,  True],
                    [False, False,  True],
                    [False, False, False]])
        """

        # for compatibility with DDP create a 3D tensor of shape (N * B, S, S)
        # where N is the number of attention heads, B is the batch size and S is the sequence length
        # see https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        batch_size = x.size(0)
        n_attention_heads = self.config.n_attention_heads
        mask = torch.triu(
            torch.ones(
                batch_size * n_attention_heads,
                x.size(1),
                x.size(1),
                device=self._device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        return mask

    def make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a tensor to be used as a mask to prevent model from attending to positions
        with padding tokens as values.

        Args:
            x (torch.Tensor): input sequence of token indices with shape ``(batch_size, seq_len)`` and ``dtype:int64``

        Outputs:
            torch.Tensor: a mask of shape ``(batch_size, seq_len)`` to use in self-attention layers.
        """
        return (x == self.padding_token_index).to(device=self._device)

    @property
    def sample_size(self) -> Tuple[int, ...]:
        return (self.seq_len,)

    @property
    def _models(self) -> List[nn.Module]:
        return [self._model]

    def summary(
        self,
        input_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        batch_dim: Optional[int] = None,
        col_names: Optional[List[str]] = None,
        depth: int = 3,
        col_width: int = 25,
        verbose: int = 1,
    ) -> ModelStatistics:
        """Returns a summary of the model.

        Args:
            input_data (Tuple, optional): dummy input data matching shape and type of actual input data. Defaults to None.
            batch_dim (int, optional): index of batch dimension axis. Defaults to None.
            col_names (List[str], optional): names of columns to display in summary. Defaults to None,
                which will use 'kernel_size' and 'num_params'. To use `input_size` and `output_size` as column names,
                input_size cannot be None.
            depth (int, optional): depth of model to display. Defaults to 3.
            col_width (int, optional): width of columns to display in summary. Defaults to 25.
            verbose (int, optional): verbosity level, options are 0, 1, 2. Defaults to 1.
        """
        if col_names is None:
            col_names = ["num_params"]

        return summary(
            self._model,
            input_data=input_data,
            batch_dim=batch_dim,
            col_names=col_names,
            row_settings=["depth", "var_names"],
            verbose=verbose,
            depth=depth,
            col_width=col_width,
            device=self._device,
        )

    @property
    def config(self) -> PATConfig:
        """Returns the configuration of the model."""
        return self._config

    @classmethod
    def _from_config(
        cls,
        config: PATConfig,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        loss_key: str = "loss",
    ) -> "MolPAT":
        """Instantiates a model from a configuration."""
        hidden_activation_fn = getattr(nn, config.hidden_activation_fn_name)
        return cls(
            vocab_size=config.get_field("vocab_size"),
            seq_len=config.get_field("seq_len"),
            prior_dim=config.prior_dim,
            start_token_index=config.get_field("start_token_index"),
            embedding_dim=config.get_field("embedding_dim"),
            hidden_dim=config.hidden_dim,
            dim_feed_forward=config.dim_feed_forward,
            n_attention_heads=config.n_attention_heads,
            n_encoder_layers=config.n_encoder_layers,
            hidden_activation_fn=hidden_activation_fn,
            dropout=config.dropout,
            optimizer_config=optimizer_config,
            padding_token_index=config.get_field("padding_token_index"),
            loss_key=loss_key,
        )
