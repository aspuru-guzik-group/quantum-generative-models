import typing
import typing as t
from typing import Optional, Tuple

import torch
from orquestra.qml.api import TorchGenerativeModel, TrainResult
from orquestra.qml.functional.layers.th import PositionalEncoding
from orquestra.qml.models.transformer._base.th import TransformerEncoderBase
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class _DiscreteTransformerEncoder(TransformerEncoderBase):
    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int = 6,
        n_attn_heads: int = 8,
        dim_feedforward: int = 512,
        hidden_activation_fn: nn.Module = nn.LeakyReLU(),
        padding_token_index: t.Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        """Encoder-only Transformer model for datasets with a fixed number of discrete classes. I.e. text, drug sequences and etc.

        Args:
            n_classes (int): number of discrete classes in the dataset.
            embedding_dim (int): the dimension of the latent embedding space.
            hidden_dim (int): the hidden dimension of the encoder.
            n_layers (int, optional): number of encoder blocks. Defaults to 6.
            n_attn_heads (int, optional): number of attention heads in the MHA block. Defaults to 8.
            dim_feedforward (int, optional): dimension of the feed-forward layers. Defaults to 512.
            hidden_activation_fn (nn.Module, optional): activation function applied to tensors flowing through the model. Defaults to nn.LeakyReLU().
            padding_token_index (int, optional): index of the padding token used to pad sequences to a fixed length.
                Specifying this parameter will result in the padding token having an all-zeros embedding vector such that it does not contribute
                to gradient updates.
            dropout (float, optional): rate of dropout of certain neurons. Defaults to 0.0.
        """
        super().__init__()

        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.dim_feedforward = dim_feedforward
        self.hidden_activation_fn = hidden_activation_fn
        self.padding_token_index = padding_token_index
        self.dropout = dropout

        self.embedding_layer = nn.Embedding(
            n_classes, embedding_dim, padding_idx=padding_token_index
        )
        self.pe_block = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            hidden_activation_fn,
            PositionalEncoding(hidden_dim, dropout),
        )
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            n_attn_heads,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation=hidden_activation_fn,
        )
        self.encoder_block = TransformerEncoder(encoder_layer, n_layers)
        self.latent_linear = nn.Linear(hidden_dim, n_classes)
        self.output_activation = nn.LogSoftmax(-1)

    def forward(
        self,
        inputs: torch.Tensor,
        attn_mask: typing.Optional[torch.Tensor] = None,
        padding_mask: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # inputs -> (batch_size, seq_len)
        # outputs -> (batch_size, seq_len, embedding_dim)
        outputs = self.embedding_layer(inputs)
        outputs = self.pe_block(outputs)

        # outputs -> (batch_size, seq_len, hidden_dim)
        outputs = self.encoder_block(
            outputs, mask=attn_mask, src_key_padding_mask=padding_mask
        )
        outputs = self.latent_linear(outputs)
        return self.output_activation(outputs)

    def get_config(self) -> t.Dict:
        """Returns model configuration."""
        # TODO use self.hidden_activation_fn.__dict__, but that is not immediately json serializable
        hidden_act_dict = {}
        hidden_act_dict["name"] = self.hidden_activation_fn.__class__.__name__

        d = dict(
            n_classes=self.n_classes,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_attn_heads=self.n_attn_heads,
            dim_feedforward=self.dim_feedforward,
            hidden_activation_fn=self.hidden_activation_fn,
            padding_token_index=self.padding_token_index,
            dropout=self.dropout,
            hidden_activation_fn_config=hidden_act_dict,
        )
        d.pop("hidden_activation_fn", None)
        return d


class DiscreteTransformerMLE(TorchGenerativeModel):
    def __init__(
        self,
        n_classes: int,
        seq_len: int,
        hidden_dim: int,
        embedding_dim: int,
        sos_token_index: int,
        padding_token_index: int,
        n_layers: int = 6,
        n_attn_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        hidden_activation_fn: nn.Module = nn.LeakyReLU(),
        use_lookahead_mask: bool = True,
        use_padding_mask: bool = True,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        device: torch.device = torch.device("cpu"),
        loss_key: str = "loss",
        model_identifier: str = "transformer_mle",
    ) -> None:
        """Transformer interface designed for use on datasets with a fixed number of discrete classes, for example
        text sentences, sequences of drug compounds and etc.

        Args:
            n_classes (int): number of unique classes (tokens) in the source dataset.
            seq_len (int): the length of the input sequences, and conversely the length of the sequences that the model
                will generate.
            hidden_dim (int): the hidden dimension of the model.
            embedding_dim (int): the dimension of the latent embedding space.
            sos_token_index (int): the index of the token signaling the start of a sequence.
            padding_token_index (int): index of the padding token used to pad sequences to a fixed length.
                Specifying this parameter will result in the padding token having an all-zeros embedding vector such that it does not contribute
                to gradient updates.
            n_layers (int, optional): number of encoder blocks in the model. Recall that the model is an encoder-only architecture. Defaults to 6.
            n_attn_heads (int, optional): number of attention heads in the MHA component. More attention heads may allow the model
                to learn a more rich representation of the data, however will impact model sizer and time for a single passthrough. Defaults to 8.
            dim_feedforward (int, optional): the dimension of the feed-forward layers of the model. Defaults to 512.
            dropout (float, optional): rate to randomly "drop" neurons in the model. Defaults to 0.
            hidden_activation_fn (nn.Module, optional): activation function applied to tensors flowing through the model. Defaults to nn.LeakyReLU().
            use_lookahead_mask (bool, optional): whether to apply a look-ahead attention mask, to allow the model to function auto-regressively. Defaults to True.
            use_padding_mask (bool, optional): whether to apply a padding mask when computing attention keys. If using padding mask,
                the padding token index must also be set. Defaults to True.
            optimizer_config (TorchOptimizerConfig, optional): configuration for the desired optimizer to use during training. Defaults to AdamConfig().
            device (torch.device): device to load model onto. Defaults to torch.device("cpu").
            loss_key (str, optional): the name of the key that will carry model training loss during training.
            model_identifier (str, optional): a unique model identifier which can be useful when saving results.
        """

        transformer_model = _DiscreteTransformerEncoder(
            n_classes,
            embedding_dim,
            hidden_dim,
            n_layers,
            n_attn_heads,
            dim_feedforward,
            hidden_activation_fn,
            padding_token_index,
            dropout,
        )
        self.loss_key = loss_key
        self.seq_len = seq_len
        self.sos_token_index = sos_token_index
        self.padding_token_index = padding_token_index
        self.optimizer = optimizer_config.optimizer(transformer_model.parameters())
        self.loss_fn = nn.NLLLoss(reduction="none")
        self.use_lookahead_mask = use_lookahead_mask
        self.use_padding_mask = use_padding_mask
        self.input_size = (seq_len,)
        self.model_identifier = model_identifier
        self._model: _DiscreteTransformerEncoder = transformer_model
        self._device = device
        self.model.to(device)

    def __call__(
        self,
        data: torch.Tensor,
        attn_mask: typing.Optional[torch.Tensor] = None,
        padding_mask: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Executes a forward pass through the model.

        Args:
            data (torch.Tensor): _description_
            attn_mask (typing.Optional[torch.Tensor], optional): a "look-ahead" mask so that the transformer
                cannot use information in later elements of sequences. Defaults to None.
            padding_mask (typing.Optional[torch.Tensor], optional): a mask to prevent padding tokens from contributing to attention scores
                and influencing loss values / gradient update. Defaults to None.
        """
        return self.model(data, attn_mask, padding_mask)

    def _make_x0(self, n_samples: int) -> torch.Tensor:
        """Returns a tensor to be used as the initial input to the model."""
        return torch.full((n_samples, 1), self.sos_token_index)

    def _generate_w_probs(
        self, n_samples: int, random_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the underlying model and return the raw form along with the
        conditional probabilities of each of the sequences.
        """
        random_generator = None
        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        x0 = self._make_x0(n_samples)  # (batch_size, 1)
        outputs = torch.zeros((n_samples, self.seq_len))
        seq_probabilities = torch.ones((n_samples, 1))  # TODO: update probs

        with torch.no_grad():
            for index in range(0, self.seq_len):
                # inputs -> (batch_size, index - 1)
                inputs = outputs[:, :index].long()

                # inputs -> (batch_size, index)
                inputs = torch.concat((x0, inputs), dim=1)

                # log_prob_dist -> (batch_size, l, vocab_size)
                attention_mask = self.lookahead_mask(inputs.size(1))
                log_prob_distribution = self.__call__(inputs, attn_mask=attention_mask)

                # output is passed through LogSoftmax so take exponent
                # prob_dist -> (batch_size, vocab_size)
                prob_dist = torch.exp(log_prob_distribution[:, -1, :]).squeeze(1)

                # sampled_token_indices -> (batch_size, 1)
                sampled_token_indices = torch.multinomial(prob_dist, 1)
                outputs[:, index] = sampled_token_indices.squeeze(1)

        return outputs, seq_probabilities

    def get_token_embeddings(self, token_indices: t.Sequence[int]) -> torch.Tensor:
        input_tensor = torch.tensor(token_indices).view((-1, 1))
        with torch.no_grad():
            embeddings = self._model.embedding_layer(input_tensor)
        return embeddings

    @property
    def device(self) -> torch.device:
        """Returns the device that the underlying model is currently loaded onto.

        Returns:
            torch.device: instance of torch.device.
        """
        return torch.device(self._device)

    @property
    def model(self) -> _DiscreteTransformerEncoder:
        """Returns the underlying model used by the interface.

        Returns:
            nn.Module: torch Module.
        """
        return self._model

    @property
    def config(self) -> t.Dict:
        """Returns model configuration."""
        model_config = self.model.get_config()
        model_config.update(dict(use_padding_mask=self.use_padding_mask))
        return model_config

    @staticmethod
    def lookahead_mask(size: int) -> torch.Tensor:
        """Returns a square look-ahead attention mask that masks out future elements in a sequence.
        Args:
            size (int): The size of the mask.
        Returns:
            torch.Tensor: Additive attention mask.
        """

        # 0s are made to be -inf, which results in a value of zero for after performing softmax attention
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).float()
        mask = mask.masked_fill(
            mask == 0, float("-inf")
        )  # upper-triangle zeros are converted to -inf
        mask = mask.masked_fill(
            mask == 1, float(0.0)
        )  # lower triangular ones are converted to 0s
        return mask

    def padding_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns a tensor to be used as a mask to prevent model from attending to positions
        with padding tokens as values.
        """
        return inputs == self.padding_token_index

    @property
    def sample_size(self) -> typing.Tuple[int, ...]:
        return (self.seq_len,)

    def _train_step(self, data: torch.Tensor, probs: torch.Tensor) -> TrainResult:
        """Trains the model on a single batch of data and returns the associated loss."""

        self.set_train_state()
        self.optimizer.zero_grad()

        if len(data.size()) != 2:
            raise ValueError(
                f"Expected 2D tensor as input, but got {len(data.size())}D tensor."
            )

        batch_size, seq_len = data.shape

        # first element will be the <START> token
        # append to that everything but last element of sequence
        x0 = self._make_x0(batch_size)
        model_inputs = torch.concat((x0, data[:, : seq_len - 1]), dim=1).long()

        padding_mask = (
            self.padding_mask(model_inputs) if self.use_padding_mask else None
        )
        attention_mask = (
            self.lookahead_mask(seq_len) if self.use_lookahead_mask else None
        )

        # using __call__ because otherwise I don't see signature hints in VSCODE
        # generated_sequence -> (batch_size, sequence_length, vocab_size)
        generated_sequence = self.__call__(model_inputs, attention_mask, padding_mask)

        # permute to fit shape needed by NLL loss function
        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
        generated_sequence_t = generated_sequence.permute(0, 2, 1)
        loss = self.loss_fn(generated_sequence_t, data.long())  # (B, L)

        # sum -w_i*log(x_i.dot(y_i)) = -log(product(w_i*x_i.dot(y_i)))
        loss = loss.sum(dim=1)
        loss = torch.dot(loss, probs)

        loss.backward()
        self.optimizer.step()

        return {self.loss_key: loss.detach()}

    def generate(
        self, n_samples: int, random_seed: typing.Optional[int] = None
    ) -> torch.Tensor:
        """Generate samples from the underlying model.
        Currently the initial input to the model will be sampled randomly from the set {0, 1}.

        Args:
            n_samples (int): number of samples to generate.
            random_seed (Optional[int], optional): random seed for reproducibility. Defaults to None.

        Returns:
            torch.Tensor: generated bitstrings
        """

        seq, _ = self._generate_w_probs(n_samples, random_seed)
        return seq

    def save_weights(self, filepath: str):
        """Saves model weights under specified filepath."""
        if not (filepath.endswith(".pt") or filepath.endswith(".pth")):
            raise ValueError("Invalid file extension. Must be `.pt` or `.pth`")

        state_dict = self.model.state_dict()
        torch.save(state_dict, filepath)

    def load_weights(self, filepath: str):
        """Loads model weights from specified filepath."""
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)
