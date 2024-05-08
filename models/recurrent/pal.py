from dataclasses import dataclass
from abc import ABC
from typing import List, Optional, Tuple
from warnings import warn
from enum import Enum

import torch
from torch import nn
from orquestra.qml.api import (Batch, TorchGenerativeModel, TrainResult,
                               convert_to_torch)
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch.distributions import Categorical

from ..layers import Concatenate, Add
from .config import NoisyLSTMv2Config


@dataclass
class PriorAssistedLSTMConfig(NoisyLSTMv2Config):
    pass


class _Model(nn.Module, ABC):
    """Base model class used for the PriorAssistedLSTM.
    
    General flow is as follows:
    
    Inputs (b, seq_len)                         Prior Samples (b, ?, ?)
            ⬇                                           ⬇               
    Embedding (b, seq_len, d0)                   PreprocessPriorSamples (b, seq_len, ?)
                                    ⬇                     
                            Combine (b, seq_len, ?)
                                    ⬇
                Linear Projection (b, seq_len, lstm_input_dim)                                 
                                    ⬇                       
                 LSTM (b, seq_len, n_directions * hidden_dim)
                                    ⬇                           
                    Classifier Head (b, seq_len, output_dim)
    
    Combine - can be any layer that combines the prior samples with the embeddings. For example adding them or concatenating them.
    PreprocessPriorSamples - can be any layer that pre-processes the prior samples, and ensures that they have the correct shape.    
    """
    def __init__(
        self,
        lstm: nn.LSTM,
        n_embeddings: int,
        embedding_dim: int,
        output_dim: int,
        combination_layer: nn.Module,
        linear_projection: nn.Module,
        output_activation: nn.Module = nn.Identity(),
        padding_token_index: int = 0,
    ) -> None:
        super().__init__()
        
        assert lstm.input_size == embedding_dim, f"Input size of LSTM ({lstm.input_size}) must match embedding dimension ({embedding_dim})"
        
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.n_directions: int = 2 if lstm.bidirectional else 1
        self.n_layers = lstm.num_layers
        self.hidden_size = lstm.hidden_size

        # model layers
        self.embedding = nn.Embedding(n_embeddings, embedding_dim, padding_token_index)
        self.combine = combination_layer
        self.linear_projection = linear_projection
        self.lstm = lstm
        self.output_head = nn.Sequential(
            nn.Linear(self.n_directions * self.hidden_size, output_dim),
            output_activation
        )

    def _preprocess_prior_samples(
        self, 
        prior_samples: torch.Tensor, 
        seq_len: int,
        *args, 
        **kwargs
    ) -> torch.Tensor:
        return prior_samples

    def forward(
        self,
        inputs: torch.Tensor,
        prior_samples: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.

        Args:
            inputs (torch.Tensor): input sequence of integers, where each integer corresponds to a token in a corpus. Shape: (b, seq_len).
            prior_samples (torch.Tensor): samples from a prior distribution. Shape: (b, ?, ?), where <b> is the batch size. Prior samples may either be,
                a 3D tensor or a 2D tensor. The final dimension may or may not be known ahead of time.
            hidden_state(Optional[Tuple[torch.Tensor, torch.Tensor]], optional): initial hidden state. Defaults to None.
                Expected shape (for each element of tuple): (b, n_dirs * n_layers, h), where <b> is the batch size and <h> is the hidden size of the LSTM,
                and <n_dirs> is 1 if the LSTM is unidirectional, and 2 if it is bidirectional.
                This is different to the default shape of the hidden state returned by the LSTM, which is (n_dirs * n_layers, b, h). The transposed
                variation is compatible with DataParallel, which splits along dimension 0. Hidden state will be transposed before being
                passed to the model (swap dimensions 0 and 1).
                
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: sequence of class logits, and the final hidden state.
        """
        
        # can only concat similar tensors so we first expand to 3D, then repeat to match shape of input
        # prior_samples shape: (b, ?, ?) -> (b, seq_len, ?)
        prior_samples = self._preprocess_prior_samples(prior_samples, inputs.shape[1])

        # (b, seq_len) -> (b, seq_len, d0)
        outputs = self.embedding(inputs)

        # [(b, seq_len, ?), (b, seq_len, d0)] -> (b, seq_len, ??)
        combined = self.combine(outputs, prior_samples)
                        
        # (b, seq_len, ??) -> (b, seq_len, lstm_input_dim)
        outputs = self.linear_projection(combined)

        
        # transpose logic is needed to support nn.DataParallel
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
        
        # outputs: (b, seq_len, n_directions * hidden_dim)
        # hidden_state: (h, c) where h, c have shape (b, n_directions * n_layers, hidden_dim)
        outputs, hidden_state = self.lstm(outputs, hidden_state)

        # transpose hidden state again this time to return expected shape
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

        # (b, seq_len, n_directions * hidden_dim) -> (b, seq_len, output_dim)
        outputs = self.output_head(outputs)

        return outputs, hidden_state


class _AddModel(_Model):
    """Model class used for the PriorAssistedLSTM, that combines prior samples and embeddings by adding them together.
    Expects prior samples to be a 3D tensor of shape (b, seq_len, d0) where d0 is the same as the embedding dimension.
    
    Inputs (b, seq_len)                         Prior Samples (b, seq_len, d0)
            ⬇                                           ⬇               
    Embedding (b, seq_len, d0)                   NoOp (b, seq_len, d0)
                                    ⬇                     
                            Add (b, seq_len, d0)
                                    ⬇
                Linear Projection (b, seq_len, lstm_input_dim)                                 
                                    ⬇                       
                 LSTM (b, seq_len, n_directions * hidden_dim)
                                    ⬇                           
                    Classifier Head (b, seq_len, output_dim)
    """
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
        linear_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            projection_activation_fn
        )
        
        super().__init__(
            lstm=lstm,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            combination_layer=Add(),
            linear_projection=linear_projection,
            output_activation=output_activation,
            padding_token_index=padding_token_index,
        )

  
class _ConcatModel(_Model):
    """Model class used for the PriorAssistedLSTM, that combines prior samples and embeddings by concatenating them together.
    Expects prior samples to be a 3D tensor of shape (b, seq_len, d1) where d1 can be any value.
    
    Inputs (b, seq_len)                         Prior Samples (b, seq_len, d1)
            ⬇                                           ⬇               
    Embedding (b, seq_len, d0)                   NoOp (b, seq_len, d1)
                                    ⬇                     
                    Concatenate (b, seq_len, d0 + d1)
                                    ⬇
                Linear Projection (b, seq_len, lstm_input_dim)                                 
                                    ⬇                       
                 LSTM (b, seq_len, n_directions * hidden_dim)
                                    ⬇                           
                    Classifier Head (b, seq_len, output_dim)
    """
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
        linear_projection = nn.Sequential(
            nn.Linear(prior_sample_dim + embedding_dim, embedding_dim),
            projection_activation_fn
        )
        
        super().__init__(
            lstm=lstm,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            combination_layer=Concatenate(),
            linear_projection=linear_projection,
            output_activation=output_activation,
            padding_token_index=padding_token_index,
        )


class _RepeatSamplesModel(_Model):
    """Base model class used for the PriorAssistedLSTM, where prior samples are expected to be a 2D tensor of shape (b, ?),
    and are repeated to be compatible with the sahpe of the input sequence.
    
    Inputs (b, seq_len)                         Prior Samples (b, ?)
            ⬇                                           ⬇               
    Embedding (b, seq_len, d0)                   Repeat (b, seq_len, ?)
                                    ⬇                     
                        Combine (b, seq_len, ?)
                                    ⬇
                Linear Projection (b, seq_len, lstm_input_dim)                                 
                                    ⬇                       
                 LSTM (b, seq_len, n_directions * hidden_dim)
                                    ⬇                           
                    Classifier Head (b, seq_len, output_dim)
    """
    def _preprocess_prior_samples(self, prior_samples: torch.Tensor, seq_len: int, *args, **kwargs) -> torch.Tensor:
        return prior_samples.unsqueeze(1).repeat(1, seq_len, 1)


class _RepeatSamplesAddModel(_RepeatSamplesModel):
    """Model class used for the PriorAssistedLSTM, that combines prior samples and embeddings by adding them together.
    Expects prior samples to be a 2D tensor of shape (b, d0) where d0 is the same as the embedding dimension.
    
    Inputs (b, seq_len)                         Prior Samples (b, d0)
            ⬇                                           ⬇               
    Embedding (b, seq_len, d0)                   Repeat (b, seq_len, d0)
                                    ⬇                     
                            Add (b, seq_len, d0)
                                    ⬇
                Linear Projection (b, seq_len, lstm_input_dim)                                 
                                    ⬇                       
                 LSTM (b, seq_len, n_directions * hidden_dim)
                                    ⬇                           
                    Classifier Head (b, seq_len, output_dim)
    """
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
        linear_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            projection_activation_fn
        )
        
        super().__init__(
            lstm=lstm,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            combination_layer=Add(),
            linear_projection=linear_projection,
            output_activation=output_activation,
            padding_token_index=padding_token_index,
        )


class _RepeatSamplesConcatModel(_RepeatSamplesModel):
    """Model class used for the PriorAssistedLSTM, that combines prior samples and embeddings by concatenating them together.
    Expects prior samples to be a 2D tensor of shape (b, d1) where d1 can be any value.
    
    Inputs (b, seq_len)                         Prior Samples (b, d1)
            ⬇                                           ⬇               
    Embedding (b, seq_len, d0)                   Repeat (b, seq_len, d1)
                                    ⬇                     
                    Concatenate (b, seq_len, d0 + d1)
                                    ⬇
                Linear Projection (b, seq_len, lstm_input_dim)                                 
                                    ⬇                       
                 LSTM (b, seq_len, n_directions * hidden_dim)
                                    ⬇                           
                    Classifier Head (b, seq_len, output_dim)
    """
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
        linear_projection = nn.Sequential(
            nn.Linear(prior_sample_dim + embedding_dim, embedding_dim),
            projection_activation_fn
        )
        
        super().__init__(
            lstm=lstm,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            combination_layer=Concatenate(),
            linear_projection=linear_projection,
            output_activation=output_activation,
            padding_token_index=padding_token_index,
        )




class CombinationMethod(Enum):
    CONCAT = "concat"
    ADD = "add"


class PriorAsisstedLSTM(TorchGenerativeModel):
    """Implements a Prior-Assisted LSTM (PAL).
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
    """

    _model_options = {
        "concat_True": _RepeatSamplesConcatModel,
        "concat_False": _ConcatModel,
        "add_True": _RepeatSamplesAddModel,
        "add_False": _AddModel,
    }
    
    def _select_model_class(
        self, 
        combination_method: str,
        repeat_samples: bool
    ) -> _Model:
        key = f"{combination_method}_{repeat_samples}"
        options = self._model_options
        if key not in options:
            raise ValueError(f"Invalid pair of combination method '{combination_method}' and repeat samples '{repeat_samples}'")
        
        return options[key]
    
    def _validate_args(
        self, 
        prior_sample_dim: int,
        embedding_dim: int, 
        combination_method: str
    ) -> None:
        if combination_method == "add" and prior_sample_dim != embedding_dim:
            raise ValueError(f"Prior sample dimension ({prior_sample_dim}) must match embedding dimension ({embedding_dim}) when using the 'add' combination method")

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
        dropout: float = 0.0,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        loss_key: str = "loss",
        model_identifier: str = "noisy-lstm-v3",
        do_greedy_sampling: bool = False,
        sampling_temperature: float = 1.0,
        combination_method: CombinationMethod = CombinationMethod.CONCAT,
        repeat_samples: bool = True
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
            combination_method (CombinationMethod): method to combine prior samples with input sequence. Defaults to ``CombinationMethod.CONCAT``.
                Note, that if using the ``CombinationMethod.ADD`` method, the prior samples must have the same dimension as the embeddings.
            repeat_samples (bool, optional): whether to repeat prior samples to expand them to a tensor of shape (b, seq_len, ?). Defaults to True.
                If True prior samples are expected to be 2D tensors of shape (b, ?). If False, prior samples are expected to be 
                3D tensors of shape (b, seq_len, ?).
        """
        super().__init__()
        lstm = nn.LSTM(
            input_size=embedding_dim,
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

        self._sampling_temperature = float(sampling_temperature)

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
        self.combination_method = combination_method
        
        model_class = self._select_model_class(self.combination_method.value, repeat_samples=repeat_samples)
        
        # check that prior sample dimension matches embedding dimension if using the add method
        self._validate_args(prior_sample_dim, embedding_dim, self.combination_method.value)
        
        self._model = model_class(
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
        n_samples = prior_samples.shape[0]
        
        if random_seed:
            random_generator = torch.Generator()
            random_generator.manual_seed(random_seed)

        inputs = self._make_xo(n_samples)  # (batch_size, 1)
        hidden_state: Tuple[
            torch.Tensor, torch.Tensor
        ] = self._make_initial_hidden_state(n_samples)
        outputs = torch.zeros((n_samples, self.seq_len)).to(self._device)
        seq_probabilities = torch.ones((n_samples, self.seq_len, self.vocab_size)).to(self._device)
        
        
        with torch.no_grad():
            for index in range(0, self.seq_len):
                # class_logit_sequence -> (batch_size, 1, vocab_size)
                # hidden_state -> (batch_size, D*n_layers, hidden_size), (batch_size, D*n_layers, hidden_size)
                
                sample = prior_samples[:, index, :].unsqueeze(1) if prior_samples.ndim == 3 else prior_samples
                class_logit_sequence, hidden_state = self(inputs, sample, hidden_state)

                # select element with highest probability
                # or sample from the distribution
                if self._do_greedy_sampling:
                    # sampled_token_indices -> (batch_size, )
                    sampled_token_indices = torch.argmax(
                        class_logit_sequence.squeeze(1), dim=-1
                    )
                    seq_probabilities[:, index, :] = nn.Softmax(-1)(class_logit_sequence.squeeze(1))
                    
                else:
                    cat_distribution = Categorical(
                        logits=class_logit_sequence.squeeze(1)
                        / self.sampling_temperature
                    )

                    # sampled_token_indices -> (batch_size, )
                    sampled_token_indices = cat_distribution.sample()

                    seq_probabilities[:, index, :] = cat_distribution.probs
                    
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
    def config(self) -> PriorAssistedLSTMConfig:
        """Returns model configuration."""
        d = {
            "name": self.model_identifier,
            "vocab_size": self.vocab_size,
            "n_embeddings": self.n_embeddings,
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "bidirectional": self.n_directions > 1,
            "padding_token_index": self.padding_token_index,
        }
        config = PriorAssistedLSTMConfig(**d)
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
        prior_samples = convert_to_torch(prior_samples).to(self._device)
        generated_sequences, probs = self._generate_w_probs(
            prior_samples, random_seed
        )
        return generated_sequences

    def enable_greedy_sampling(self) -> None:
        """Enables greedy sampling."""
        self._do_greedy_sampling = True

    def disable_greedy_sampling(self) -> None:
        """Disables greedy sampling."""
        self._do_greedy_sampling = False


## Add a wrapper to sample multiple times from a given sampler to get a certain shape