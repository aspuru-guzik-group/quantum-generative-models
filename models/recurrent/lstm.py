from typing import Optional, Tuple, List, Dict

import torch
from orquestra.qml.api import Batch, TorchGenerativeModel, TrainResult
from orquestra.qml.core import get_logger
from orquestra.qml.losses.th import weighted_nll_loss
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn


_DEBUG = 10
LOGGER = get_logger("Molecular LSTM - Drug Discovery", level=_DEBUG)


class _MolLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        lstm: nn.LSTM,
        output_activation: nn.Module = nn.LogSoftmax(-1),
        padding_token_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = lstm.input_size
        self.latent_dim = lstm.hidden_size
        self.n_layers = lstm.num_layers
        self.dropout = lstm.dropout
        self._D = 1  # always have uni-directional LSTM
        
        self.padding_token_index = padding_token_index

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_token_index,
        )

        self.lstm = lstm
        self.latent_linear = nn.Linear(self._D * self.latent_dim, self.vocab_size)
        self.output_activation = output_activation

    def _transpose_state_for_data_parallel(self, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transpose the hidden state and cell state tensors to be compatible with ``torch.nn.DataParallel``.
        
        ``DataParallel`` will split input tensors along the first dimension (expected to be the batch dimension), however,
        LSTM state is a tuple of tensors, both having shapes (n_layers, batch_size, hidden_size). Therefore, if a split is performed,
        along the 0th dimension, this will result in shape mismatch errors. To avoid this, we transpose the hidden state and cell state
        tensors to have shape (batch_size, n_layers, hidden_size) before splitting.

        Args:
            state (Tuple[torch.Tensor, torch.Tensor]): lstm state, a tuple of tensors, both having shapes (n_layers, batch_size, hidden_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: lstm state, compatible with ``torch.nn.DataParallel``. Both tensors have shape (batch_size, n_layers, hidden_size).
        """
        h,c = state
        h = h.transpose_(0, 1).contiguous()
        c = c.transpose_(0, 1).contiguous()
        return h, c
    
    def _transpose_state(self, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transpose the hidden state and cell state tensors to be compatible with ``torch.nn.LSTM`` or ``torch.nn.DataParallel``,
        depending on the starting shape of the state tensors.
            
        ``torch.nn.LSTM`` expects the hidden state and cell state tensors to have shape (n_layers, batch_size, hidden_size).
        However, the input tensors to ``torch.nn.DataParallel`` are expected to have shape (batch_size, n_layers, hidden_size),
        and are returned in this format. Therefore, to ensure that outputs of ``torch.nn.DataParallel`` are compatible with ``torch.nn.LSTM``,
        and vice versa, we transpose the hidden state and cell state tensors.
        
        If ``state`` initially has shape (n_layers, batch_size, hidden_size), then we transpose to (batch_size, n_layers, hidden_size).
        If ``state`` initially has shape (batch_size, n_layers, hidden_size), then we transpose to (n_layers, batch_size, hidden_size).
        """
        h, c = state
        h = h.transpose_(0, 1).contiguous()
        c = c.transpose_(0, 1).contiguous()
        return h, c
    
    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            inputs (torch.Tensor): tensor of shape (batch_size, seq_length) and dtype torch.long
            initial_state (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): tuple of tensors giving the last hidden state and cell state of the LSTM.
                Each of these tensors have shape (n_lstm_layers, batch_size, lstm_hidden_size). Defaults to None.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: tuple of tensors,
                the first being the output of the LSTM of shape (batch_size, seq_length, vocab_size) and the second being
                the last hidden state and cell state of the LSTM. Each of these tensors have shape (batch_size, n_lstm_layers, lstm_hidden_size).
        """
        # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        inputs = inputs.long()
        embedded_inputs = self.embedding(inputs)

        
        if initial_state is not None:
            # to make outputs of torch.nn.DataParallel compatible with torch.nn.LSTM
            # (batch_size, n_layers, hidden_size), (n_layers, batch_size, hidden_size)
            initial_state = self._transpose_state(initial_state)
            
        # (batch_size, sequence_length, embedding_dim) -> (batch_size, sequence_length, latent_dim)
        output_sequence, state = self.lstm(embedded_inputs, initial_state)

        # to make outputs of torch.nn.LSTM compatible with torch.nn.DataParallel
        # (n_layers, batch_size, hidden_size), (batch_size, n_layers, hidden_size)
        state = self._transpose_state(state)
        
        # (batch_size, sequence_length, latent_dim) -> (batch_size, sequence_length, vocab_size)
        logits = self.latent_linear(output_sequence)

        return self.output_activation(logits), state

    @property
    def hidden_size(self) -> int:
        return self.latent_dim


class MolLSTM(TorchGenerativeModel):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        sos_token_index: int,
        padding_token_index: Optional[int] = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        loss_key: str = "loss",
        model_identifier: str = "mol_lstm",
    ) -> None:
        super().__init__()

        if dropout == 0.0:
            LOGGER.warning("Default dropout value has been changed to 0.0. Please specify dropout if this is not desired.")
        
        lstm = nn.LSTM(
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

        self._model: _MolLSTM = _MolLSTM(
            vocab_size, 
            lstm, 
            padding_token_index=padding_token_index
        )

        self.optimizer = optimizer_config.optimizer(self._model.parameters())
        self.loss_fn = weighted_nll_loss
        self.loss_key = loss_key
        self.model_identifier = model_identifier

    def __call__(
        self,
        inputs: torch.Tensor,
        initial_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # outputs[0] -> (batch_size, sequence_length, vocab_size)
        return self._model(inputs, initial_hidden_state)

    def _make_xo(self, n_samples: int) -> torch.Tensor:
        return torch.full((n_samples, 1), self.sos_token_index).to(self._device)

    def _train_on_batch(self, batch: Batch) -> TrainResult:
        data: torch.Tensor = batch.data
        probs = batch.probs

        self.set_train_state()
        self.optimizer.zero_grad()

        if len(data.size()) != 2:
            raise ValueError(
                f"Expected 2D tensor as input, but got {len(data.size())}D tensor."
            )

        batch_size, seq_len = data.shape

        # first element will be the <START> token
        # append to that everything but last element of sequence
        x0 = self._make_xo(batch_size)
        model_inputs = torch.concat((x0, data[:, : seq_len - 1]), dim=1)

        # output -> (batch_size, sequence_length, vocab_size)
        generated_sequence, *_ = self(model_inputs, None)

        # permute to fit shape needed by NLL loss function
        # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)
        generated_sequence_t = generated_sequence.permute(0, 2, 1)
        loss = self.loss_fn(generated_sequence_t, data.to(self._device).long(), probs)  # (B, L)

        loss.backward()
        self.optimizer.step()

        return {self.loss_key: loss.item()}

    def _generate_w_probs(
        self, n_samples: int, random_seed: Optional[int] = None
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

        # (batch_size, 1)
        inputs = self._make_xo(n_samples)  
        
        # (n_layers, batch_size, lstm_hidden_size)
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # (batch_size, seq_len)
        outputs = torch.zeros((n_samples, self.seq_len))
        
        # (batch_size, 1)
        seq_probabilities = torch.ones((n_samples, 1))  # TODO: update probs

        with torch.no_grad():
            for index in range(0, self.seq_len):
                # (batch_size, 1) -> (batch_size, 1, vocab_size)
                log_prob_dist, hidden_state = self(inputs, hidden_state)

                # output is passed through LogSoftmax so take exponent
                # (batch_size, 1, vocab_size) -> (batch_size, vocab_size)
                prob_dist = torch.exp(log_prob_dist).squeeze(1)

                # (batch_size, vocab_size) -> (batch_size, 1)
                sampled_token_indices = torch.multinomial(prob_dist, 1)
                
                # (batch_size, 1) -> (batch_size, )
                outputs[:, index] = sampled_token_indices.squeeze(1)

                # (batch_size, 1)
                inputs = sampled_token_indices

        return outputs, seq_probabilities

    @property
    def _models(self) -> List[nn.Module]:
        return [self._model]

    @property
    def sample_size(self) -> Tuple[int, ...]:
        return tuple(self.generate(1).shape[1:])

    @property
    def input_size(self) -> Tuple[int, ...]:
        return self._input_size

    @property
    def config(self) -> Dict:
        """Returns model configuration."""
        d = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self._model.embedding_dim,
            "latent_dim": self._model.latent_dim,
            "n_layers": self._model.n_layers,
            "dropout": self._model.dropout,
            "bidirectional": True if self._model._D == 2 else False,
            "padding_token_index": self.padding_token_index,
        }

        return d

    def _generate(
        self, n_samples: int, random_seed: Optional[int] = None
    ) -> torch.Tensor:
        generated_sequences, probs = self._generate_w_probs(n_samples, random_seed)
        return generated_sequences
