from typing import Optional, Tuple

import torch
from orquestra.qml.api import Batch, TrainResult
from orquestra.qml.models.adversarial.th import Generator
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig

from ..recurrent import NoisyMLELSTM, _NoisyLSTM


class NoisyLSTMGenerator(Generator):
    def __init__(
        self,
        pretrained_lstm: NoisyMLELSTM,
        optimizer_config: TorchOptimizerConfig = AdamConfig(),
        loss_key: str = "NoisyLSTMGenerator",
    ) -> None:
        model: _NoisyLSTM = pretrained_lstm._model
        input_size = pretrained_lstm.input_size
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        optimizer = optimizer_config.optimizer(model.parameters())

        super().__init__(
            model, (input_size,), optimizer, loss_fn=loss_fn, loss_key=loss_key
        )
        self.model_interface = pretrained_lstm

    def train_on_batch(self, batch: Batch[torch.Tensor]) -> TrainResult:
        # we compute the generator loss but we cannot train the generator
        predicted_labels = batch.data
        target_labels = batch.targets
        probs = batch.probs

        loss = self.loss_fn(target_labels, predicted_labels, probs)
        return {self.loss_key: loss.item()}

    def generate(
        self, data: torch.Tensor, random_seed: Optional[int] = None
    ) -> torch.Tensor:
        hidden_dim = self.model_interface._model.hidden_size
        n_layers = self.model_interface._model.n_layers
        n_samples, data_dim = data.shape

        assert (
            n_layers == 1
        ), "Using LSTMs with more than a single layer is not currently supported."

        # we are going to split c0 and h0 from data
        assert data_dim == 2 * hidden_dim

        # recall these need to be made into shape (n_layers, batch_size, hidden_size)
        data = data.expand((n_layers, n_samples, 2 * hidden_dim))

        return self.model_interface.generate(n_samples, data)
