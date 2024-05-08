from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from orquestra.qml.api import Batch, TorchGenerativeModel, TrainResult
from orquestra.qml.optimizers.th import AdamConfig, TorchOptimizerConfig
from torch import nn


@dataclass
class NoisyLSTMConfig:
    vocab_size: int
    n_embeddings: int
    embedding_dim: int
    latent_dim: int
    n_layers: int
    dropout: float
    bidirectional: bool
    padding_token_index: Optional[int]


@dataclass
class NoisyLSTMv2Config:
    name: str
    vocab_size: int
    projection_dim: int
    n_embeddings: int
    embedding_dim: int
    latent_dim: int
    n_layers: int
    dropout: float
    bidirectional: bool
    padding_token_index: Optional[int]

    def as_dict(self) -> Dict:
        """Returns the configuration as a dictionary."""
        return self.__dict__
