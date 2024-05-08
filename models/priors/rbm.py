from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import torch
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer

# from orquestra.qml.trainers.th.simple_trainer import train
# from orquestra.qml.trainers.callbacks import Callback
from orquestra.qml.api import Callback, GenerativeModel, TorchGenerativeModel
from orquestra.qml.data_loaders import CardinalityDataLoader, new_data_loader
from orquestra.qml.models import qcbm
from orquestra.qml.models.qcbm import layer_builders
from orquestra.qml.models.rbm.th import RBM
from orquestra.qml.optimizers.th import AdamConfig
from orquestra.qml.trainers.simple_trainer import SimpleTrainer
from torch import nn


class RBMSamplingFunction_v2:
    def __init__(self, shape: Sequence[int], rbm_hidden_units: int) -> None:
        """In shape -1 is the placeholder that will be replaced with number of samples during when calling SamplingFunction.

        Args:
            shape (Sequence[int]): shape of output tensor. For now must be a Sequence of 3 integers and at least one of three must be -1.
                I.e. (10, -1, 100).
        """
        visible_units = shape[-1]
        self.rbm_hidden_units = rbm_hidden_units
        rbm = RBM(visible_units, rbm_hidden_units, RBMParams(1))
        self.main = nn.Linear(visible_units, 256)
        self.shape = shape
        self.rbm = rbm
        self.trainer = SimpleTrainer()

    def config(self) -> dict:
        d = {
            "name": self.__class__.__name__,
            "shape": self.shape,
            "rbm_hidden_units": self.rbm.n_hidden_units,
        }
        return d

    def as_string(self) -> str:
        name = self.__class__.__name__
        rbm_s = "RBM(visible={visible}, hidden={hidden})".format(
            visible=self.rbm.n_visible_units, hidden=self.rbm.n_hidden_units
        )
        s = "{name}(shape={shape}, rbm={rbm})".format(
            name=name, shape=self.shape, rbm=rbm_s
        )
        return s

    def __call__(self, n_samples: int) -> Any:
        shape = list(self.shape)
        n_iterations = shape[0]

        # RBM can only generate 2D tensors, so to generate 3D tensors
        # we will generate several 2D tensors and put them together.
        shape = shape[1:]

        for dim_idx, dim_size in enumerate(shape):
            if dim_size == -1:
                break

        shape[dim_idx] = n_samples
        samples = torch.zeros((n_iterations, *shape))
        for iteration in range(n_iterations):
            samples[iteration] = self.rbm.generate(n_samples)

        return self.main(samples)

    def generate(self, n_samples: int):
        return self.__call__(n_samples)

    def train(self, batch, n_epochs=20, disable_progress_bar=True):
        # train(self.rbm,dataloader=batch,n_epochs=n_epochs)
        self.trainer.train(
            self.rbm,
            data_loader=batch,
            n_epochs=n_epochs,
            disable_progress_bar=disable_progress_bar,
        )

    def save_weights(self, filepath):
        self.rbm.save_weights(filepath=filepath)
