from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import torch
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer

# from orquestra.qml.trainers.th.simple_trainer import train
# from orquestra.qml.trainers.callbacks import Callback
from orquestra.qml.api import (
    Batch,
    Callback,
    GenerativeModel,
    TorchGenerativeModel,
    TrainResult,
)
from orquestra.qml.data_loaders import CardinalityDataLoader, new_data_loader
from orquestra.qml.models import qcbm
from orquestra.qml.models.qcbm import layer_builders
from orquestra.qml.models.rbm.th import RBM
from orquestra.qml.optimizers.th import AdamConfig
from orquestra.qml.trainers.simple_trainer import SimpleTrainer
from torch import nn


class QCBMSamplingFunction_v2:
    def __init__(
        self,
        shape: Sequence[int],
        n_hidden_unit: int,
        map_to: int = 256,
        back_end=QulacsSimulator(),
        optimizer=ScipyOptimizer("Powell", options={"maxiter": 1}),
    ) -> None:
        """In shape -1 is the placeholder that will be replaced with number of samples during when calling SamplingFunction.

        Args:
            shape (Sequence[int]): shape of output tensor. For now must be a Sequence of 3 integers and at least one of three must be -1.
                I.e. (10, -1, 100).
        """
        visible_units = shape[-1]
        self.n_qubits = n_hidden_unit
        self.main = nn.Linear(visible_units, map_to)
        self.shape = shape
        self.trainer = SimpleTrainer()
        self.backend = back_end
        self.optimizer = optimizer
        self.n_output_variable = visible_units
        self.ansatz = qcbm.EntanglingLayerAnsatz(
            n_qubits=self.n_qubits,
            n_layers=4,
            entangling_layer_builder=layer_builders.LineEntanglingLayerBuilder(
                n_qubits=self.n_qubits
            ),
        )

        self.model = qcbm.WavefunctionQCBM(
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            backend=self.backend,
            choices=(0.0, 1.0),
            use_efficient_training=False,
        )

    def config(self) -> dict:
        d = {
            "name": self.__class__.__name__,
            "shape": self.shape,
            "rbm_hidden_units": self.model.n_qubits,
        }
        return d

    def as_string(self) -> str:
        name = self.__class__.__name__
        rbm_s = "QCBM(visible={visible}, hidden={hidden})".format(
            visible=self.n_output_variable, hidden=self.model.n_qubits
        )
        s = "{name}(shape={shape}, qcbm={rbm})".format(
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
            samples[iteration] = torch.from_numpy(self.model.generate(n_samples))

        return self.main(samples)

    def generate(self, n_samples: int):
        return self.__call__(n_samples=n_samples)

    def train(
        self, batch, n_epochs=20, disable_progress_bar=True
    ):  # , device: Optional[torch.device] = None):
        # train(self.rbm,dataloader=batch,n_epochs=n_epochs)
        self.trainer.train(
            self.model,
            data_loader=batch,
            n_epochs=n_epochs,
            disable_progress_bar=disable_progress_bar,
        )

    def save_weights(self, filepath):
        return True
        # self.model.save_weights(filepath=filepath)


class QCBMSamplingFunction_v3:
    def __init__(
        self,
        # shape: Sequence[int],
        n_qubits: int,
        # map_to: int = 256,
        back_end=QulacsSimulator(),
        optimizer=ScipyOptimizer("Powell", options={"maxiter": 1}),
    ) -> None:
        """In shape -1 is the placeholder that will be replaced with number of samples during when calling SamplingFunction.

        Args:
            shape (Sequence[int]): shape of output tensor. For now must be a Sequence of 3 integers and at least one of three must be -1.
                I.e. (10, -1, 100).
        """
        self.n_qubits = n_qubits
        # self.main = nn.Linear(visible_units, map_to)
        # self.shape = shape
        self.trainer = SimpleTrainer()
        self.backend = back_end
        self.optimizer = optimizer
        self.n_output_variable = self.n_qubits
        self.ansatz = qcbm.EntanglingLayerAnsatz(
            n_qubits=self.n_qubits,
            n_layers=4,
            entangling_layer_builder=layer_builders.LineEntanglingLayerBuilder(
                n_qubits=self.n_qubits
            ),
        )

        self.model = qcbm.WavefunctionQCBM(
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            backend=self.backend,
            choices=(0.0, 1.0),
            use_efficient_training=False,
        )

    def config(self) -> dict:
        d = {
            "name": self.__class__.__name__,
            "n_qubits": self.n_qubits,
        }
        return d

    def as_string(self) -> str:
        name = self.__class__.__name__
        qcbm_s = "QCBM(n_output_variable={visible}, n_qubits={hidden})".format(
            visible=self.n_output_variable, hidden=self.model.n_qubits
        )
        s = "{name}(n_qubits={n_qubits}, qcbm={qcbm_s})".format(
            name=name, n_qubits=self.n_qubits, qcbm_s=qcbm_s
        )
        return s

    def generate(self, n_samples: int, unique=False) -> Any:
        if unique:
            samples = torch.from_numpy(self.model.generate(n_samples)).float()
            unique_samples = samples.unique(dim=0)
            while n_samples < unique_samples.shape[0]:
                tmp_samples = torch.from_numpy(
                    self.model.generate(n_samples - unique_samples.shape[0])
                ).float()
                unique_samples = (
                    tmp_samples.unique(dim=0).cat(tmp_samples, 0).unique(dim=0)
                )

        else:
            unique_samples = torch.from_numpy(self.model.generate(n_samples)).float()

        return unique_samples

    # def train(self, batch, n_epochs=20):  # , device: Optional[torch.device] = None):
    #     # train(self.rbm,dataloader=batch,n_epochs=n_epochs)
    #     self.trainer.train(
    #         self.model, data_loader=batch, n_epochs=n_epochs, disable_progress_bar=True
    #     )

    def train_on_batch(self, batch):
        return self.model.train_on_batch(batch)

    def save_weights(self, filepath):
        return True
        # self.model.save_weights(filepath=filepath)

    def __str__(self) -> str:
        return self.as_string()
