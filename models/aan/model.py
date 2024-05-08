from typing import Dict, Optional

import torch
from orquestra.qml.api import Batch, TorchGenerativeModel
from orquestra.qml.models.adversarial.th import AdversarialGenerativeModel
from torch import nn

from .discriminator import LSTMAANDiscriminator
from .generator import NoisyLSTMGenerator


class DrugDiscoveryAAN(AdversarialGenerativeModel):
    def __init__(
        self,
        generator: NoisyLSTMGenerator,
        discriminator: LSTMAANDiscriminator,
        prior: TorchGenerativeModel,
        model_identifier: str = "drug_discovery_aan",
    ) -> None:
        # RBM has 2 * hidden_dim because Generator will split samples into h0 and c0
        # self.prior = RBM(2 * hidden_dim, rbm_hidden_units, rbm_training_parametes)
        self.model_identifier = model_identifier
        super().__init__(generator, discriminator, prior, True, 0.03)
        self.posterior_samples = torch.zeros(0)

    def update_posterior_samples(self, h0: torch.Tensor, c0: torch.Tensor):
        # recall that h0 and c0 have shapes (n_layers, batch_size, hidden_size)
        # we want the state vectors of the last layers
        h0 = h0[0]
        c0 = c0[0]
        new_samples = torch.cat([h0, c0], dim=1)
        self.posterior_samples = torch.cat([self.posterior_samples, new_samples])

    def reset_posterior_samples(self):
        self.posterior_samples = torch.zeros(0)

    def generate(
        self, n_samples: int, random_seed: Optional[int] = None
    ) -> torch.Tensor:
        rbm_samples = self.prior.generate(n_samples)
        encoded_compounds = self.generator.generate(rbm_samples)
        return encoded_compounds

    def train_on_batch(self, batch: Batch[torch.Tensor]) -> Dict:
        data = batch.data
        batch_size = batch.batch_size

        real_labels, fake_labels = self.generate_target_labels(
            batch_size, True, flip_probability=0.03
        )

        real_compounds_label_logits, (h0, c0) = self.discriminator.__call__(data.long())

        disc_real_loss = self.train_discriminator(
            real_labels, real_compounds_label_logits
        )
        self.update_posterior_samples(h0, c0)

        generated_compounds = self.generate(batch_size)
        generated_compounds_label_logits, (h0, c0) = self.discriminator.__call__(
            generated_compounds.detach().long()
        )
        disc_fake_loss = self.train_discriminator(
            fake_labels, generated_compounds_label_logits
        )
        self.update_posterior_samples(h0, c0)

        disc_loss = 0.5 * (disc_real_loss + disc_fake_loss)

        generated_compounds_label_logits, (h0, c0) = self.discriminator.__call__(
            generated_compounds.long()
        )
        generator_loss = self.train_generator(
            real_labels, generated_compounds_label_logits
        )

        return {"GeneratorLoss": generator_loss, "DiscriminatorLoss": disc_loss}

    def train_generator(
        self, target_labels: torch.Tensor, predicted_label_logits: torch.Tensor
    ) -> float:
        # we can't train generator because of discrete outputs
        self.generator.optimizer.zero_grad()

        loss = self.generator.loss_fn(target_labels, predicted_label_logits, None)

        loss.backward()
        self.generator.optimizer.step()

        return loss.item()

    def train_discriminator(
        self, target_labels: torch.Tensor, predicted_label_logits: torch.Tensor
    ) -> float:
        self.discriminator.optimizer.zero_grad()

        loss = self.discriminator.loss_fn(target_labels, predicted_label_logits, None)

        loss.backward()
        self.discriminator.optimizer.step()

        return loss.item()

    def train_prior(self):
        # fetch posterior samples, deduplicate and get probabilities
        posterior_samples = nn.Sigmoid()(self.posterior_samples)
        n_samples = posterior_samples.shape[0]
        posterior_samples, sample_counts = torch.unique(
            posterior_samples, dim=0, return_counts=True
        )
        sample_probabilities = sample_counts / n_samples

        # clear existing posterior samples
        self.reset_posterior_samples()
        batch = Batch(data=posterior_samples, probs=sample_probabilities)
        loss = self.prior.train_on_batch(batch)

        return {"PriorLoss": loss["loss"].item()}
