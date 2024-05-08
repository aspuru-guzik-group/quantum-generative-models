import typing as t

import torch
from torch import nn

from .distance import EuclidianCodebookDistance


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        distance_fn: nn.Module = EuclidianCodebookDistance(),
    ):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings

        self.codebook = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(
            -1 / self.n_embeddings, 1 / self.n_embeddings
        )
        self._distance_fn = distance_fn

    def compute_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Compute a matrix of pair-wise distances between vectors in tensor t1 and tensor t2.

        Returns:
            torch.Tensor: matrix of pairwise distances of shape (n_vec_t1, n_vec_t2).
        """
        return self._distance_fn(t1, t2)

    def forward(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = self.compute_distance(flat_input, self.codebook.weight)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.n_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

        return quantized, encodings


class EMAVectorQuantizer(VectorQuantizer):
    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        decay: float,
        epsilon: float = 1e-5,
        distance_fn: nn.Module = EuclidianCodebookDistance(),
    ):
        super().__init__(n_embeddings, embedding_dim, distance_fn)

        self.codebook.weight.data.normal_()

        self.register_buffer("ema_cluster_size", torch.zeros(n_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(n_embeddings, embedding_dim))
        self.ema_w.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        # expects that inputs have shape (B, H, W, C) and are contiguous in memory
        quantized, encodings = super().forward(inputs)

        # we will need the flat inputs in the EMA update
        flat_input = inputs.view(-1, self.embedding_dim)

        # Use EMA to update the embedding vectors
        # TODO: Move this into trainer
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.ema_cluster_size * self.epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

            self.codebook.weight = nn.Parameter(
                self.ema_w / self.ema_cluster_size.unsqueeze(1)
            )

        return quantized, encodings
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
