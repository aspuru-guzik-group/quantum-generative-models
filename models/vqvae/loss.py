import torch
from torch import nn


class QuantizationLoss(nn.Module):
    def __init__(self, commitment_term: float) -> None:
        super().__init__()
        self.commitment_term = commitment_term
        self.mse = nn.MSELoss()

    def forward(
        self, input_vectors: torch.Tensor, quantized_vectors: torch.Tensor
    ) -> torch.Tensor:
        encoding_latent_loss = self.mse(quantized_vectors.detach(), input_vectors)
        quantized_latent_loss = self.mse(quantized_vectors, input_vectors.detach())
        loss = quantized_latent_loss + self.commitment_term * encoding_latent_loss
        return loss
