import typing as t

import torch
from torch import nn

from .codebook import VectorQuantizer
from .decoder import Decoder
from .encoder import Encoder


class VQVAE(nn.Module):
    def __init__(
        self,
        img_channels: int,
        hidden_dim: int = 128,
        n_residual_layers: int = 2,
        residual_layer_hidden_dim: int = 32,
        n_embeddings: int = 512,
        embedding_dim: int = 64,
        hidden_activation_fn: nn.Module = nn.ReLU(),
        ema_decay: float = 0.0,
    ):
        """Initializes the VQ-VAE.

        Args:
            img_channels (int): number of channels in the input images.
            hidden_dim (int, optional): hidden dimension of the model.. Defaults to 128.
            n_residual_layers (int, optional): number of layers in the residual stack. Defaults to 2.
            residual_layer_hidden_dim (int, optional): hidden dimension of each residual layer. Defaults to 32.
            n_embeddings (int, optional): number of 'codes' in the codebook. Defaults to 512.
            embedding_dim (int, optional): dimension of eaach code in the codebook. Defaults to 64.
            hidden_activation_fn (nn.Module, optional): activation function used in hidden layers. Defaults to nn.ReLU().
            ema_decay (float, optional): ema decay. If not 0, then EMA vector quantization layer will be used.
        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(
            img_channels,
            hidden_dim,
            n_residual_layers,
            residual_layer_hidden_dim,
            hidden_activation_fn,
        )
        self.pre_quantizer_conv = nn.Conv2d(
            in_channels=hidden_dim, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        self.quantizer = VectorQuantizer(n_embeddings, embedding_dim)
        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            n_residual_layers,
            residual_layer_hidden_dim,
            hidden_activation_fn,
        )

    @property
    def n_codes(self) -> int:
        return self.quantizer.n_embeddings

    @property
    def code_vector_dim(self) -> int:
        return self.quantizer.embedding_dim

    @property
    def codebook_vectors(self) -> torch.Tensor:
        return self.quantizer.codebook.weight.data.detach()

    def get_code_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(self.codebook_vectors.device)
        return torch.index_select(self.codebook_vectors, dim=0, index=indices)

    def get_random_code_vectors(
        self, n_code_vectors: int, seed: t.Optional[int] = None
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        code_vector_indices = torch.randint(0, self.n_codes, (n_code_vectors,)).int()
        code_vectors = self.get_code_vectors(code_vector_indices)

        return code_vectors

    def forward(self, x):
        z = self.encoder(x)
        z: torch.Tensor = self.pre_quantizer_conv(z)

        z: torch.Tensor = z.permute(0, 2, 3, 1).contiguous()
        quantized, _ = self.quantizer(z)
        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 3, 1, 2)

        return self.decoder(quantized)
