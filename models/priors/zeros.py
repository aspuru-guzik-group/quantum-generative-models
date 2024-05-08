from typing import Optional, Tuple, List
import torch


from orquestra.qml.api import TorchGenerativeModel, TrainResult, Batch


class ZerosSampler(TorchGenerativeModel):
    """A sampler that always returns a vector of zeros.
    """
    _repr_fields = ["sample_size", "dtype", "device"]
    
    def __init__(
        self, 
        shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.float32
    ) -> None:
        """Initialize the sampler.

        Args:
            shape (Optional[Tuple[int, ...]], optional): The shape of the output tensor, excluding the batch dimension. Defaults to None.
                Example: if `shape` is `(2, 3)`, then the output tensor will have shape `(n_samples, 2, 3)`.
                Example: if `shape` is `(2,)`, then the output tensor will have shape `(n_samples, 2)`.
            dtype (torch.dtype, optional): The dtype of the output tensor. Defaults to torch.float32.
        """
        self._sample_shape = shape
        self._dtype = dtype
        
    def config(self) -> dict:
        """Returns a dictionary containing the configuration of the sampler.
        """
        d = {
            "name": self.__class__.__name__,
            "sample_size": self.sample_size,
            "dtype": str(self.dtype),
        }
        return d

    def as_string(self) -> str:
        """Returns a string representation of the sampler.
        """
        name = self.__class__.__name__
        s = "{name}(sample_size={sample_size})".format(
            name=name, sample_size=self.sample_size
        )
        return s
    
    def _generate(self, n_samples: int, random_seed: Optional[int] = None) -> torch.Tensor:
        return torch.zeros((n_samples, *self.sample_size), dtype=self.dtype, device=self.device)
    
    def _train_on_batch(self, batch: Batch) -> TrainResult:
        """Does not support training."""
        return {}

    @property
    def dtype(self) -> int:
        return self._dtype
    
    @property
    def _models(self) -> List[torch.nn.Module]:
        return []
    
    @property
    def sample_size(self) -> Tuple[int, ...]:
        return self._sample_shape