
from orquestra.qml.api import ContinuousDistanceMeasure, DiscreteDistanceMeasure
from overrides import overrides
import torch
import numpy as np

class ExactNLLTorch(DiscreteDistanceMeasure):
    @overrides
    def measure_distance(self, true_probs: np.ndarray, pred_probs: np.ndarray) -> float:
        # Convert numpy arrays to PyTorch tensors and move them to the appropriate device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        epsilon = 1e-10
        true_probs_torch = torch.from_numpy(true_probs).float().to(device)
        pred_probs_torch = torch.from_numpy(pred_probs).float().to(device)

        # Clip probabilities to avoid log(0)
        pred_probs_torch = torch.clamp(pred_probs_torch, min=epsilon)

        # Compute negative log likelihood
        nll = -torch.sum(true_probs_torch * torch.log(pred_probs_torch))

        # Move the result back to CPU and convert to python float for further usage
        return nll.cpu().item()