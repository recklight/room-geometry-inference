"""A deep denoising autoencoder for impulse responses (needs the ``dl`` extra).

The network maps a noisy identified response (peak-normalized magnitude) to
the clean one. This is the denoising autoencoder of Vincent et al. (2008) in
the deep form Lu et al. (2013) used for speech enhancement: a stack of fully
connected layers with a nonlinear activation and a linear output layer,
trained on the mean squared error with Adam (Kingma & Ba, 2015). Inputs and
targets are standardized per tap with the training means and standard
deviations. Those statistics are buffers of the network, so they travel with
its weights.

PyTorch is imported inside the functions that need it, so this module can be
imported without it. The estimator follows the scikit-learn regressor
interface (``fit`` / ``predict``) and can run on ``cpu`` or ``cuda``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .errors import require_module

_LOGGER = logging.getLogger(__name__)

#: Hidden-layer activations accepted by the autoencoder and ``denoising.activation``.
ACTIVATIONS: tuple[str, ...] = ("logistic", "relu", "tanh")
#: Torch devices accepted by ``denoising.device``.
DEVICES: tuple[str, ...] = ("cpu", "cuda")

#: Rows per forward pass at prediction time.
_PREDICT_BATCH = 1024
#: Epochs between two progress messages in the log.
_LOG_EVERY = 50


def _build_network(
    torch: Any, n_inputs: int, n_outputs: int, hidden_units: Sequence[int], activation: str
) -> Any:
    """Fully connected layers between the standardization buffers, as an ``nn.Module``.

    The class lives in here because subclassing ``nn.Module`` needs torch, and
    importing this module must not.
    """
    nn = torch.nn
    layers: list[Any] = []
    previous = n_inputs
    nonlinearity = {"logistic": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    for width in hidden_units:
        layers += [nn.Linear(previous, int(width)), nonlinearity()]
        previous = int(width)
    layers.append(nn.Linear(previous, n_outputs))

    class DenoisingNetwork(nn.Module):
        """Standardize, fully connected layers, destandardize."""

        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(*layers)
            for name, size in (
                ("input_mean", n_inputs),
                ("input_scale", n_inputs),
                ("target_mean", n_outputs),
                ("target_scale", n_outputs),
            ):
                self.register_buffer(name, torch.zeros(size))

        def forward(self, inputs: Any) -> Any:
            """Standardized outputs (the training target)."""
            return self.layers((inputs - self.input_mean) / self.input_scale)

        def denoise(self, inputs: Any) -> Any:
            """Outputs on the scale of the targets."""
            return self.forward(inputs) * self.target_scale + self.target_mean

    return DenoisingNetwork()


def _scale(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-column mean and standard deviation (1 where a column is constant)."""
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.0)


class DeepDenoisingAutoencoder(RegressorMixin, BaseEstimator):
    """Fully connected denoising autoencoder with the scikit-learn interface.

    Parameters
    ----------
    hidden_units : sequence of int, optional
        Widths of the hidden layers.
    activation : {"logistic", "relu", "tanh"}, optional
        Hidden-layer nonlinearity.
    n_epochs : int, optional
        Passes over the training data.
    learning_rate : float, optional
        Adam step size.
    batch_size : int, optional
        Mini-batch size (the last batch of an epoch may be smaller).
    weight_decay : float, optional
        L2 penalty of Adam.
    device : {"cpu", "cuda"}, optional
        Where to train and predict.
    random_state : int, optional
        Seed of the initialization and the batch order.

    References
    ----------
    .. [1] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol,
       "Extracting and composing robust features with denoising autoencoders,"
       in Proc. 25th International Conference on Machine Learning (ICML),
       2008, pp. 1096-1103.
    .. [2] X. Lu, Y. Tsao, S. Matsuda, and C. Hori, "Speech enhancement based
       on deep denoising autoencoder," in Proc. Interspeech 2013, pp. 436-440.
    .. [3] D. P. Kingma and J. Ba, "Adam: A method for stochastic
       optimization," in Proc. 3rd International Conference on Learning
       Representations (ICLR), 2015.
    """

    def __init__(
        self,
        hidden_units: Sequence[int] = (256, 128, 256),
        activation: str = "logistic",
        n_epochs: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        self.hidden_units = hidden_units
        self.activation = activation
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state

    def _torch(self) -> Any:
        torch = require_module("torch", extra="dl", feature="the deep denoising autoencoder")
        if self.device not in DEVICES:
            raise ValueError(f"device={self.device!r} is not supported; choose one of {DEVICES}")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but PyTorch sees no CUDA device")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation={self.activation!r} is not supported; choose one of {ACTIVATIONS}"
            )
        return torch

    def fit(self, X: ArrayLike, y: ArrayLike) -> DeepDenoisingAutoencoder:
        """Train on noisy inputs ``X`` and clean targets ``y``."""
        torch = self._torch()
        inputs = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"X and y must be two-dimensional with equal rows, got {inputs.shape} and "
                f"{targets.shape}"
            )
        torch.manual_seed(int(self.random_state))
        network = _build_network(
            torch, inputs.shape[1], targets.shape[1], self.hidden_units, self.activation
        )
        for name, values in zip(
            ("input_mean", "input_scale", "target_mean", "target_scale"),
            (*_scale(inputs), *_scale(targets)),
            strict=True,
        ):
            getattr(network, name).copy_(torch.as_tensor(values, dtype=torch.float32))
        network = network.to(self.device)
        features = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        standardized = torch.as_tensor(
            (targets - targets.mean(axis=0)) / _scale(targets)[1],
            dtype=torch.float32,
            device=self.device,
        )
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        generator = torch.Generator().manual_seed(int(self.random_state))
        loss_function = torch.nn.MSELoss()
        n_rows = features.shape[0]
        network.train()
        for epoch in range(int(self.n_epochs)):
            order = torch.randperm(n_rows, generator=generator).to(self.device)
            total = 0.0
            for start in range(0, n_rows, int(self.batch_size)):
                batch = order[start : start + int(self.batch_size)]
                optimizer.zero_grad()
                loss = loss_function(network(features[batch]), standardized[batch])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * int(batch.numel())
            if (epoch + 1) % _LOG_EVERY == 0:
                _LOGGER.debug("epoch %d: training loss %.4g", epoch + 1, total / n_rows)
        network.eval()
        self.network_ = network
        self.n_features_in_ = inputs.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Denoised responses, shape ``(n, n_outputs)``."""
        check_is_fitted(self, "network_")
        torch = self._torch()
        inputs = np.asarray(X, dtype=np.float64)
        outputs = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], _PREDICT_BATCH):
                chunk = torch.as_tensor(
                    inputs[start : start + _PREDICT_BATCH], dtype=torch.float32, device=self.device
                )
                outputs.append(self.network_.denoise(chunk).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float64)
