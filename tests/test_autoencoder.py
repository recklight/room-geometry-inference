"""The PyTorch denoising autoencoder (runs where torch is installed)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from room_geometry_inference import DeepDenoisingAutoencoder, MissingDependencyError
from room_geometry_inference.denoising import build_denoiser, compare_denoisers


@pytest.fixture
def pairs(rng):
    clean = np.zeros((30, 24))
    for row in range(30):
        clean[row, [2, 6 + row % 5, 15 + row % 4]] = [1.0, 0.5, 0.3]
    noisy = clean + 0.05 * rng.standard_normal(clean.shape)
    return noisy, clean


def test_training_reduces_the_error_and_is_reproducible(pairs):
    pytest.importorskip("torch")
    noisy, clean = pairs
    settings = {"hidden_units": (32,), "n_epochs": 150, "batch_size": 10, "learning_rate": 5e-3}
    model = DeepDenoisingAutoencoder(**settings).fit(noisy, clean)
    before = float(np.mean((noisy - clean) ** 2))
    after = float(np.mean((model.predict(noisy) - clean) ** 2))
    assert after < before / 4
    again = DeepDenoisingAutoencoder(**settings).fit(noisy, clean)
    assert np.allclose(model.predict(noisy), again.predict(noisy), atol=1e-6)


def test_standardization_statistics_are_saved_with_the_weights(pairs):
    pytest.importorskip("torch")
    noisy, clean = pairs
    model = DeepDenoisingAutoencoder(hidden_units=(8,), n_epochs=2).fit(noisy, clean)
    state = model.network_.state_dict()
    for name in ("input_mean", "input_scale", "target_mean", "target_scale"):
        assert name in state
    assert np.allclose(state["target_mean"].numpy(), clean.mean(axis=0), atol=1e-6)


def test_the_deep_autoencoder_takes_part_in_the_comparison(pairs):
    pytest.importorskip("torch")
    noisy, clean = pairs
    scores = compare_denoisers(
        noisy,
        clean,
        methods=("identity", "deep_autoencoder"),
        architectures=((16,),),
        n_splits=3,
        options={"n_epochs": 20, "batch_size": 8},
    )
    assert set(scores["method"]) == {"identity", "deep_autoencoder 16"}
    assert scores.shape[0] == 60


@pytest.mark.parametrize(
    ("settings", "fragment"),
    [({"activation": "softplus"}, "activation="), ({"device": "tpu"}, "device=")],
)
def test_invalid_settings_are_rejected(settings, fragment, pairs):
    pytest.importorskip("torch")
    noisy, clean = pairs
    with pytest.raises(ValueError, match=fragment):
        DeepDenoisingAutoencoder(**settings).fit(noisy, clean)


def test_cuda_without_a_device_is_explained(pairs):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is available")
    noisy, clean = pairs
    with pytest.raises(ValueError, match="no CUDA device"):
        DeepDenoisingAutoencoder(device="cuda").fit(noisy, clean)


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="PyTorch is installed")
def test_without_torch_the_error_names_the_extra(pairs):
    noisy, clean = pairs
    with pytest.raises(MissingDependencyError, match=r"\[dl\]"):
        DeepDenoisingAutoencoder().fit(noisy, clean)
    with pytest.raises(MissingDependencyError, match=r"\[dl\]"):
        build_denoiser("deep_autoencoder")
