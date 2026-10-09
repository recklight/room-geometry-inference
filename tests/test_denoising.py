"""Response denoisers and their cross-validated comparison."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from room_geometry_inference import (
    LinearCalibration,
    MissingDependencyError,
    PCADenoiser,
    build_denoiser,
    compare_denoisers,
    denoising_summary,
    per_sample_mse_db,
)


@pytest.fixture
def pairs(rng):
    """Sparse clean responses in a few positions, plus a noise floor and a gain error."""
    clean = np.zeros((24, 40))
    for row in range(24):
        clean[row, [3, 10 + row % 4, 25 + row % 3]] = [1.0, 0.6, 0.4]
    noisy = 0.8 * clean + 0.05 + 0.02 * rng.standard_normal(clean.shape)
    groups = np.repeat([f"room_{k}" for k in range(6)], 4)
    return noisy, clean, groups


def test_mse_in_db_per_row():
    scores = per_sample_mse_db([[1.0, 0.0], [1.0, 1.0]], [[0.9, 0.1], [1.0, 1.0]])
    assert scores.tolist() == pytest.approx([-20.0, -300.0])
    with pytest.raises(ValueError, match="shapes differ"):
        per_sample_mse_db([[1.0]], [[1.0, 2.0]])


def test_linear_calibration_recovers_gain_and_offset(pairs):
    noisy, clean, _ = pairs
    model = LinearCalibration().fit(noisy, clean)
    assert model.gain_ == pytest.approx(1.25, abs=0.05)
    assert (
        per_sample_mse_db(clean, model.predict(noisy)).mean()
        < per_sample_mse_db(clean, noisy).mean()
    )


def test_pca_keeps_only_what_clean_responses_look_like():
    clean = np.eye(3)[[0, 1, 0, 1]]
    model = PCADenoiser(n_components=1).fit(clean, clean)
    assert model.predict([[0.0, 0.0, 1.0]])[0].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert model.predict([[1.0, 0.0, 0.0]])[0].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_pca_is_reproducible_on_few_long_responses(rng):
    """Wide data (long responses, few rows) would otherwise get a randomized SVD."""
    noisy, clean = rng.random((30, 1600)), rng.random((30, 1600))
    first = PCADenoiser(n_components=10).fit(noisy, clean).predict(noisy)
    second = PCADenoiser(n_components=10).fit(noisy, clean).predict(noisy)
    assert np.array_equal(first, second)


def test_comparison_scores_every_item_once_per_method_without_room_leakage(pairs):
    noisy, clean, groups = pairs
    scores = compare_denoisers(
        noisy,
        clean,
        groups,
        methods=("identity", "linear", "pca", "mlp"),
        architectures=((16,), (8, 8)),
        n_splits=3,
        options={"n_epochs": 30},
    )
    assert scores.columns.tolist() == ["item", "group", "fold", "method", "mse_db"]
    assert scores["method"].unique().tolist() == ["identity", "linear", "pca", "mlp 16", "mlp 8-8"]
    assert (scores.groupby("method")["item"].nunique() == 24).all()
    folds_per_room = scores.groupby("group")["fold"].nunique()
    assert (folds_per_room == 1).all()


def test_the_comparison_is_reproducible(pairs):
    noisy, clean, groups = pairs
    first = compare_denoisers(
        noisy,
        clean,
        groups,
        methods=("mlp",),
        architectures=((8,),),
        n_splits=3,
        options={"n_epochs": 20},
    )
    second = compare_denoisers(
        noisy,
        clean,
        groups,
        methods=("mlp",),
        architectures=((8,),),
        n_splits=3,
        options={"n_epochs": 20},
    )
    pd.testing.assert_frame_equal(first, second)


def test_linear_calibration_beats_the_noisy_input(pairs):
    noisy, clean, groups = pairs
    summary = denoising_summary(
        compare_denoisers(noisy, clean, groups, methods=("identity", "linear"), n_splits=3)
    )
    assert summary["method"].tolist() == ["identity", "linear"]
    assert summary.loc[1, "gain_db"] > 1.0 and summary.loc[0, "gain_db"] == 0.0


def test_summary_uses_the_sample_standard_deviation():
    table = pd.DataFrame({"method": ["a", "a", "a"], "mse_db": [-10.0, -12.0, -14.0]})
    summary = denoising_summary(table)
    assert summary.loc[0, "std_db"] == pytest.approx(2.0)
    assert np.isnan(summary.loc[0, "gain_db"])


def test_invalid_comparisons_are_rejected(pairs):
    noisy, clean, groups = pairs
    with pytest.raises(ValueError, match="unknown denoiser"):
        compare_denoisers(noisy, clean, methods=("wiener",))
    with pytest.raises(ValueError, match="equal 2-D arrays"):
        compare_denoisers(noisy, clean[:5])
    with pytest.raises(ValueError, match="groups"):
        compare_denoisers(noisy, clean, groups, n_splits=10)
    with pytest.raises(ValueError, match="unknown denoiser"):
        build_denoiser("wiener")


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="PyTorch is installed")
def test_without_torch_the_deep_autoencoder_names_the_extra():
    with pytest.raises(MissingDependencyError, match=r"\[dl\]"):
        build_denoiser("deep_autoencoder")
