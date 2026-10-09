"""Figures render without a display and come with matching tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from conftest import LARGE_ROOM, simulate
from room_geometry_inference import (
    Config,
    confusion_matrix_table,
    cross_validate,
    denoising_scores_table,
    impulse_response_table,
    learning_curve_table,
    new_figure,
    plot_confusion_matrix,
    plot_denoising_scores,
    plot_impulse_response,
    plot_learning_curve,
    plot_roc_curve,
    plot_room_estimates,
    plot_sensitivity,
    roc_curve_table,
    room_estimates_table,
    save_figure,
    sensitivity_table,
)


@pytest.fixture
def result(make_toy, rng):
    config = Config().override({"model.name": "knn", "evaluation.n_splits": 3})
    return cross_validate(make_toy(rng, separation=1.0), config)


def test_confusion_matrix_figure_and_table(result):
    assert isinstance(plot_confusion_matrix(result, dpi=60), Figure)
    table = confusion_matrix_table(result)
    assert list(table.columns) == ["label", "predicted", "count", "fraction"]
    assert table["count"].sum() == result.metrics["n_items"]
    assert np.allclose(table.groupby("label")["fraction"].sum(), 1.0)


def test_roc_curve_figure_and_table(result):
    assert isinstance(plot_roc_curve(result.predictions, "large", dpi=60), Figure)
    table = roc_curve_table(result.predictions, "large")
    assert table.columns.tolist() == ["false_positive_rate", "true_positive_rate", "threshold"]
    assert table["true_positive_rate"].is_monotonic_increasing


def test_impulse_response_figure_and_table():
    truth = simulate(LARGE_ROOM)
    noisy = truth.rir + 0.01 * np.random.default_rng(0).standard_normal(truth.rir.size)
    figure = plot_impulse_response(noisy, 16_000, truth=truth.rir, echo_taps=[66, 307], dpi=60)
    assert isinstance(figure, Figure)
    table = impulse_response_table(noisy, 16_000, truth=truth.rir, echo_taps=[66, 307])
    assert table.columns.tolist() == ["tap", "time_ms", "identified", "true", "picked"]
    assert table["picked"].sum() == 2 and table["true"].max() == 1.0
    assert impulse_response_table(noisy, 16_000)["true"].isna().all()


def test_learning_curve_figure_and_table():
    curve = np.linspace(0.0, -40.0, 100)
    assert isinstance(plot_learning_curve(curve, 1000, dpi=60), Figure)
    table = learning_curve_table(curve, 1000)
    assert table["time_s"].iloc[-1] == pytest.approx(0.099)


def test_sensitivity_figure_and_table():
    summary = pd.DataFrame(
        {
            "sweep": ["source_shift"] * 4,
            "parameter": ["x", "x", "y", "y"],
            "value": [0.0, 0.5, 0.0, 0.5],
            "room_error_mean_m": [0.0, 0.07, 0.0, 0.07],
            "room_error_std_m": [np.nan] * 4,
        }
    )
    assert isinstance(plot_sensitivity(summary, "source_shift", dpi=60), Figure)
    assert len(sensitivity_table(summary, "source_shift")) == 4
    assert sensitivity_table(summary, "snr").empty


def test_denoising_figure_and_table():
    scores = pd.DataFrame(
        {
            "item": [0, 1, 0, 1],
            "group": ["", "", "", ""],
            "fold": [0, 1, 0, 1],
            "method": ["identity", "identity", "linear", "linear"],
            "mse_db": [-30.0, -31.0, -33.0, -34.0],
        }
    )
    assert isinstance(plot_denoising_scores(scores, dpi=60), Figure)
    assert denoising_scores_table(scores).columns.tolist() == [
        "item",
        "group",
        "fold",
        "method",
        "mse_db",
    ]


def test_room_estimates_figure_and_table():
    estimates = pd.DataFrame(
        {
            "path": ["a", "b", "c"],
            "label": ["large", "small", "small"],
            "status": ["ok", "ok", "no_solution"],
            "width_m": [10.6, 3.0, np.nan],
            "depth_m": [7.6, 3.6, np.nan],
            "true_width_m": [10.7, 3.5, 3.5],
            "true_depth_m": [7.5, 3.0, 3.0],
        }
    )
    assert isinstance(plot_room_estimates(estimates, dpi=60), Figure)
    table = room_estimates_table(estimates)
    assert table["side"].tolist() == ["longer", "shorter", "longer", "shorter"]
    assert table["estimated_m"].tolist() == [10.6, 7.6, 3.6, 3.0]


def test_saving_keeps_dots_in_the_file_name(tmp_path):
    figure = new_figure(2, 2, 50)
    assert figure.get_size_inches().tolist() == [2.0, 2.0]
    for fmt in ("png", "svg", "pdf"):
        path = save_figure(figure, tmp_path / "run.v1" / "scores.final", fmt)
        assert path.name == f"scores.final.{fmt}" and path.stat().st_size > 0
