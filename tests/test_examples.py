"""Every example script runs end to end on synthetic data (with --quick)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_has_a_test():
    tested = {
        "end_to_end_synthetic",
        "invert_simulated_room",
        "sweep_inversion_sensitivity",
        "compare_rir_denoisers",
        "compare_feature_representations",
    }
    assert {path.stem for path in EXAMPLES.glob("*.py")} == tested


def test_end_to_end_synthetic(tmp_path):
    out = tmp_path / "end_to_end_synthetic"
    assert load("end_to_end_synthetic").main(["--quick", "--out", str(out)]) == 0
    for name in (
        "data/dataset.csv",
        "features.npz",
        "items.csv",
        "metrics.json",
        "folds.csv",
        "predictions.csv",
        "model.joblib",
        "new_predictions.csv",
    ):
        assert (out / name).is_file(), name
    predictions = pd.read_csv(out / "new_predictions.csv")
    assert predictions["predicted"].tolist() == ["large", "small"]


def test_invert_simulated_room(tmp_path):
    out = tmp_path / "invert_simulated_room"
    assert load("invert_simulated_room").main(["--quick", "--out", str(out)]) == 0
    estimates = pd.read_csv(out / "estimates.csv")
    assert estimates["room"].tolist() == ["large", "small"]
    assert (estimates["room_error_m"] < 0.05).all()
    for name in ("large_impulse_response.png", "small_candidates.csv", "bearing_family.csv"):
        assert (out / name).is_file(), name
    ranges = json.loads((out / "bearing_family_range.json").read_text(encoding="utf-8"))
    low, high = ranges["width_range_m"]
    assert low < 10.7 < high


def test_sweep_inversion_sensitivity(tmp_path):
    out = tmp_path / "sweep_inversion_sensitivity"
    assert load("sweep_inversion_sensitivity").main(["--quick", "--out", str(out)]) == 0
    summary = pd.read_csv(out / "sensitivity_summary.csv")
    assert set(summary["sweep"]) == {"delay_offset", "source_bearing", "source_shift", "snr"}
    for sweep in ("delay_offset", "snr"):
        assert (out / f"sensitivity_{sweep}.png").is_file()
        assert (out / f"sensitivity_{sweep}.csv").is_file()


def test_compare_rir_denoisers(tmp_path):
    out = tmp_path / "compare_rir_denoisers"
    assert load("compare_rir_denoisers").main(["--quick", "--out", str(out)]) == 0
    summary = pd.read_csv(out / "denoising_summary.csv")
    assert summary["method"].tolist()[:3] == ["identity", "linear", "pca"]
    assert (out / "denoising_scores.png").is_file() and (out / "denoising_scores.csv").is_file()


def test_compare_feature_representations(tmp_path):
    out = tmp_path / "compare_feature_representations"
    assert load("compare_feature_representations").main(["--quick", "--out", str(out)]) == 0
    table = pd.read_csv(out / "representation_comparison.csv")
    assert set(table["representation"]) == {"magnitude", "decay_curve", "echo_delays", "imf_energy"}
    assert (out / "representation_comparison.png").is_file()
