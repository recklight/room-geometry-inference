"""Command line, end to end on synthetic data."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from room_geometry_inference.cli import app
from room_geometry_inference.logging_utils import _OWN_HANDLER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUICK = PROJECT_ROOT / "configs" / "quick_demo.yaml"
COMMANDS = (
    "synthesize",
    "index",
    "prepare",
    "evaluate",
    "train",
    "predict",
    "simulate",
    "invert",
    "sensitivity",
    "denoise",
)
MANIFEST_KEYS = [
    "command",
    "created_utc",
    "package_version",
    "git_commit",
    "seed",
    "versions",
    "inputs",
    "config",
    "details",
]

RUNNER = CliRunner()


def run(*args):
    return RUNNER.invoke(app, [str(a) for a in args])


def _detach_cli_handlers():
    """The CLI logs to the runner's temporary stderr; detach that handler afterwards."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _OWN_HANDLER, False):
            root.removeHandler(handler)


@pytest.fixture(autouse=True)
def _restore_root_logger():
    level = logging.getLogger().level
    yield
    _detach_cli_handlers()
    logging.getLogger().setLevel(level)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Synthetic data and features shared by the CLI tests."""
    root = tmp_path_factory.mktemp("cli")
    data = root / "data"
    assert run("synthesize", "--config", QUICK, "--out", data).exit_code == 0
    prepare = root / "prepare"
    result = run("prepare", "--config", QUICK, "--dataset", data / "dataset.csv", "--out", prepare)
    assert result.exit_code == 0, result.output
    _detach_cli_handlers()
    return root


def test_the_whole_workflow_writes_every_documented_file(prepared):
    data, root = prepared / "data", prepared
    assert {"dataset.csv", "config.yaml", "manifest.json"} <= {p.name for p in data.iterdir()}
    for name in ("features.npz", "items.csv", "config.yaml", "manifest.json"):
        assert (root / "prepare" / name).is_file(), name
    features = root / "prepare" / "features.npz"

    evaluated = run("evaluate", features, "--config", QUICK, "--out", root / "evaluate")
    assert evaluated.exit_code == 0, evaluated.output
    assert "UAR        : " in evaluated.output and "ROC-AUC    : " in evaluated.output
    for name in (
        "metrics.json",
        "folds.csv",
        "predictions.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "roc_curve.csv",
        "roc_curve.png",
        "config.yaml",
        "manifest.json",
    ):
        assert (root / "evaluate" / name).is_file(), name
    manifest = json.loads((root / "evaluate" / "manifest.json").read_text(encoding="utf-8"))
    assert list(manifest) == MANIFEST_KEYS and manifest["command"] == "evaluate"
    columns = pd.read_csv(root / "evaluate" / "predictions.csv").columns.tolist()
    assert columns == ["path", "group", "fold", "label", "predicted", "score_large", "score_small"]

    trained = run("train", features, "--config", QUICK, "--out", root / "train")
    assert trained.exit_code == 0, trained.output
    model = root / "train" / "model.joblib"
    assert model.is_file()

    inputs = [data / "small" / "small_000.wav", data / "large" / "large_001.wav"]
    predicted = run("predict", *inputs, "--model", model, "--out", root / "predict")
    assert predicted.exit_code == 0, predicted.output
    table = pd.read_csv(root / "predict" / "predictions.csv")
    assert table.columns.tolist() == ["path", "predicted", "score_large", "score_small"]
    assert table["path"].tolist() == [path.as_posix() for path in inputs]
    assert table["predicted"].tolist() == ["small", "large"]
    assert predicted.output.strip().splitlines()[-1].startswith("written to ")


def test_the_research_commands_write_their_files(prepared):
    root, data = prepared, prepared / "data" / "dataset.csv"
    simulated = run("simulate", "--config", QUICK, "--out", root / "simulate")
    assert simulated.exit_code == 0, simulated.output
    assert "size error : " in simulated.output
    for name in (
        "images.csv",
        "recording.wav",
        "impulse_response.csv",
        "impulse_response.png",
        "learning_curve.csv",
        "learning_curve.png",
        "candidates.csv",
        "estimate.json",
    ):
        assert (root / "simulate" / name).is_file(), name
    estimate = json.loads((root / "simulate" / "estimate.json").read_text(encoding="utf-8"))
    assert estimate["room_error_m"] < 0.05

    inverted = run("invert", "--config", QUICK, "--dataset", data, "--out", root / "invert")
    assert inverted.exit_code == 0, inverted.output
    estimates = pd.read_csv(root / "invert" / "estimates.csv")
    assert len(estimates) == 16 and estimates["room_error_m"].median() < 0.05
    assert (root / "invert" / "room_estimates.png").is_file()

    one = run(
        "invert",
        prepared / "data" / "small" / "small_000.wav",
        "--config",
        QUICK,
        "--out",
        root / "invert_one",
    )
    assert one.exit_code == 0, one.output
    assert "room_error_m" not in pd.read_csv(root / "invert_one" / "estimates.csv").columns

    swept = run(
        "sensitivity",
        "--config",
        QUICK,
        "--set",
        'sensitivity.sweeps=["delay_offset"]',
        "--out",
        root / "sensitivity",
    )
    assert swept.exit_code == 0, swept.output
    for name in (
        "sensitivity.csv",
        "sensitivity_summary.csv",
        "sensitivity_delay_offset.png",
        "sensitivity_delay_offset.csv",
    ):
        assert (root / "sensitivity" / name).is_file(), name

    denoised = run(
        "denoise",
        "--config",
        QUICK,
        "--dataset",
        data,
        "--set",
        'denoising.methods=["identity","linear"]',
        "--out",
        root / "denoise",
    )
    assert denoised.exit_code == 0, denoised.output
    summary = pd.read_csv(root / "denoise" / "denoising_summary.csv")
    assert summary["method"].tolist() == ["identity", "linear"]
    assert (root / "denoise" / "denoising_scores.png").is_file()
    assert " dB (gain " in denoised.output


def test_denoise_summary_leaves_out_the_gain_without_the_identity_baseline(prepared, tmp_path):
    result = run(
        "denoise",
        "--config",
        QUICK,
        "--dataset",
        prepared / "data" / "dataset.csv",
        "--set",
        'denoising.methods=["linear"]',
        "--out",
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    best = [line for line in result.output.splitlines() if line.startswith("best ")]
    assert len(best) == 1 and best[0].endswith(" dB") and "gain" not in best[0]


def test_evaluation_is_byte_for_byte_reproducible(prepared):
    features = prepared / "prepare" / "features.npz"
    for name in ("first", "second"):
        result = run("evaluate", features, "--config", QUICK, "--seed", 5, "--out", prepared / name)
        assert result.exit_code == 0, result.output
    for name in ("predictions.csv", "metrics.json", "folds.csv"):
        assert (prepared / "first" / name).read_bytes() == (prepared / "second" / name).read_bytes()


def test_without_out_a_run_folder_is_created_under_output_root(prepared, tmp_path):
    features = prepared / "prepare" / "features.npz"
    result = run(
        "evaluate", features, "--config", QUICK, "--set", f"output.root={tmp_path.as_posix()}"
    )
    assert result.exit_code == 0, result.output
    folders = [path.name for path in tmp_path.iterdir()]
    assert len(folders) == 1 and folders[0].startswith("evaluate_") and folders[0].endswith("Z")


def test_index_writes_a_dataset_for_a_class_folder_layout(prepared, tmp_path):
    copy = tmp_path / "copy"
    for path in (prepared / "data").glob("*/*.wav"):
        target = copy / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    result = run("index", copy)
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(copy / "dataset.csv")) == 16
    assert sorted(path.name for path in copy.iterdir()) == ["dataset.csv", "large", "small"]
    result = run("index", copy)
    assert result.exit_code == 2 and "already exists" in result.output
    assert run("index", copy, "--force").exit_code == 0


def test_help_version_and_bare_invocation():
    result = run("--help")
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output
    assert run("--version").output.strip() == "room-geometry-inference 0.1.0"
    assert "Usage" in run().output


@pytest.mark.parametrize(
    "args",
    [
        ("evaluate", "{tmp}/absent.npz"),
        ("prepare", "--dataset", "{tmp}/absent.csv"),
        ("prepare", "--set", "echoes.n_echo=9", "--dataset", "{tmp}/absent.csv"),
        ("prepare", "--set", "model.name=forest"),
        ("prepare", "--set", "model.name"),
        ("prepare", "--config", "{tmp}/broken.yaml"),
        ("--log-level", "LOUD", "synthesize", "--out", "{tmp}/unused"),
        ("evaluate", "{features}", "--config", "{quick}", "--set", "features.floor_db=-40"),
        ("predict", "{tmp}/a.wav", "--model", "{tmp}/absent.joblib"),
        ("invert", "{tmp}/a.wav", "--dataset", "{tmp}/dataset.csv"),
        ("simulate", "--excitation", "{tmp}/absent.wav"),
    ],
)
def test_expected_failures_exit_with_code_2(args, prepared, tmp_path):
    (tmp_path / "broken.yaml").write_text("model: [unclosed\n", encoding="utf-8")
    values = {
        "tmp": tmp_path.as_posix(),
        "features": (prepared / "prepare" / "features.npz").as_posix(),
        "quick": QUICK.as_posix(),
    }
    result = run(*(part.format(**values) for part in args))
    assert result.exit_code == 2, result.output
    assert "error:" in result.output or "Usage" in result.output
    assert isinstance(result.exception, SystemExit)


def test_feature_mismatch_names_the_changed_key(prepared):
    features = prepared / "prepare" / "features.npz"
    result = run("train", features, "--config", QUICK, "--set", "features.representation=magnitude")
    assert result.exit_code == 2
    assert "features.representation" in result.output and "run 'prepare' again" in result.output


def test_module_entry_point_lists_every_command():
    completed = subprocess.run(
        [sys.executable, "-m", "room_geometry_inference", "--help"],
        capture_output=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONIOENCODING="utf-8"),
        check=True,
    )
    for command in COMMANDS:
        assert command in completed.stdout
