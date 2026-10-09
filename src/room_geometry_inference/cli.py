"""Command-line interface.

Commands
--------
synthesize
    Write synthetic measurements (two-channel WAV files and ``dataset.csv``) of rooms of known size.
index
    Write ``dataset.csv`` for a ``<folder>/<label>/<file>.wav`` layout; an
    existing one is replaced only with ``--force``.
prepare
    Identify the impulse response of every measurement and compute its features.
evaluate
    Cross-validate the room classifier, grouped by room.
train
    Train the room classifier on every measurement and save it.
predict
    Classify new measurements with a saved model.
simulate
    Simulate one room, identify its response and estimate the room from the echoes.
invert
    Estimate room size and source distance of measurements by echo inversion.
sensitivity
    Sweep timing offsets, source bearing and position, and noise around one scene.
denoise
    Cross-validate impulse-response denoisers on measurements of known geometry.

Commands that use settings take ``--config`` (a YAML file) and any number of
``--set section.key=value`` overrides; ``--seed`` and ``--dataset`` are
shortcuts for ``--set seed=...`` and ``--set data.dataset=...``. Results go to
``--out``, or to a new ``<command>_<UTC time>`` folder under ``output.root``,
together with ``config.yaml`` and ``manifest.json`` (``index`` writes only
``dataset.csv``). ``predict`` uses only the settings stored in the model file.

Expected failures (a missing file, an invalid configuration, a missing
optional dependency) end with a one-line message and exit code 2; the
traceback is logged at DEBUG level.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from . import __version__
from .config import Config
from .dataset import (
    index_audio_folder,
    items_table,
    load_dataset_index,
    load_features,
    save_features,
    save_svmlight,
)
from .denoising import compare_denoisers, denoising_summary
from .errors import ConfigError
from .evaluation import cross_validate
from .figures import (
    confusion_matrix_table,
    denoising_scores_table,
    impulse_response_table,
    learning_curve_table,
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
from .logging_utils import configure_logging
from .manifest import build_manifest, run_directory, write_json, write_manifest
from .models import load_model, save_model
from .pipeline import (
    denoising_pairs,
    invert_measurements,
    predict_inputs,
    prepare_features,
    train_model,
)
from .recording import load_excitation, write_recording
from .room_estimation import simulate_scene
from .sensitivity import run_sweeps, summarize_sweeps
from .synthetic import write_synthetic_dataset

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Local variables of a failing frame can hold whole datasets; never dump them.
    pretty_exceptions_show_locals=False,
    help="Room geometry from a single microphone: simulation, identification, echo inversion.",
)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML configuration file.")
]
SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override one setting: section.key=value (repeatable)."),
]
SeedOption = Annotated[
    int | None, typer.Option("--seed", help="Master seed; overrides the configuration.")
]
DatasetOption = Annotated[
    Path | None,
    typer.Option("--dataset", "-d", help="Dataset index CSV; overrides data.dataset."),
]
OutOption = Annotated[
    Path | None,
    typer.Option(
        "--out", "-o", help="Output folder (default: a new run folder under output.root)."
    ),
]
ModelOption = Annotated[Path, typer.Option("--model", "-m", help="Model file written by 'train'.")]


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"room-geometry-inference {__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    log_level: Annotated[
        str, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")
    ] = "INFO",
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    try:
        configure_logging(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from None


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=2)


@contextmanager
def _reported_errors() -> Iterator[None]:
    """Turn the library's expected failures into a one-line message and exit code 2."""
    try:
        yield
    except (ValueError, FileNotFoundError, ImportError) as exc:
        _LOGGER.debug("command failed", exc_info=True)
        raise _fail(str(exc)) from None


def _coerce(raw: str) -> Any:
    """Parse a ``--set`` value: true/false, none/null, any JSON literal, else the plain string."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _load_config(
    config_file: Path | None,
    overrides: list[str] | None,
    shortcuts: Mapping[str, Any] | None = None,
) -> Config:
    """Load ``--config``, apply every ``--set``, then every shortcut that is not ``None``."""
    config = Config.from_yaml(config_file) if config_file is not None else Config()
    changes: dict[str, Any] = {}
    for item in overrides or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects section.key=value, got {item!r}")
        changes[key.strip()] = _coerce(raw.strip())
    for key, value in (shortcuts or {}).items():
        if value is not None:
            changes[key] = value.as_posix() if isinstance(value, Path) else value
    return config.override(changes) if changes else config


def _report(label: str, value: object) -> None:
    """Print one line of the closing summary, the label padded to 11 characters."""
    typer.echo(f"{label:<11}: {value}")


def _run_folder(config: Config, command: str, out: Path | None) -> Path:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        return out
    return run_directory(config.output.root, command)


def _finish(
    folder: Path,
    command: str,
    config: Config,
    inputs: Sequence[Path],
    details: Mapping[str, Any],
) -> None:
    """Write ``config.yaml`` and ``manifest.json`` and print the closing line."""
    config.to_yaml(folder / "config.yaml")
    manifest = build_manifest(
        command=command, config=config.to_dict(), inputs=inputs, seed=config.seed, extra=details
    )
    write_manifest(folder / "manifest.json", manifest)
    typer.echo(f"written to {folder.as_posix()}")


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")


def _mean_std(summary: Mapping[str, Mapping[str, float]], metric: str) -> str:
    return f"{summary[metric]['mean']:.3f} +/- {summary[metric]['std']:.3f}"


def _counts(values: Sequence[str]) -> str:
    counts = Counter(values)
    return ", ".join(f"{name} {counts[name]}" for name in sorted(counts))


@app.command("synthesize")
def synthesize_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Folder for the synthetic dataset.")
    ] = Path("data/synthetic"),
) -> None:
    """Write synthetic measurements (two-channel WAV and dataset.csv) of rooms of known size."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        index = write_synthetic_dataset(
            out, config.synthetic, rng=config.seed, acoustics=config.acoustics
        )
        table = pd.read_csv(index, dtype=str)
        _report("items", len(table))
        _report("rooms", table["group"].nunique())
        _report("classes", _counts(table["label"].tolist()))
        _finish(out, "synthesize", config, [], {"n_items": len(table)})


@app.command("index")
def index_command(
    folder: Annotated[
        Path, typer.Argument(help="Folder with one sub-folder of measurement WAV files per class.")
    ],
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing dataset.csv.")
    ] = False,
) -> None:
    """Write dataset.csv for a <folder>/<label>/<file>.wav layout."""
    with _reported_errors():
        index = index_audio_folder(folder, overwrite=force)
        table = pd.read_csv(index, dtype=str)
        _report("items", len(table))
        _report("classes", _counts(table["label"].tolist()))
        typer.echo(f"written to {index.as_posix()}")


@app.command("prepare")
def prepare_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
    svmlight: Annotated[
        bool,
        typer.Option("--svmlight/--no-svmlight", help="Also export features.svm (LIBSVM format)."),
    ] = False,
) -> None:
    """Identify the impulse response of every measurement and compute its features."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"data.dataset": dataset})
        features = prepare_features(config.data.dataset, config)
        folder = _run_folder(config, "prepare", out)
        save_features(folder / "features.npz", features)
        _write_csv(items_table(features), folder / "items.csv")
        if svmlight:
            save_svmlight(folder / "features.svm", features)
        _report("items", features.n_items)
        _report("classes", _counts(features.y.tolist()))
        _report("features", f"{features.X.shape[1]} ({config.features.representation})")
        _finish(
            folder,
            "prepare",
            config,
            [Path(config.data.dataset)],
            {"n_items": features.n_items, "n_features": int(features.X.shape[1])},
        )


@app.command("evaluate")
def evaluate_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
) -> None:
    """Cross-validate the room classifier; write metrics, predictions and figures."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        result = cross_validate(data, config)
        folder = _run_folder(config, "evaluate", out)
        write_json(folder / "metrics.json", result.metrics)
        _write_csv(result.folds, folder / "folds.csv")
        _write_csv(result.predictions, folder / "predictions.csv")
        fmt, dpi = config.output.figure_format, config.output.dpi
        save_figure(plot_confusion_matrix(result, dpi=dpi), folder / "confusion_matrix", fmt)
        _write_csv(confusion_matrix_table(result), folder / "confusion_matrix.csv")
        summary = result.metrics["summary"]
        if "roc_auc" in summary:
            positive = config.evaluation.positive_class
            figure = plot_roc_curve(result.predictions, positive, dpi=dpi)
            save_figure(figure, folder / "roc_curve", fmt)
            _write_csv(roc_curve_table(result.predictions, positive), folder / "roc_curve.csv")
        _report("model", config.model.name)
        _report("items", result.metrics["n_items"])
        _report("folds", result.metrics["n_splits"])
        _report("UAR", _mean_std(summary, "uar"))
        _report("accuracy", _mean_std(summary, "accuracy"))
        if "roc_auc" in summary:
            _report("ROC-AUC", _mean_std(summary, "roc_auc"))
        _finish(folder, "evaluate", config, [features], {"n_items": result.metrics["n_items"]})


@app.command("train")
def train_command(
    features: Annotated[Path, typer.Argument(help="features.npz written by 'prepare'.")],
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
) -> None:
    """Train the room classifier on every measurement and save it."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        data = load_features(features, config)
        estimator = train_model(data, config)
        folder = _run_folder(config, "train", out)
        destination = save_model(folder / "model.joblib", estimator, config, data.classes)
        _report("model", config.model.name)
        _report("items", data.n_items)
        _report("classes", ", ".join(data.classes))
        _report("file", destination.name)
        _finish(folder, "train", config, [features], {"n_items": data.n_items})


@app.command("predict")
def predict_command(
    inputs: Annotated[list[Path], typer.Argument(help="Measurement WAV files to classify.")],
    model: ModelOption,
    out: OutOption = None,
) -> None:
    """Classify new measurements with a saved model."""
    with _reported_errors():
        bundle = load_model(model)
        config = Config.from_dict(bundle["config"])
        table = predict_inputs(inputs, bundle)
        folder = _run_folder(config, "predict", out)
        _write_csv(table, folder / "predictions.csv")
        _report("items", len(table))
        _report("predicted", _counts(table["predicted"].tolist()))
        _finish(folder, "predict", config, [model, *inputs], {"n_items": len(table)})


@app.command("simulate")
def simulate_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
    excitation: Annotated[
        Path | None,
        typer.Option(
            "--excitation", help="Mono WAV file to play instead of the synthetic excitation."
        ),
    ] = None,
) -> None:
    """Simulate the configured room, identify its response and estimate the room from the echoes."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        signal = (
            None
            if excitation is None
            else load_excitation(excitation, config.acoustics.sample_rate)
        )
        scene = simulate_scene(config, rng=config.seed, excitation=signal)
        folder = _run_folder(config, "simulate", out)
        rate = config.acoustics.sample_rate
        truth = scene.truth
        images = pd.DataFrame(
            {
                "image": truth.labels(),
                "order": truth.orders,
                "distance_m": truth.distances_m,
                "delay_ms": 1000.0 * truth.delays_s,
                "tap": truth.taps,
                "amplitude": truth.amplitudes,
            }
        ).sort_values(["tap", "image"], kind="stable")
        _write_csv(images, folder / "images.csv")
        write_recording(
            folder / "recording.wav",
            scene.recording.excitation,
            scene.recording.microphone,
            rate,
        )
        fmt, dpi = config.output.figure_format, config.output.dpi
        figure = plot_impulse_response(
            scene.weights, rate, truth=truth.rir, echo_taps=scene.inversion.echo_taps, dpi=dpi
        )
        save_figure(figure, folder / "impulse_response", fmt)
        _write_csv(
            impulse_response_table(
                scene.weights, rate, truth=truth.rir, echo_taps=scene.inversion.echo_taps
            ),
            folder / "impulse_response.csv",
        )
        curve = scene.learning_curve()
        if curve.size:
            save_figure(plot_learning_curve(curve, rate, dpi=dpi), folder / "learning_curve", fmt)
            _write_csv(learning_curve_table(curve, rate), folder / "learning_curve.csv")
        _write_csv(scene.inversion.candidates, folder / "candidates.csv")
        estimate = scene.inversion.estimate
        write_json(
            folder / "estimate.json",
            {
                "estimate": estimate.as_dict(),
                "true_size_m": list(config.room.size_m),
                "room_error_m": scene.room_error_m,
                "misalignment_db": scene.misalignment_db,
                "echo_taps": scene.inversion.echo_taps,
                "tap_shift": scene.tap_shift,
            },
        )
        width, depth = config.room.size_m
        _report("room", f"{width:g} x {depth:g} m")
        _report("estimate", f"{estimate.width_m:.3f} x {estimate.depth_m:.3f} m")
        _report("size error", f"{scene.room_error_m:.3f} m")
        _report("distance", f"{estimate.source_distance_m:.3f} m")
        _report("misalign.", f"{scene.misalignment_db:.1f} dB")
        _report("echoes", scene.inversion.echo_taps.size)
        _finish(
            folder,
            "simulate",
            config,
            [] if excitation is None else [excitation],
            {"room_error_m": scene.room_error_m, "misalignment_db": scene.misalignment_db},
        )


@app.command("invert")
def invert_command(
    inputs: Annotated[
        list[Path] | None,
        typer.Argument(help="Measurement WAV files (default: every file of data.dataset)."),
    ] = None,
    config_file: ConfigOption = None,
    set_: SetOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
) -> None:
    """Estimate room size and source distance of measurements by echo inversion."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"data.dataset": dataset})
        if inputs:
            if dataset is not None:
                raise _fail("give either measurement files or --dataset, not both")
            table = invert_measurements(inputs, config)
            sources = list(inputs)
        else:
            index = load_dataset_index(config.data.dataset, require_label=False)
            table = invert_measurements(index["path"].tolist(), config, index=index)
            sources = [Path(config.data.dataset)]
        folder = _run_folder(config, "invert", out)
        _write_csv(table, folder / "estimates.csv")
        done = table[table["status"] == "ok"]
        _report("items", len(table))
        _report("inverted", len(done))
        details: dict[str, Any] = {"n_items": len(table), "n_inverted": len(done)}
        if "room_error_m" in table.columns:
            fmt, dpi = config.output.figure_format, config.output.dpi
            save_figure(plot_room_estimates(table, dpi=dpi), folder / "room_estimates", fmt)
            _write_csv(room_estimates_table(table), folder / "room_estimates.csv")
            errors = done["room_error_m"].to_numpy(dtype=np.float64)
            distances = done["distance_error_m"].to_numpy(dtype=np.float64)
            if errors.size:
                _report(
                    "size error", f"median {np.median(errors):.3f} m, mean {errors.mean():.3f} m"
                )
                _report("dist. error", f"median {np.median(distances):.3f} m")
                details["median_room_error_m"] = float(np.median(errors))
        _finish(folder, "invert", config, sources, details)


@app.command("sensitivity")
def sensitivity_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    out: OutOption = None,
) -> None:
    """Sweep timing offsets, source bearing and position, and noise around the room scene."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed})
        table = run_sweeps(config, rng=config.seed)
        summary = summarize_sweeps(table)
        folder = _run_folder(config, "sensitivity", out)
        _write_csv(table, folder / "sensitivity.csv")
        _write_csv(summary, folder / "sensitivity_summary.csv")
        fmt, dpi = config.output.figure_format, config.output.dpi
        for sweep in config.sensitivity.sweeps:
            stem = folder / f"sensitivity_{sweep}"
            save_figure(plot_sensitivity(summary, sweep, dpi=dpi), stem, fmt)
            _write_csv(sensitivity_table(summary, sweep), stem.with_name(f"{stem.name}.csv"))
        _report("sweeps", ", ".join(config.sensitivity.sweeps))
        _report("estimates", len(table))
        _report("failed", int(np.sum(table["status"] != "ok")))
        _finish(folder, "sensitivity", config, [], {"n_rows": len(table)})


@app.command("denoise")
def denoise_command(
    config_file: ConfigOption = None,
    set_: SetOption = None,
    seed: SeedOption = None,
    dataset: DatasetOption = None,
    out: OutOption = None,
) -> None:
    """Cross-validate impulse-response denoisers on measurements of known geometry."""
    with _reported_errors():
        config = _load_config(config_file, set_, {"seed": seed, "data.dataset": dataset})
        noisy, clean, groups = denoising_pairs(config.data.dataset, config)
        settings = config.denoising
        scores = compare_denoisers(
            noisy,
            clean,
            groups,
            methods=settings.methods,
            architectures=settings.architectures,
            n_splits=config.evaluation.n_splits,
            random_state=config.seed,
            options=settings.options(),
        )
        summary = denoising_summary(scores)
        folder = _run_folder(config, "denoise", out)
        _write_csv(denoising_scores_table(scores), folder / "denoising_scores.csv")
        _write_csv(summary, folder / "denoising_summary.csv")
        fmt, dpi = config.output.figure_format, config.output.dpi
        save_figure(plot_denoising_scores(scores, dpi=dpi), folder / "denoising_scores", fmt)
        _report("items", noisy.shape[0])
        _report("methods", len(summary))
        best = summary.loc[summary["mean_db"].idxmin()]
        line = f"{best['method']}: {best['mean_db']:.2f} +/- {best['std_db']:.2f} dB"
        # gain_db is nan when identity was not among the compared methods.
        if np.isfinite(best["gain_db"]):
            line += f" (gain {best['gain_db']:.2f} dB)"
        _report("best", line)
        _finish(
            folder, "denoise", config, [Path(config.data.dataset)], {"n_items": int(noisy.shape[0])}
        )


def main() -> None:
    """Entry point of the ``room-geometry-inference`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
