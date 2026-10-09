"""Report figures, each paired with a table of exactly what it shows.

Only the object-oriented :class:`matplotlib.figure.Figure` API is used (never
``pyplot``), so figures render without a display. Titles and axis labels are
English, so no CJK font is needed.

* :func:`plot_confusion_matrix`, :func:`confusion_matrix_table`: pooled
  out-of-fold confusion matrix with counts and row shares.
* :func:`plot_roc_curve`, :func:`roc_curve_table`: ROC curve of the positive
  class (two classes).
* :func:`plot_impulse_response`, :func:`impulse_response_table`: true and
  identified responses with the picked echoes.
* :func:`plot_learning_curve`, :func:`learning_curve_table`: squared error of
  the adaptive filter over time.
* :func:`plot_sensitivity`, :func:`sensitivity_table`: size error along one
  sweep, with error bars.
* :func:`plot_denoising_scores`, :func:`denoising_scores_table`: box plot of
  the per-response MSE of every denoiser.
* :func:`plot_room_estimates`, :func:`room_estimates_table`: estimated
  against true side lengths.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.metrics import roc_curve

if TYPE_CHECKING:
    from .evaluation import CrossValidationResult

#: Colors of successive data series.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c9a227")
#: Markers of successive data series, a second encoding beside the color.
SERIES_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v")


def new_figure(width: float = 6.0, height: float = 4.0, dpi: int = 150) -> Figure:
    """A blank figure with constrained layout.

    Examples
    --------
    >>> new_figure(4, 3, 100).get_size_inches().tolist()
    [4.0, 3.0]
    """
    return Figure(figsize=(width, height), dpi=dpi, layout="constrained")


def save_figure(figure: Figure, path_stem: str | Path, figure_format: str = "png") -> Path:
    """Save ``figure`` as ``<path_stem>.<figure_format>`` and return the path.

    The extension is appended, never substituted, so dots in the stem survive.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     save_figure(new_figure(1, 1, 50), Path(folder) / "scores.v2", "svg").name
    'scores.v2.svg'
    """
    stem = Path(path_stem)
    destination = stem.with_name(f"{stem.name}.{figure_format}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format=figure_format)
    return destination


def confusion_matrix_table(result: CrossValidationResult) -> pd.DataFrame:
    """Pooled confusion matrix in long form.

    Returns
    -------
    pandas.DataFrame
        Columns ``label`` (true class), ``predicted``, ``count`` and
        ``fraction`` (share of the true class's items).
    """
    pooled = result.metrics["pooled"]
    classes = list(pooled["classes"])
    counts = np.asarray(pooled["confusion_matrix"], dtype=np.int64)
    totals = counts.sum(axis=1, keepdims=True)
    fractions = counts / np.maximum(totals, 1)
    return pd.DataFrame(
        {
            "label": np.repeat(classes, len(classes)),
            "predicted": np.tile(classes, len(classes)),
            "count": counts.ravel(),
            "fraction": fractions.ravel(),
        }
    )


def plot_confusion_matrix(result: CrossValidationResult, *, dpi: int = 150) -> Figure:
    """Heat map of the pooled confusion matrix with counts and row shares."""
    table = confusion_matrix_table(result)
    classes = result.classes
    size = len(classes)
    fractions = table["fraction"].to_numpy().reshape(size, size)
    counts = table["count"].to_numpy().reshape(size, size)
    figure = new_figure(2.4 + 1.1 * size, 1.2 + 1.0 * size, dpi)
    axes = figure.add_subplot()
    image = axes.imshow(fractions, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(size):
        for column in range(size):
            share = fractions[row, column]
            axes.text(
                column,
                row,
                f"{counts[row, column]}\n{share:.0%}",
                ha="center",
                va="center",
                color="white" if share > 0.5 else "black",
                fontsize=9,
            )
    axes.set_xticks(range(size), classes)
    axes.set_yticks(range(size), classes)
    axes.set_xlabel("predicted class")
    axes.set_ylabel("true class")
    axes.set_title(f"Pooled out-of-fold predictions: UAR {result.metrics['pooled']['uar']:.3f}")
    figure.colorbar(image, ax=axes, label="share of true class")
    return figure


def roc_curve_table(predictions: pd.DataFrame, positive_class: str) -> pd.DataFrame:
    """False- and true-positive rates of the positive class at every threshold.

    Examples
    --------
    >>> rows = pd.DataFrame({"label": ["large", "small", "large", "small"],
    ...                      "score_large": [0.9, 0.2, 0.6, 0.7]})
    >>> roc_curve_table(rows, "large").columns.tolist()
    ['false_positive_rate', 'true_positive_rate', 'threshold']
    """
    truth = predictions["label"].astype(str).to_numpy() == positive_class
    false_positive, true_positive, thresholds = roc_curve(
        truth, predictions[f"score_{positive_class}"].to_numpy()
    )
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive,
            "true_positive_rate": true_positive,
            "threshold": np.minimum(thresholds, 1.0),
        }
    )


def plot_roc_curve(predictions: pd.DataFrame, positive_class: str, *, dpi: int = 150) -> Figure:
    """ROC curve of the pooled out-of-fold scores of the positive class."""
    table = roc_curve_table(predictions, positive_class)
    figure = new_figure(4.2, 4.0, dpi)
    axes = figure.add_subplot()
    axes.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="chance")
    axes.plot(
        table["false_positive_rate"],
        table["true_positive_rate"],
        color=SERIES_COLORS[0],
        marker=SERIES_MARKERS[0],
        markersize=3,
        label=f"{positive_class} vs rest",
    )
    axes.set_xlabel("false positive rate (1 - specificity)")
    axes.set_ylabel("true positive rate (sensitivity)")
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.set_title(f"ROC curve ({positive_class})")
    axes.legend(loc="lower right")
    return figure


def impulse_response_table(
    identified: ArrayLike,
    sample_rate: int,
    *,
    truth: ArrayLike | None = None,
    echo_taps: ArrayLike | None = None,
) -> pd.DataFrame:
    """Identified (and true) response tap by tap, with the picked echoes flagged.

    Returns
    -------
    pandas.DataFrame
        Columns ``tap``, ``time_ms``, ``identified`` (peak-normalized
        magnitude), ``true`` (peak-normalized magnitude, empty without a
        truth) and ``picked`` (whether an echo was picked at that tap).

    Examples
    --------
    >>> table = impulse_response_table([0.0, 2.0, -1.0], 1000, echo_taps=[1.0])
    >>> table[["identified", "picked"]].values.tolist()
    [[0.0, False], [1.0, True], [0.5, False]]
    """
    magnitude = np.abs(np.asarray(identified, dtype=np.float64).ravel())
    peak = float(magnitude.max()) or 1.0
    taps = np.arange(magnitude.size)
    table = pd.DataFrame(
        {"tap": taps, "time_ms": 1000.0 * taps / sample_rate, "identified": magnitude / peak}
    )
    true_values = np.full(magnitude.size, np.nan)
    if truth is not None:
        reference = np.abs(np.asarray(truth, dtype=np.float64).ravel())[: magnitude.size]
        true_values[: reference.size] = reference / (float(reference.max()) or 1.0)
    table["true"] = true_values
    picked = np.zeros(magnitude.size, dtype=bool)
    if echo_taps is not None:
        positions = np.clip(
            np.round(np.asarray(echo_taps, dtype=np.float64)), 0, magnitude.size - 1
        )
        picked[positions.astype(np.int64)] = True
    table["picked"] = picked
    return table


def plot_impulse_response(
    identified: ArrayLike,
    sample_rate: int,
    *,
    truth: ArrayLike | None = None,
    echo_taps: ArrayLike | None = None,
    dpi: int = 150,
) -> Figure:
    """Identified response (and the true one, mirrored below the axis) with picked echoes."""
    table = impulse_response_table(identified, sample_rate, truth=truth, echo_taps=echo_taps)
    figure = new_figure(7.0, 3.6, dpi)
    axes = figure.add_subplot()
    axes.plot(
        table["time_ms"],
        table["identified"],
        color=SERIES_COLORS[0],
        linewidth=0.8,
        label="identified |h|",
    )
    if table["true"].notna().any():
        axes.plot(
            table["time_ms"],
            -table["true"],
            color=SERIES_COLORS[1],
            linewidth=0.8,
            label="true |h| (mirrored)",
        )
    chosen = table[table["picked"]]
    axes.plot(
        chosen["time_ms"],
        chosen["identified"],
        linestyle="none",
        marker=SERIES_MARKERS[2],
        color=SERIES_COLORS[2],
        label="picked echoes",
    )
    axes.axhline(0.0, color="0.5", linewidth=0.5)
    axes.set_xlabel("time after emission (ms)")
    axes.set_ylabel("normalized magnitude")
    axes.set_title("Impulse response")
    axes.legend(loc="upper right", fontsize=8)
    return figure


def learning_curve_table(curve_db: ArrayLike, sample_rate: int) -> pd.DataFrame:
    """Learning curve with time stamps: ``sample``, ``time_s``, ``squared_error_db``.

    Examples
    --------
    >>> learning_curve_table([-1.0, -2.0], 2).values.tolist()
    [[0.0, 0.0, -1.0], [1.0, 0.5, -2.0]]
    """
    values = np.asarray(curve_db, dtype=np.float64).ravel()
    samples = np.arange(values.size)
    return pd.DataFrame(
        {"sample": samples, "time_s": samples / float(sample_rate), "squared_error_db": values}
    )


def plot_learning_curve(curve_db: ArrayLike, sample_rate: int, *, dpi: int = 150) -> Figure:
    """Squared error of the adaptive filter (moving average, dB) over time."""
    table = learning_curve_table(curve_db, sample_rate)
    figure = new_figure(6.0, 3.4, dpi)
    axes = figure.add_subplot()
    axes.plot(table["time_s"], table["squared_error_db"], color=SERIES_COLORS[0], linewidth=0.8)
    axes.set_xlabel("time (s)")
    axes.set_ylabel("squared error (dB)")
    axes.set_title("Learning curve of the adaptive filter")
    axes.grid(alpha=0.3)
    return figure


#: Axis label of the swept quantity of every sweep.
_SWEEP_LABELS = {
    "delay_offset": "arrival offset (samples)",
    "source_bearing": "true source bearing (degrees)",
    "source_shift": "source displacement (m)",
    "snr": "signal-to-noise ratio (dB)",
}


def sensitivity_table(summary: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """The rows of one sweep of :func:`~room_geometry_inference.sensitivity.summarize_sweeps`.

    Examples
    --------
    >>> rows = pd.DataFrame({"sweep": ["snr", "delay_offset"], "value": [1.0, 2.0]})
    >>> sensitivity_table(rows, "snr")["value"].tolist()
    [1.0]
    """
    return summary[summary["sweep"] == sweep].reset_index(drop=True)


def plot_sensitivity(summary: pd.DataFrame, sweep: str, *, dpi: int = 150) -> Figure:
    """Mean size error (error bars: sample standard deviation) along one sweep."""
    table = sensitivity_table(summary, sweep)
    figure = new_figure(5.6, 3.6, dpi)
    axes = figure.add_subplot()
    for position, (parameter, part) in enumerate(table.groupby("parameter", sort=False)):
        axes.errorbar(
            part["value"],
            part["room_error_mean_m"],
            yerr=part["room_error_std_m"].fillna(0.0),
            color=SERIES_COLORS[position % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[position % len(SERIES_MARKERS)],
            capsize=3,
            label=str(parameter),
        )
    axes.set_xlabel(_SWEEP_LABELS.get(sweep, sweep))
    axes.set_ylabel("room size error (m)")
    axes.set_title(f"Sensitivity: {sweep.replace('_', ' ')}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    return figure


def denoising_scores_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-response MSE of every denoiser: ``item``, ``group``, ``fold``, ``method``, ``mse_db``."""
    return scores[["item", "group", "fold", "method", "mse_db"]].reset_index(drop=True)


def plot_denoising_scores(scores: pd.DataFrame, *, dpi: int = 150) -> Figure:
    """Box plot of the per-response MSE of every denoiser."""
    table = denoising_scores_table(scores)
    methods = list(dict.fromkeys(table["method"]))
    figure = new_figure(1.6 + 1.1 * len(methods), 3.8, dpi)
    axes = figure.add_subplot()
    data = [table.loc[table["method"] == name, "mse_db"].to_numpy() for name in methods]
    axes.boxplot(data, showmeans=True)
    axes.set_xticks(range(1, len(methods) + 1), methods, rotation=20, ha="right")
    axes.set_ylabel("MSE per response (dB)")
    axes.set_title("Out-of-fold denoising error")
    axes.grid(axis="y", alpha=0.3)
    return figure


def room_estimates_table(estimates: pd.DataFrame) -> pd.DataFrame:
    """Estimated against true side lengths in long form (successful inversions only).

    Returns
    -------
    pandas.DataFrame
        Columns ``path``, ``label``, ``side`` (``longer`` or ``shorter``),
        ``true_m`` and ``estimated_m``.

    Examples
    --------
    >>> rows = pd.DataFrame({"path": ["a"], "label": ["x"], "status": ["ok"],
    ...     "width_m": [7.6], "depth_m": [10.6], "true_width_m": [10.7], "true_depth_m": [7.5]})
    >>> room_estimates_table(rows)[["side", "true_m", "estimated_m"]].values.tolist()
    [['longer', 10.7, 10.6], ['shorter', 7.5, 7.6]]
    """
    done = estimates[estimates["status"] == "ok"]
    estimated = np.sort(done[["width_m", "depth_m"]].to_numpy(dtype=np.float64), axis=1)[:, ::-1]
    true = np.sort(done[["true_width_m", "true_depth_m"]].to_numpy(dtype=np.float64), axis=1)[
        :, ::-1
    ]
    return pd.DataFrame(
        {
            "path": np.repeat(done["path"].to_numpy(), 2),
            "label": np.repeat(done["label"].to_numpy(), 2),
            "side": np.tile(["longer", "shorter"], len(done)),
            "true_m": true.ravel(),
            "estimated_m": estimated.ravel(),
        }
    )


def plot_room_estimates(estimates: pd.DataFrame, *, dpi: int = 150) -> Figure:
    """Scatter of estimated against true side lengths, with the identity line."""
    table = room_estimates_table(estimates)
    figure = new_figure(4.6, 4.2, dpi)
    axes = figure.add_subplot()
    values = table[["true_m", "estimated_m"]].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    margin = 0.05 * (high - low + 1.0)
    axes.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        color="0.6",
        linestyle="--",
        linewidth=1,
    )
    for position, side in enumerate(("longer", "shorter")):
        part = table[table["side"] == side]
        axes.plot(
            part["true_m"],
            part["estimated_m"],
            linestyle="none",
            marker=SERIES_MARKERS[position],
            color=SERIES_COLORS[position],
            label=f"{side} side",
        )
    axes.set_xlabel("true side length (m)")
    axes.set_ylabel("estimated side length (m)")
    axes.set_title("Room size by echo inversion")
    axes.legend(fontsize=8)
    return figure
