"""The high-level steps shared by the command line and the examples.

* :func:`prepare_features`: dataset index to feature vectors (``prepare``);
* :func:`train_model`: one classifier on every measurement (``train``);
* :func:`predict_inputs`: new measurements through a saved model, using
  only the settings stored in the model file (``predict``);
* :func:`invert_measurements`: room size and source distance of every
  measurement by echo inversion, compared with the true geometry when the
  index has it (``invert``);
* :func:`denoising_pairs`: identified and true responses of every
  measurement with known geometry (``denoise``).

A measurement goes through :func:`measurement_response`: read the
two-channel WAV file at ``acoustics.sample_rate`` and identify the impulse
response from its excitation and microphone channels.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .dataset import FeatureSet, has_geometry, load_dataset_index
from .errors import DatasetError, InversionError
from .evaluation import class_scores, score_columns
from .geometry import room_size_error
from .identification import prune_small_taps
from .models import build_model, fit_estimator
from .recording import read_recording
from .rir_features import represent_response
from .room_estimation import identify_recording, invert_response, simulate_truth

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from .config import Config

_LOGGER = logging.getLogger(__name__)

#: Columns of the table returned by :func:`invert_measurements` (truth columns follow).
ESTIMATE_COLUMNS: tuple[str, ...] = (
    "path",
    "label",
    "group",
    "status",
    "n_echoes",
    "width_m",
    "depth_m",
    "source_distance_m",
    "mic_x_m",
    "mic_y_m",
    "source_x_m",
    "source_y_m",
    "residual_m",
    "score",
    "n_equivalent",
)
#: Columns added by :func:`invert_measurements` when the true geometry is known.
TRUTH_COLUMNS: tuple[str, ...] = (
    "true_width_m",
    "true_depth_m",
    "true_source_distance_m",
    "room_error_m",
    "distance_error_m",
)


def measurement_response(path: str | Path, config: Config) -> NDArray[np.float64]:
    """Identified impulse response of one measurement file.

    Parameters
    ----------
    path : str or pathlib.Path
        Two-channel WAV file (0: excitation, 1: microphone).
    config : Config
        ``acoustics.sample_rate`` and ``identification`` are used.

    Returns
    -------
    numpy.ndarray, shape (identification.n_taps,)
    """
    recording = read_recording(path, config.acoustics.sample_rate)
    if recording.excitation.size <= config.identification.n_taps:
        raise DatasetError(
            f"{Path(path).as_posix()} has {recording.excitation.size} samples at "
            f"{config.acoustics.sample_rate} Hz, not more than identification.n_taps="
            f"{config.identification.n_taps}"
        )
    return identify_recording(recording, config)


def response_features(response: ArrayLike, config: Config) -> NDArray[np.float64]:
    """The configured feature vector of an identified response.

    Examples
    --------
    >>> from room_geometry_inference.config import Config
    >>> h = np.zeros(1000)
    >>> h[[66, 307, 399]] = [1.0, 0.3, 0.2]
    >>> response_features(h, Config()).shape
    (1000,)
    """
    features = config.features
    return represent_response(
        response,
        features.representation,
        sample_rate=config.acoustics.sample_rate,
        relative_threshold=config.identification.prune_threshold,
        floor_db=features.floor_db,
        echo_options=config.echoes.options(),
        max_imfs=features.max_imfs,
        emd_tolerance=features.emd_tolerance,
    )


def prepare_features(dataset: str | Path, config: Config) -> FeatureSet:
    """Identify the response of every measurement of a dataset index and describe it.

    Parameters
    ----------
    dataset : str or pathlib.Path
        Dataset index (``dataset.csv``).
    config : Config
        Complete configuration; it is stored in the result.

    Returns
    -------
    FeatureSet
        One feature vector per measurement, in the order of the index.

    Raises
    ------
    FileNotFoundError
        If the index does not exist.
    DatasetError
        If the index or a measurement file is invalid.
    """
    index = load_dataset_index(dataset)
    rows = []
    for position, row in enumerate(index.itertuples(index=False), start=1):
        rows.append(response_features(measurement_response(row.path, config), config))
        if position % 10 == 0:
            _LOGGER.info("identified %d of %d measurements", position, len(index))
    features = FeatureSet(
        X=np.vstack(rows),
        y=index["label"].to_numpy(dtype=str),
        groups=index["group"].to_numpy(dtype=str),
        paths=index["source"].to_numpy(dtype=str),
        config=config.to_dict(),
    )
    _LOGGER.info(
        "prepared %d measurements, %d %s features each",
        features.n_items,
        features.X.shape[1],
        config.features.representation,
    )
    return features


def train_model(features: FeatureSet, config: Config) -> BaseEstimator:
    """Fit the configured classifier on every measurement.

    Parameters
    ----------
    features : FeatureSet
        Measurements written by ``prepare``.
    config : Config
        Complete configuration.

    Returns
    -------
    sklearn.base.BaseEstimator
        The fitted pipeline; store it with
        :func:`~room_geometry_inference.models.save_model`.
    """
    labels = np.asarray(features.y).astype(str)
    if np.unique(labels).size < 2:
        raise DatasetError(f"training needs at least two classes, found {sorted(set(labels))}")
    _LOGGER.info("training %s on %d measurements", config.model.name, features.n_items)
    return fit_estimator(build_model(config), features.X, labels)


def _predicted_labels(estimator: Any, X: NDArray[np.float64]) -> NDArray[np.str_]:
    """Predictions; the template matcher may also answer ``unknown``."""
    final = estimator[-1] if hasattr(estimator, "steps") else estimator
    if hasattr(final, "predict_with_unknown"):
        inputs = estimator[:-1].transform(X) if len(getattr(estimator, "steps", [])) > 1 else X
        return np.asarray(final.predict_with_unknown(inputs)).astype(str)
    return np.asarray(estimator.predict(X)).astype(str)


def predict_inputs(inputs: Sequence[str | Path], bundle: Mapping[str, Any]) -> pd.DataFrame:
    """Classify new measurements with a saved model.

    Only the configuration stored in the model file is used, so the inputs
    are identified and described exactly as the training data were.

    Parameters
    ----------
    inputs : sequence of str or pathlib.Path
        Two-channel measurement WAV files.
    bundle : mapping
        Model bundle returned by :func:`~room_geometry_inference.models.load_model`.

    Returns
    -------
    pandas.DataFrame
        One row per input (``predictions.csv``): ``path``, ``predicted`` and
        ``score_<class>``. The template matcher answers ``unknown`` when no
        stored room matches well (``model.template_min_score``); such
        measurements are candidates for ``invert``.
    """
    from .config import Config  # deferred: config imports this package's modules

    config = Config.from_dict(bundle["config"])
    estimator = bundle["estimator"]
    classes = [str(name) for name in bundle["classes"]]
    if not inputs:
        raise DatasetError("no input files were given")
    X = np.vstack(
        [response_features(measurement_response(path, config), config) for path in inputs]
    )
    scores = class_scores(estimator, X, classes)
    table = pd.DataFrame(
        {
            "path": [Path(path).as_posix() for path in inputs],
            "predicted": _predicted_labels(estimator, X),
        }
    )
    for position, column in enumerate(score_columns(classes)):
        table[column] = scores[:, position]
    return table


def _truth(row: Mapping[str, Any]) -> tuple[tuple[float, float], float]:
    size = (float(row["room_width_m"]), float(row["room_depth_m"]))
    distance = float(
        np.hypot(row["source_x_m"] - row["mic_x_m"], row["source_y_m"] - row["mic_y_m"])
    )
    return size, distance


def invert_measurements(
    inputs: Sequence[str | Path],
    config: Config,
    *,
    index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Room size and source distance of every measurement by echo inversion.

    Parameters
    ----------
    inputs : sequence of str or pathlib.Path
        Two-channel measurement WAV files.
    config : Config
        Complete configuration (``acoustics``, ``identification``, ``echoes``
        and ``inversion``).
    index : pandas.DataFrame, optional
        The loaded dataset index the inputs come from (same order); it
        supplies labels, groups and, when present, the true geometry.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`ESTIMATE_COLUMNS`, followed by :data:`TRUTH_COLUMNS`
        when the true geometry is known. A measurement whose echoes admit no
        layout has ``status = "no_solution"`` and empty estimates.
    """
    if index is not None and len(index) != len(inputs):
        raise DatasetError(f"the index has {len(index)} rows for {len(inputs)} inputs")
    with_truth = index is not None and has_geometry(index)
    rows = []
    for position, path in enumerate(inputs):
        meta = index.iloc[position] if index is not None else None
        row: dict[str, Any] = {
            "path": str(meta["source"]) if meta is not None else Path(path).as_posix(),
            "label": str(meta["label"]) if meta is not None else "",
            "group": str(meta["group"]) if meta is not None else "",
        }
        response = measurement_response(path, config)
        try:
            result = invert_response(response, config)
        except InversionError as exc:
            _LOGGER.warning("%s: %s", row["path"], exc)
            row.update(status="no_solution", n_echoes=0)
        else:
            estimate = result.estimate
            row.update(
                status="ok",
                n_echoes=int(result.echo_taps.size),
                width_m=estimate.width_m,
                depth_m=estimate.depth_m,
                source_distance_m=estimate.source_distance_m,
                mic_x_m=estimate.mic_xy[0],
                mic_y_m=estimate.mic_xy[1],
                source_x_m=estimate.source_xy[0],
                source_y_m=estimate.source_xy[1],
                residual_m=estimate.residual_m,
                score=estimate.score,
                n_equivalent=estimate.n_equivalent,
            )
        if with_truth and meta is not None:
            size, distance = _truth(meta)
            row.update(
                true_width_m=max(size),
                true_depth_m=min(size),
                true_source_distance_m=distance,
            )
            if row["status"] == "ok":
                row["room_error_m"] = room_size_error((row["width_m"], row["depth_m"]), size)
                row["distance_error_m"] = abs(row["source_distance_m"] - distance)
        rows.append(row)
    columns = list(ESTIMATE_COLUMNS) + (list(TRUTH_COLUMNS) if with_truth else [])
    table = pd.DataFrame(rows).reindex(columns=columns)
    n_ok = int(np.sum(table["status"] == "ok"))
    _LOGGER.info("inverted %d of %d measurements", n_ok, len(table))
    return table


def denoising_pairs(
    dataset: str | Path, config: Config
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.str_]]:
    """Identified (noisy) and true (clean) responses of every measurement.

    The true response is simulated from the geometry columns of the index
    with the ``acoustics`` settings, so the index must carry them (synthetic
    data always do).

    Parameters
    ----------
    dataset : str or pathlib.Path
        Dataset index with the columns of
        :data:`~room_geometry_inference.dataset.GEOMETRY_COLUMNS`.
    config : Config
        Complete configuration.

    Returns
    -------
    noisy, clean : numpy.ndarray, shape (n_items, identification.n_taps)
        Peak-normalized magnitudes (the noisy one pruned like every response).
    groups : numpy.ndarray of str, shape (n_items,)
        The ``group`` column (rooms).

    Raises
    ------
    DatasetError
        If the index has no geometry columns.
    """
    index = load_dataset_index(dataset, require_label=False)
    if not has_geometry(index):
        raise DatasetError(
            f"{Path(dataset).as_posix()} has no geometry columns; denoising needs the true "
            "responses (synthetic data have them)"
        )
    noisy, clean = [], []
    n_taps = config.identification.n_taps
    for row in index.to_dict(orient="records"):
        response = measurement_response(row["path"], config)
        noisy.append(prune_small_taps(response, config.identification.prune_threshold))
        truth = simulate_truth(
            config,
            room_m=(row["room_width_m"], row["room_depth_m"]),
            microphone_m=(row["mic_x_m"], row["mic_y_m"]),
            source_m=(row["source_x_m"], row["source_y_m"]),
            n_taps=n_taps,
        )
        clean.append(np.abs(truth.rir))
    _LOGGER.info("built %d noisy/clean response pairs", len(noisy))
    return np.vstack(noisy), np.vstack(clean), index["group"].to_numpy(dtype=str)
