"""Dataset index (``dataset.csv``) and the model inputs written by ``prepare``.

The index is one CSV file. Its ``path`` column holds two-channel measurement
WAV files (channel 0 excitation, channel 1 microphone) relative to the folder
of the CSV (POSIX separators), ``label`` the room class and the optional
``group`` column the room itself: every measurement of one room stays in the
same cross-validation fold. When the geometry is known (always for
synthetic data, for real data if you measured it), the columns of
:data:`GEOMETRY_COLUMNS` hold it, and ``invert`` and ``denoise`` compare
their results with it. File names carry no meaning; labels, groups and
geometry are read from the columns only.

``prepare`` turns the index into a :class:`FeatureSet`, one feature vector
per measurement, and stores it as ``features.npz`` with the keys ``X``,
``y``, ``groups``, ``paths`` and ``config_json``. The archive is written
with fixed time stamps, so the same input and settings give a
byte-identical file. :func:`save_svmlight` also exports the features in the
sparse LIBSVM text format (Chang & Lin, 2011) for other tools.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

from .errors import ConfigError, DatasetError

if TYPE_CHECKING:
    from .config import Config

_LOGGER = logging.getLogger(__name__)

#: File name of the dataset index written by ``synthesize`` and ``index``.
DATASET_INDEX_NAME: str = "dataset.csv"
#: Columns every dataset index must have (``label`` only for classification).
REQUIRED_COLUMNS: tuple[str, ...] = ("path", "label")
#: Optional columns with the true planar geometry, in meters.
GEOMETRY_COLUMNS: tuple[str, ...] = (
    "room_width_m",
    "room_depth_m",
    "mic_x_m",
    "mic_y_m",
    "source_x_m",
    "source_y_m",
)
#: Audio file extensions picked up by :func:`index_audio_folder`.
AUDIO_SUFFIXES: tuple[str, ...] = (".wav",)

#: Time stamp written into every ``features.npz`` member (the zip epoch).
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_NPZ_KEYS = ("X", "y", "groups", "paths", "config_json")


@dataclass(frozen=True, eq=False)
class FeatureSet:
    """Model inputs: one row per measurement.

    Attributes
    ----------
    X : numpy.ndarray of float64, shape (n_items, n_features)
        Feature vectors (see :mod:`room_geometry_inference.rir_features`).
    y : numpy.ndarray of str, shape (n_items,)
        Room class of each measurement.
    groups : numpy.ndarray of str, shape (n_items,)
        Room of each measurement; empty strings when the index has no
        ``group`` column.
    paths : numpy.ndarray of str, shape (n_items,)
        Measurement file, as written in the dataset index.
    config : dict
        The complete configuration used by ``prepare`` (``Config.to_dict()``).
    """

    X: NDArray[np.float64]
    y: NDArray[np.str_]
    groups: NDArray[np.str_]
    paths: NDArray[np.str_]
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise DatasetError(f"X must be two-dimensional, got shape {self.X.shape}")
        n_items = self.X.shape[0]
        for name, values in (("y", self.y), ("groups", self.groups), ("paths", self.paths)):
            if values.shape != (n_items,):
                raise DatasetError(
                    f"{name} must have one entry per row of X ({n_items}), got shape {values.shape}"
                )

    @property
    def n_items(self) -> int:
        """Number of measurements."""
        return int(self.X.shape[0])

    @property
    def classes(self) -> list[str]:
        """Class names in sorted order."""
        return sorted({str(label) for label in self.y})


def load_dataset_index(path: str | Path, *, require_label: bool = True) -> pd.DataFrame:
    """Read and validate a dataset index.

    Parameters
    ----------
    path : str or pathlib.Path
        The CSV file.
    require_label : bool, optional
        Whether the ``label`` column is required (classification); without
        it, an empty ``label`` column is added.

    Returns
    -------
    pandas.DataFrame
        One row per measurement with the columns ``path`` (absolute),
        ``source`` (the path as written, POSIX), ``label`` and ``group``
        (empty strings when the index has none), plus any extra columns; the
        geometry columns, when present, are numbers.

    Raises
    ------
    FileNotFoundError
        If the index does not exist.
    DatasetError
        If a required column is missing, a value is empty or not a number, or
        listed files do not exist.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     _ = (Path(folder) / "a.wav").write_bytes(b"")
    ...     _ = (Path(folder) / "dataset.csv").write_text("path,label\\na.wav,small\\n")
    ...     index = load_dataset_index(Path(folder) / "dataset.csv")
    >>> index["source"].tolist(), index["group"].tolist(), Path(index["path"][0]).is_absolute()
    (['a.wav'], [''], True)
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"dataset index not found: {source}. Point data.dataset (or --dataset) at your own "
            "dataset.csv, or run 'room-geometry-inference synthesize' first"
        )
    try:
        table = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{source} is not a readable CSV file: {exc}") from exc
    required = REQUIRED_COLUMNS if require_label else ("path",)
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise DatasetError(f"{source} lacks the column(s) {missing}; found {list(table.columns)}")
    if table.empty:
        raise DatasetError(f"{source} lists no recordings")
    for column in required:
        empty = table[column].str.strip() == ""
        if empty.any():
            rows = (np.flatnonzero(empty.to_numpy()) + 2).tolist()[:5]
            raise DatasetError(f"{source}: empty {column!r} on line(s) {rows}")
    for column in ("label", "group"):
        if column not in table.columns:
            table[column] = ""
    present = [column for column in GEOMETRY_COLUMNS if column in table.columns]
    if present and len(present) != len(GEOMETRY_COLUMNS):
        absent = [column for column in GEOMETRY_COLUMNS if column not in present]
        raise DatasetError(f"{source} has some geometry columns but lacks {absent}")
    for column in present:
        numbers = pd.to_numeric(table[column], errors="coerce")
        if numbers.isna().any():
            rows = (np.flatnonzero(numbers.isna().to_numpy()) + 2).tolist()[:5]
            raise DatasetError(f"{source}: {column!r} is not a number on line(s) {rows}")
        table[column] = numbers.astype(np.float64)
    base = source.resolve().parent
    table["source"] = table["path"].str.replace("\\", "/", regex=False)
    table["path"] = [str((base / entry).resolve()) for entry in table["source"]]
    absent_files = [
        entry
        for entry, full in zip(table["source"], table["path"], strict=True)
        if not Path(full).is_file()
    ]
    if absent_files:
        listed = ", ".join(absent_files[:5])
        more = f" ... and {len(absent_files) - 5} more" if len(absent_files) > 5 else ""
        raise DatasetError(
            f"{len(absent_files)} file(s) listed in {source} do not exist (paths are relative "
            f"to {base.as_posix()}): {listed}{more}"
        )
    ordered = ["path", "source", "label", "group"]
    return table[ordered + [column for column in table.columns if column not in ordered]]


def has_geometry(index: pd.DataFrame) -> bool:
    """Whether a loaded index carries the true geometry columns.

    Examples
    --------
    >>> has_geometry(pd.DataFrame({"path": ["a.wav"]}))
    False
    """
    return all(column in index.columns for column in GEOMETRY_COLUMNS)


def index_audio_folder(folder: str | Path, *, overwrite: bool = False) -> Path:
    """Write ``dataset.csv`` for a ``<folder>/<label>/<file>.wav`` layout.

    Every sub-folder is a class. Each measurement becomes its own group (its
    path without the extension); edit the ``group`` column so that all
    measurements of one room share a group, and add the geometry columns if
    you know them.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder holding one sub-folder per class.
    overwrite : bool, optional
        Replace an existing ``dataset.csv``. Off by default, because its
        ``group`` and geometry columns are usually edited by hand.

    Returns
    -------
    pathlib.Path
        The written index, ``<folder>/dataset.csv``.

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist.
    DatasetError
        If no audio file is found, or if ``dataset.csv`` already exists and
        ``overwrite`` is false.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     for name in ("small/a.wav", "large/b.wav"):
    ...         (Path(folder) / name).parent.mkdir(exist_ok=True)
    ...         _ = (Path(folder) / name).write_bytes(b"")
    ...     print(index_audio_folder(folder).read_text(), end="")
    path,label,group
    large/b.wav,large,large/b
    small/a.wav,small,small/a
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"folder not found: {root}")
    destination = root / DATASET_INDEX_NAME
    if destination.exists() and not overwrite:
        raise DatasetError(
            f"{destination.as_posix()} already exists and was left unchanged; "
            "use --force (overwrite=True) to rebuild it"
        )
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("*/*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        raise DatasetError(
            f"no {'/'.join(AUDIO_SUFFIXES)} files found in the class sub-folders of {root}"
        )
    table = pd.DataFrame(
        {
            "path": files,
            "label": [entry.split("/")[0] for entry in files],
            "group": [entry.rsplit(".", 1)[0] for entry in files],
        }
    )
    table.to_csv(destination, index=False, lineterminator="\n")
    _LOGGER.info("indexed %d measurements in %d classes", len(table), table["label"].nunique())
    return destination


def items_table(features: FeatureSet) -> pd.DataFrame:
    """One row per measurement: ``path``, ``label``, ``group`` (``items.csv``)."""
    return pd.DataFrame({"path": features.paths, "label": features.y, "group": features.groups})


def _write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    """``np.savez_compressed`` with fixed member time stamps (reproducible bytes)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_EPOCH)
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, buffer.getvalue())


def save_features(path: str | Path, features: FeatureSet) -> Path:
    """Write a :class:`FeatureSet` as ``features.npz`` and return the path.

    Examples
    --------
    >>> import tempfile
    >>> data = FeatureSet(
    ...     X=np.zeros((2, 4)), y=np.array(["large", "small"]),
    ...     groups=np.array(["", ""]), paths=np.array(["a.wav", "b.wav"]), config={"seed": 0},
    ... )
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     loaded = load_features(save_features(Path(folder) / "features.npz", data))
    >>> loaded.y.tolist(), loaded.config, loaded.X.dtype.name
    (['large', 'small'], {'seed': 0}, 'float64')
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": np.ascontiguousarray(features.X, dtype=np.float64),
        "y": np.asarray(features.y, dtype=str),
        "groups": np.asarray(features.groups, dtype=str),
        "paths": np.asarray(features.paths, dtype=str),
        "config_json": np.array(json.dumps(features.config, ensure_ascii=False)),
    }
    _write_npz(destination, arrays)
    return destination


def _differing_keys(
    stored: Mapping[str, Any], current: Mapping[str, Any], prefix: str
) -> list[str]:
    keys = sorted(set(stored) | set(current))
    return [f"{prefix}.{key}" for key in keys if stored.get(key) != current.get(key)]


def load_features(path: str | Path, config: Config | None = None) -> FeatureSet:
    """Read ``features.npz``; with ``config``, check it matches the feature settings.

    Parameters
    ----------
    path : str or pathlib.Path
        File written by :func:`save_features`.
    config : Config, optional
        Current configuration. Its ``FEATURE_SECTIONS`` must equal the ones
        stored by ``prepare``.

    Returns
    -------
    FeatureSet

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DatasetError
        If the file is not a features archive of this package.
    ConfigError
        If the feature settings differ from the ones ``prepare`` used.
    """
    from .config import FEATURE_SECTIONS  # deferred: config imports this package's modules

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"features file not found: {source}. Run 'prepare' first")
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = [key for key in _NPZ_KEYS if key not in archive.files]
            if missing:
                raise DatasetError(f"{source} is not a features file; missing {missing}")
            stored_config = json.loads(str(archive["config_json"]))
            features = FeatureSet(
                X=archive["X"].astype(np.float64),
                y=archive["y"],
                groups=archive["groups"],
                paths=archive["paths"],
                config=stored_config,
            )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"{source} is not a readable features file: {exc}") from exc
    if config is not None:
        current = config.to_dict()
        differing: list[str] = []
        for section in FEATURE_SECTIONS:
            differing += _differing_keys(
                stored_config.get(section, {}), current.get(section, {}), section
            )
        if differing:
            raise ConfigError(
                f"feature settings differ from the ones 'prepare' used: {', '.join(differing)}; "
                "run 'prepare' again or use the same --config"
            )
    return features


def save_svmlight(path: str | Path, features: FeatureSet) -> Path:
    """Export features in the LIBSVM sparse text format.

    Classes are coded as integers in sorted order (0, 1, ...); with two
    classes the positive class of LIBSVM is the second one.

    References
    ----------
    .. [1] C.-C. Chang and C.-J. Lin, "LIBSVM: A library for support vector
       machines," ACM Transactions on Intelligent Systems and Technology,
       vol. 2, no. 3, article 27, 2011.

    Examples
    --------
    >>> import tempfile
    >>> data = FeatureSet(X=np.array([[0.0, 1.5], [2.0, 0.0]]), y=np.array(["small", "large"]),
    ...                   groups=np.array(["", ""]), paths=np.array(["a", "b"]), config={})
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     target = save_svmlight(Path(folder) / "features.svm", data)
    ...     text = target.read_text().splitlines()
    ...     X, y = load_svmlight(target)
    >>> text, X.tolist(), y.tolist()
    (['1 2:1.5', '0 1:2'], [[0.0, 1.5], [2.0, 0.0]], [1, 0])
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    codes = np.searchsorted(np.asarray(features.classes), np.asarray(features.y).astype(str))
    dump_svmlight_file(features.X, codes, str(destination), zero_based=False)
    return destination


def load_svmlight(
    path: str | Path, n_features: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Read a LIBSVM sparse text file as a dense matrix and integer labels."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LIBSVM file not found: {source}")
    try:
        sparse, labels = load_svmlight_file(str(source), n_features=n_features, zero_based=False)
    except ValueError as exc:
        raise DatasetError(f"{source} is not a readable LIBSVM file: {exc}") from exc
    return np.asarray(sparse.toarray(), dtype=np.float64), labels.astype(np.int64)
