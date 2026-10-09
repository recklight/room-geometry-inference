"""Synthetic measurement datasets with known room geometry.

Every class of :class:`~room_geometry_inference.config.SyntheticConfig` is a
nominal room size. For each class, ``n_rooms_per_class`` distinct rooms are
drawn with sides within ``room_jitter_m`` of the nominal ones, and each room
is measured ``n_recordings_per_room`` times with a new placement: a direct
path drawn from ``source_distance_range_m``, the source at
``source_bearing_deg`` from the microphone, both at least ``wall_margin_m``
from every wall. A measurement plays a synthetic excitation through the
image-source response of the room (``acoustics`` settings) and adds noise at
``snr_db``; it is written as a two-channel 16-bit WAV file (channel 0
excitation, channel 1 microphone), exactly the format expected of real
measurements.

The dataset index lists every file with its class, its room as the
``group`` (so cross-validation never trains and tests on the same room) and
the true geometry in the columns of
:data:`~room_geometry_inference.dataset.GEOMETRY_COLUMNS`, rounded to
millimeters before the simulation so the stored values are the simulated
ones. Rooms of one class differ, and placements differ within a room, so a
classifier has to generalize to rooms it has not seen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import AcousticsConfig, SyntheticConfig
from .dataset import DATASET_INDEX_NAME, GEOMETRY_COLUMNS
from .excitation import make_excitation
from .image_source import image_source_rir
from .recording import simulate_recording, write_recording

_LOGGER = logging.getLogger(__name__)

#: Rounding of synthetic geometry, in decimals of a meter (millimeters).
_GEOMETRY_DECIMALS = 3


def make_room_size(
    nominal_m: tuple[float, ...],
    jitter_m: float,
    *,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Room sides drawn uniformly within ``jitter_m`` of the nominal sides (millimeters).

    Examples
    --------
    >>> size = make_room_size((10.7, 7.5), 0.3, rng=0)
    >>> bool(np.all(np.abs(size - [10.7, 7.5]) <= 0.3)), size.tolist() == size.round(3).tolist()
    (True, True)
    """
    rng = np.random.default_rng(rng)
    nominal = np.asarray(nominal_m, dtype=np.float64)
    sides = nominal + rng.uniform(-jitter_m, jitter_m, size=nominal.size)
    return np.round(sides, _GEOMETRY_DECIMALS)


def make_placement(
    size_m: NDArray[np.float64],
    distance_range_m: tuple[float, ...],
    bearing_deg: float,
    margin_m: float,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A random microphone position and the source at the given bearing from it.

    Parameters
    ----------
    size_m : numpy.ndarray, shape (2,)
        Room sides.
    distance_range_m : tuple of float
        Direct-path distance drawn uniformly from ``[low, high]``.
    bearing_deg : float
        Source bearing seen from the microphone.
    margin_m : float
        Minimum distance of both points from every wall.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    microphone, source : numpy.ndarray, shape (2,)
        Positions rounded to millimeters.

    Examples
    --------
    >>> mic, src = make_placement(np.array([3.5, 3.0]), (0.8, 1.6), 45.0, 0.3, rng=0)
    >>> bearing = np.degrees(np.arctan2(src[1] - mic[1], src[0] - mic[0]))
    >>> bool(np.all((mic >= 0.3) & (src <= [3.2, 2.7]))), round(float(bearing))
    (True, 45)
    """
    rng = np.random.default_rng(rng)
    distance = rng.uniform(*distance_range_m)
    theta = np.deg2rad(bearing_deg)
    offset = distance * np.array([np.cos(theta), np.sin(theta)])
    low = margin_m + np.maximum(0.0, -offset)
    high = np.asarray(size_m, dtype=np.float64) - margin_m - np.maximum(0.0, offset)
    if np.any(high <= low):
        raise ValueError(
            f"a {distance:.2f} m source does not fit in the {size_m.tolist()} m room "
            f"with {margin_m} m margins"
        )
    microphone = np.round(rng.uniform(low, high), _GEOMETRY_DECIMALS)
    source = np.round(microphone + offset, _GEOMETRY_DECIMALS)
    return microphone, source


def write_synthetic_dataset(
    folder: str | Path,
    config: SyntheticConfig | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    acoustics: AcousticsConfig | None = None,
) -> Path:
    """Write synthetic measurements and their ``dataset.csv`` index.

    Parameters
    ----------
    folder : str or pathlib.Path
        Output folder; files go to ``<label>/<label>_<NNN>.wav``.
    config : SyntheticConfig, optional
        Sizes and generative parameters (defaults to ``SyntheticConfig()``).
    rng : numpy.random.Generator or int, optional
        Random generator or seed; the same seed writes identical files.
    acoustics : AcousticsConfig, optional
        Sampling rate and image-source model (defaults to ``AcousticsConfig()``).

    Returns
    -------
    pathlib.Path
        The written ``dataset.csv``.

    Examples
    --------
    >>> import tempfile
    >>> from room_geometry_inference.config import SyntheticConfig
    >>> small = SyntheticConfig(n_rooms_per_class=1, n_recordings_per_room=2, duration_s=0.2)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     index = pd.read_csv(write_synthetic_dataset(folder, small, rng=0))
    >>> index["path"].tolist()
    ['large/large_000.wav', 'large/large_001.wav', 'small/small_000.wav', 'small/small_001.wav']
    >>> index["group"].unique().tolist(), list(index.columns[3:5])
    (['large_room_00', 'small_room_00'], ['room_width_m', 'room_depth_m'])
    """
    settings = SyntheticConfig() if config is None else config
    physics = AcousticsConfig() if acoustics is None else acoustics
    rng = np.random.default_rng(rng)
    root = Path(folder)
    rate = physics.sample_rate
    n_samples = round(settings.duration_s * rate)
    rows = []
    for label, nominal in zip(settings.class_names, settings.room_sizes_m, strict=True):
        count = 0
        for room in range(settings.n_rooms_per_class):
            size = make_room_size(nominal, settings.room_jitter_m, rng=rng)
            group = f"{label}_room_{room:02d}"
            for _ in range(settings.n_recordings_per_room):
                microphone, source = make_placement(
                    size,
                    settings.source_distance_range_m,
                    settings.source_bearing_deg,
                    settings.wall_margin_m,
                    rng=rng,
                )
                response = image_source_rir(
                    size,
                    source,
                    microphone,
                    sample_rate=rate,
                    n_taps=physics.rir_taps,
                    speed_of_sound_m_s=physics.speed,
                    max_order=physics.max_order,
                    order_mode=physics.order_mode,
                    reflection=physics.reflection,
                    delay=physics.delay,
                ).rir
                excitation = make_excitation(settings.excitation, n_samples, rate, rng=rng)
                recording = simulate_recording(
                    excitation,
                    response,
                    snr_db=settings.snr_db,
                    noise_kind=settings.noise,
                    sample_rate=rate,
                    rng=rng,
                )
                relative = f"{label}/{label}_{count:03d}.wav"
                write_recording(root / relative, excitation, recording, rate)
                rows.append(
                    [relative, label, group, *size.tolist(), *microphone.tolist(), *source.tolist()]
                )
                count += 1
    table = pd.DataFrame(rows, columns=["path", "label", "group", *GEOMETRY_COLUMNS])
    destination = root / DATASET_INDEX_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False, lineterminator="\n")
    _LOGGER.info(
        "wrote %d synthetic measurements of %d rooms to %s",
        len(table),
        table["group"].nunique(),
        root.as_posix(),
    )
    return destination
