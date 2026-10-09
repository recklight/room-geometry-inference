"""How sensitive the room estimate is to timing errors, the source model and noise.

Four sweeps around the ``room`` scene of the configuration, each a tidy
table with one row per estimate:

* ``delay_offset``: a constant offset (in samples, fractions allowed) is
  added to every exact first-order arrival, as an unknown playback latency or
  a timing bias would. The direct path absorbs most of it, so the source
  distance and with it every wall position shift.
* ``source_bearing``: the source is moved around the microphone at the
  scene's distance and the arrivals are inverted under the assumed bearing
  (``inversion.source_bearing_deg``). Because first-order arrivals cannot
  reveal the bearing (see :mod:`room_geometry_inference.inversion`), this
  measures the cost of a wrong assumption.
* ``source_shift``: the source is displaced along x and along y.
* ``snr``: the complete chain (simulated recording, identification, echo
  picking, inversion and ranking) at several signal-to-noise ratios, with
  ``n_trials`` independent noise realizations each.

Errors are measured without cancellation: ``room_error_m`` is the sum of
absolute errors of the sorted sides, ``distance_error_m`` the absolute error
of the source distance. ``tap_shift`` is the mean arrival change from the
reference arrivals in samples (for ``snr``: picked against true arrivals).
A failed inversion is a row with ``status`` set and the errors left empty,
so failure rates are part of the result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .echoes import distances_to_taps, tap_shift_error
from .errors import ConfigError, InversionError
from .geometry import room_size_error
from .inversion import best_estimate, first_order_distances, invert_echoes

if TYPE_CHECKING:
    from .config import Config
    from .inversion import RoomEstimate

_LOGGER = logging.getLogger(__name__)

#: Columns of every sweep table, in order.
SWEEP_COLUMNS: tuple[str, ...] = (
    "sweep",
    "parameter",
    "value",
    "trial",
    "status",
    "width_m",
    "depth_m",
    "source_distance_m",
    "room_error_m",
    "distance_error_m",
    "tap_shift",
)


def _scene_distances(
    size: tuple[float, ...], microphone: NDArray[np.float64], source: NDArray[np.float64]
) -> NDArray[np.float64]:
    offset = source - microphone
    return first_order_distances(size[0], size[1], microphone[0], microphone[1], *offset)


def _inside(size: tuple[float, ...], point: NDArray[np.float64]) -> bool:
    return bool(0 < point[0] < size[0] and 0 < point[1] < size[1])


def _estimate(distances: NDArray[np.float64], config: Config) -> RoomEstimate:
    candidates = invert_echoes(
        distances,
        bearing_deg=config.inversion.source_bearing_deg,
        refine=config.inversion.refine,
        max_candidates=config.inversion.max_candidates,
    )
    return best_estimate(candidates)


def _row(
    sweep: str,
    parameter: str,
    value: float,
    trial: int,
    estimate: RoomEstimate | None,
    truth: tuple[tuple[float, ...], float],
    tap_shift: float,
    status: str = "ok",
) -> dict[str, Any]:
    size, distance = truth
    nan = float("nan")
    return {
        "sweep": sweep,
        "parameter": parameter,
        "value": float(value),
        "trial": int(trial),
        "status": status,
        "width_m": estimate.width_m if estimate else nan,
        "depth_m": estimate.depth_m if estimate else nan,
        "source_distance_m": estimate.source_distance_m if estimate else nan,
        "room_error_m": room_size_error(estimate.size_m, size) if estimate else nan,
        "distance_error_m": abs(estimate.source_distance_m - distance) if estimate else nan,
        "tap_shift": float(tap_shift),
    }


def _invert_distances(
    sweep: str,
    parameter: str,
    value: float,
    distances: NDArray[np.float64],
    reference: NDArray[np.float64],
    truth: tuple[tuple[float, ...], float],
    config: Config,
) -> dict[str, Any]:
    acoustics = config.acoustics
    shift = tap_shift_error(
        distances_to_taps(distances, acoustics.sample_rate, acoustics.speed),
        distances_to_taps(reference, acoustics.sample_rate, acoustics.speed),
    )
    try:
        estimate = _estimate(distances, config)
    except InversionError:
        return _row(sweep, parameter, value, 0, None, truth, shift, "no_solution")
    return _row(sweep, parameter, value, 0, estimate, truth, shift)


def sweep_delay_offset(config: Config) -> pd.DataFrame:
    """Room estimates when every arrival is delayed by a constant offset.

    Parameters
    ----------
    config : Config
        ``room``, ``acoustics``, ``inversion`` and
        ``sensitivity.delay_offsets_samples`` are used.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SWEEP_COLUMNS`; ``parameter`` is ``offset``.

    Examples
    --------
    >>> from room_geometry_inference.config import Config
    >>> config = Config().override({"sensitivity.delay_offsets_samples": [0.0, 10.0]})
    >>> table = sweep_delay_offset(config)
    >>> table["room_error_m"].round(3).tolist(), table["tap_shift"].round(6).tolist()
    ([0.0, 0.346], [0.0, 10.0])
    """
    room, acoustics = config.room, config.acoustics
    microphone, source = np.asarray(room.microphone_m), np.asarray(room.source_m)
    exact = _scene_distances(room.size_m, microphone, source)
    truth = (room.size_m, float(np.linalg.norm(source - microphone)))
    meters_per_sample = acoustics.speed / acoustics.sample_rate
    rows = [
        _invert_distances(
            "delay_offset",
            "offset",
            offset,
            exact + offset * meters_per_sample,
            exact,
            truth,
            config,
        )
        for offset in config.sensitivity.delay_offsets_samples
    ]
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def sweep_source_bearing(config: Config) -> pd.DataFrame:
    """Room estimates when the true source bearing differs from the assumed one.

    The source keeps its distance from the microphone and moves to each of
    ``sensitivity.bearings_deg``; positions outside the room are reported
    with ``status = "outside_room"``.

    Examples
    --------
    >>> from room_geometry_inference.config import Config
    >>> config = Config().override({"sensitivity.bearings_deg": [45.0, 0.0, 90.0]})
    >>> sweep_source_bearing(config)["room_error_m"].round(3).tolist()
    [0.0, 0.114, 0.116]
    """
    room = config.room
    microphone, base = np.asarray(room.microphone_m), np.asarray(room.source_m)
    distance = float(np.linalg.norm(base - microphone))
    reference = _scene_distances(room.size_m, microphone, base)
    truth = (room.size_m, distance)
    rows = []
    for bearing in config.sensitivity.bearings_deg:
        theta = np.deg2rad(bearing)
        source = microphone + distance * np.array([np.cos(theta), np.sin(theta)])
        if not _inside(room.size_m, source):
            rows.append(
                _row(
                    "source_bearing", "bearing_deg", bearing, 0, None, truth, np.nan, "outside_room"
                )
            )
            continue
        distances = _scene_distances(room.size_m, microphone, source)
        rows.append(
            _invert_distances(
                "source_bearing", "bearing_deg", bearing, distances, reference, truth, config
            )
        )
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def sweep_source_shift(config: Config) -> pd.DataFrame:
    """Room estimates when the source is displaced along x and along y.

    Examples
    --------
    >>> from room_geometry_inference.config import Config
    >>> config = Config().override({"sensitivity.shifts_m": [0.0, 0.5]})
    >>> table = sweep_source_shift(config)
    >>> table["parameter"].tolist(), table["room_error_m"].round(3).tolist()
    (['x', 'x', 'y', 'y'], [0.0, 0.072, 0.0, 0.073])
    """
    room = config.room
    microphone, base = np.asarray(room.microphone_m), np.asarray(room.source_m)
    reference = _scene_distances(room.size_m, microphone, base)
    rows = []
    for axis, name in enumerate(("x", "y")):
        for shift in config.sensitivity.shifts_m:
            source = base.copy()
            source[axis] += shift
            truth = (room.size_m, float(np.linalg.norm(source - microphone)))
            if not _inside(room.size_m, source) or np.allclose(source, microphone):
                rows.append(
                    _row("source_shift", name, shift, 0, None, truth, np.nan, "outside_room")
                )
                continue
            distances = _scene_distances(room.size_m, microphone, source)
            rows.append(
                _invert_distances("source_shift", name, shift, distances, reference, truth, config)
            )
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def sweep_snr(config: Config, *, rng: np.random.Generator | int | None = None) -> pd.DataFrame:
    """The full estimation chain at several signal-to-noise ratios.

    Every trial simulates a new excitation and noise realization of the
    ``room`` scene (``synthetic`` settings), identifies the response,
    picks the echoes, inverts and ranks.

    Parameters
    ----------
    config : Config
        Complete configuration; ``sensitivity.snr_db`` and
        ``sensitivity.n_trials`` set the sweep.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SWEEP_COLUMNS`; ``parameter`` is ``snr_db``.

    Examples
    --------
    >>> from room_geometry_inference.config import Config
    >>> config = Config().override(
    ...     {"sensitivity.snr_db": [20.0], "sensitivity.n_trials": 1, "synthetic.duration_s": 0.5})
    >>> table = sweep_snr(config, rng=0)
    >>> table["status"].tolist(), bool(table["room_error_m"].iloc[0] < 0.05)
    (['ok'], True)
    """
    from .room_estimation import simulate_scene, simulate_truth  # deferred: avoids a cycle

    rng = np.random.default_rng(rng)
    room = config.room
    truth_response = simulate_truth(config)
    distance = float(np.linalg.norm(np.subtract(room.source_m, room.microphone_m)))
    truth = (room.size_m, distance)
    rows = []
    for snr in config.sensitivity.snr_db:
        for trial in range(config.sensitivity.n_trials):
            try:
                scene = simulate_scene(config, rng=rng, snr_db=snr, truth=truth_response)
            except InversionError:
                rows.append(_row("snr", "snr_db", snr, trial, None, truth, np.nan, "no_solution"))
                continue
            row = _row(
                "snr", "snr_db", snr, trial, scene.inversion.estimate, truth, scene.tap_shift
            )
            rows.append(row)
        _LOGGER.info("snr %g dB: %d trials done", snr, config.sensitivity.n_trials)
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


_SWEEPS: dict[str, Callable[..., pd.DataFrame]] = {
    "delay_offset": sweep_delay_offset,
    "source_bearing": sweep_source_bearing,
    "source_shift": sweep_source_shift,
    "snr": sweep_snr,
}
#: Sweeps accepted by :func:`run_sweeps` and ``sensitivity.sweeps``.
SWEEP_NAMES: tuple[str, ...] = tuple(_SWEEPS)


def run_sweeps(config: Config, *, rng: np.random.Generator | int | None = None) -> pd.DataFrame:
    """Every sweep listed in ``sensitivity.sweeps``, concatenated in that order.

    Raises
    ------
    ConfigError
        If no sweep is configured or a sweep name is unknown.

    Examples
    --------
    >>> from room_geometry_inference.config import Config
    >>> config = Config().override(
    ...     {"sensitivity.sweeps": ["delay_offset"], "sensitivity.delay_offsets_samples": [0.0]})
    >>> run_sweeps(config)[["sweep", "value"]].values.tolist()
    [['delay_offset', 0.0]]
    """
    rng = np.random.default_rng(rng)
    tables = []
    for name in config.sensitivity.sweeps:
        if name not in _SWEEPS:
            raise ConfigError(
                f"sensitivity.sweeps={name!r} is not supported; choose one of {list(SWEEP_NAMES)}"
            )
        _LOGGER.info("running the %s sweep", name)
        sweep = _SWEEPS[name]
        tables.append(sweep(config, rng=rng) if name == "snr" else sweep(config))
    if not tables:
        raise ConfigError("sensitivity.sweeps lists no sweep to run")
    return pd.concat(tables, ignore_index=True)


def summarize_sweeps(table: pd.DataFrame) -> pd.DataFrame:
    """Per sweep setting: trials, success rate, mean and sample std of the errors.

    Examples
    --------
    >>> rows = pd.DataFrame({"sweep": ["snr"] * 3, "parameter": ["snr_db"] * 3,
    ...     "value": [0.0, 0.0, 0.0], "status": ["ok", "ok", "no_solution"],
    ...     "room_error_m": [0.1, 0.3, float("nan")], "distance_error_m": [0.0, 0.0, float("nan")],
    ...     "tap_shift": [1.0, 2.0, float("nan")]})
    >>> summary = summarize_sweeps(rows)
    >>> summary[["n_trials", "success_rate", "room_error_mean_m"]].round(3).values.tolist()
    [[3.0, 0.667, 0.2]]
    """
    grouped = table.groupby(["sweep", "parameter", "value"], sort=False)
    summary = grouped.agg(
        n_trials=("status", "size"),
        success_rate=("status", lambda status: float(np.mean(status == "ok"))),
        room_error_mean_m=("room_error_m", "mean"),
        room_error_std_m=("room_error_m", "std"),
        distance_error_mean_m=("distance_error_m", "mean"),
        tap_shift_mean=("tap_shift", "mean"),
    )
    return summary.reset_index()
