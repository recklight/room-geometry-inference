"""Echo arrivals in an impulse response: picking, timing and comparison.

An identified impulse response is a handful of sharp peaks (the direct sound
and the reflections) on top of estimation noise. There are two pickers:

* ``peaks`` (default): local maxima of :math:`|h|` found by
  :func:`scipy.signal.find_peaks`, at least ``min_separation_taps`` apart and
  at least ``relative_height`` of the largest peak, of which the
  ``n_echoes`` strongest are kept. The separation rule stops the side lobes
  of one smeared arrival from being counted as several echoes.
* ``top_k``: the ``n_echoes`` largest magnitudes, whatever their spacing.
  This is the naive rule; adjacent taps of one blurred arrival can fill
  several of the slots, as the first example of :func:`pick_echoes` shows.

Picked taps can be refined to sub-sample precision by fitting a parabola
through each peak and its two neighbors. Arrival samples become distances by
:math:`d = t c / f_s` (tap 0 is the emission time). :func:`tap_shift_error`
compares two sets of arrivals after pairing them optimally, so a missing or
spurious echo does not shift every later comparison.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

#: Echo pickers accepted by ``echoes.method``.
PICK_METHODS: tuple[str, ...] = ("peaks", "top_k")


def _magnitude(response: ArrayLike) -> NDArray[np.float64]:
    values = np.abs(np.asarray(response, dtype=np.float64))
    if values.ndim != 1 or values.size < 3:
        raise ValueError(
            f"response must be one-dimensional with at least 3 taps, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("response contains NaN or infinite values")
    return values


def pick_echoes(
    response: ArrayLike,
    n_echoes: int = 9,
    *,
    method: str = "peaks",
    min_separation_taps: int = 3,
    relative_height: float = 0.02,
    interpolate: bool = False,
    compensate_spreading: bool = False,
) -> NDArray[np.float64]:
    """Arrival samples of the strongest echoes, in increasing order.

    Parameters
    ----------
    response : array_like, shape (n_taps,)
        Impulse response (its magnitude is used).
    n_echoes : int, optional
        Number of arrivals to keep; fewer are returned when the response has
        fewer qualifying peaks.
    method : {"peaks", "top_k"}, optional
        Picking rule (see the module documentation).
    min_separation_taps : int, optional
        Minimum distance between two peaks (``peaks`` only).
    relative_height : float, optional
        Peaks whose magnitude is below this fraction of the largest
        magnitude are ignored (``peaks`` only; the delay weighting of
        ``compensate_spreading`` does not apply to this test).
    interpolate : bool, optional
        Refine each arrival with a parabola through the peak and its two
        neighbors.
    compensate_spreading : bool, optional
        Rank the peaks by magnitude times delay. A reflection's amplitude
        falls as :math:`1/d` with its path length while its delay grows as
        :math:`d`, so the weighted echoes have equal heights and distant
        reflections are not outranked by near ones.

    Returns
    -------
    numpy.ndarray, shape (n_found,)
        Arrival positions in samples (whole numbers unless ``interpolate``).

    Examples
    --------
    >>> h = np.zeros(100)
    >>> h[[10, 11, 40, 70]] = [1.0, 0.9, 0.6, 0.5]
    >>> pick_echoes(h, 3).tolist(), pick_echoes(h, 3, method="top_k").tolist()
    ([10.0, 40.0, 70.0], [10.0, 11.0, 40.0])
    >>> g = np.zeros(100)
    >>> g[[10, 20, 40, 70, 90]] = [1.0, 0.25, 0.6, 0.5, 0.2]  # 90: distant, weaker than 20
    >>> pick_echoes(g, 4).tolist(), pick_echoes(g, 4, compensate_spreading=True).tolist()
    ([10.0, 20.0, 40.0, 70.0], [10.0, 40.0, 70.0, 90.0])
    """
    magnitude = _magnitude(response)
    ranking = magnitude * np.arange(magnitude.size) if compensate_spreading else magnitude
    if int(n_echoes) < 1:
        raise ValueError(f"n_echoes must be at least 1, got {n_echoes!r}")
    if method == "top_k":
        order = np.argsort(-ranking, kind="stable")[: int(n_echoes)]
        taps = np.sort(order)
    elif method == "peaks":
        peak = float(magnitude.max())
        if peak == 0.0:
            return np.zeros(0)
        padded = np.concatenate([[0.0], ranking, [0.0]])
        found, _ = find_peaks(padded, distance=max(1, int(min_separation_taps)))
        found = found - 1
        # The height test uses the plain magnitude, so the delay weighting
        # cannot lift the late estimation noise above the threshold.
        found = found[magnitude[found] >= float(relative_height) * peak]
        strongest = np.argsort(-ranking[found], kind="stable")[: int(n_echoes)]
        taps = np.sort(found[strongest])
    else:
        raise ValueError(f"method={method!r} is not supported; choose one of {PICK_METHODS}")
    positions = taps.astype(np.float64)
    if interpolate and taps.size:
        positions = positions + _parabolic_offsets(magnitude, taps)
    return positions


def _parabolic_offsets(
    magnitude: NDArray[np.float64], taps: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Sub-sample offset of the vertex of the parabola through each peak and its neighbors."""
    inner = (taps > 0) & (taps < magnitude.size - 1)
    offsets = np.zeros(taps.size)
    left = magnitude[np.clip(taps - 1, 0, None)]
    center = magnitude[taps]
    right = magnitude[np.clip(taps + 1, None, magnitude.size - 1)]
    curvature = left - 2.0 * center + right
    usable = inner & (curvature < 0)
    offsets[usable] = 0.5 * (left - right)[usable] / curvature[usable]
    return np.clip(offsets, -0.5, 0.5)


def taps_to_distances(
    taps: ArrayLike, sample_rate: float, speed_of_sound_m_s: float = 343.0
) -> NDArray[np.float64]:
    """Travel distances of arrivals given in samples after emission.

    Examples
    --------
    >>> taps_to_distances([0, 160, 320], 16_000).round(3).tolist()
    [0.0, 3.43, 6.86]
    """
    if not sample_rate > 0 or not speed_of_sound_m_s > 0:
        raise ValueError("sample_rate and speed_of_sound_m_s must be positive")
    return np.asarray(taps, dtype=np.float64) * float(speed_of_sound_m_s) / float(sample_rate)


def distances_to_taps(
    distances_m: ArrayLike, sample_rate: float, speed_of_sound_m_s: float = 343.0
) -> NDArray[np.float64]:
    """Arrival samples of travel distances (not rounded); inverse of :func:`taps_to_distances`.

    Examples
    --------
    >>> distances_to_taps([3.43], 16_000).tolist()
    [160.0]
    """
    if not sample_rate > 0 or not speed_of_sound_m_s > 0:
        raise ValueError("sample_rate and speed_of_sound_m_s must be positive")
    return (
        np.asarray(distances_m, dtype=np.float64) * float(sample_rate) / float(speed_of_sound_m_s)
    )


def tap_shift_error(estimated: ArrayLike, reference: ArrayLike) -> float:
    """Mean absolute shift between two sets of arrivals after optimal pairing.

    Arrivals are paired one to one so that the total absolute difference is
    smallest (Hungarian algorithm); unpaired arrivals of the larger set are
    ignored.

    Parameters
    ----------
    estimated, reference : array_like
        Arrival positions in samples (any order).

    Returns
    -------
    float
        Mean absolute difference of the paired arrivals, in samples; ``nan``
        when either set is empty.

    Examples
    --------
    >>> tap_shift_error([12, 31, 50], [10, 30, 50, 90])
    1.0
    """
    first = np.asarray(estimated, dtype=np.float64).ravel()
    second = np.asarray(reference, dtype=np.float64).ravel()
    if first.size == 0 or second.size == 0:
        return float("nan")
    cost = np.abs(first[:, None] - second[None, :])
    rows, columns = linear_sum_assignment(cost)
    return float(cost[rows, columns].mean())
