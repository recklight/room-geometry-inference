"""Fixed-length feature vectors computed from an identified impulse response.

Four representations of the same response, from raw to compact:

* ``magnitude``: the peak-normalized magnitude :math:`|\\hat h| / \\max|\\hat h|`
  with taps below ``relative_threshold`` set to zero (one value per tap).
* ``decay_curve``: the backward-integrated energy decay curve
  (Schroeder, 1965),

  .. math::

      E(t) = 10 \\log_{10} \\frac{\\sum_{\\tau \\ge t} \\hat h^2(\\tau)}
                                  {\\sum_{\\tau} \\hat h^2(\\tau)} ,

  floored at ``floor_db``. It is monotone and shifts smoothly when the
  echoes move, so rooms with similar sizes get similar curves even when the
  individual echoes do not line up.
* ``echo_delays``: the arrival times (ms) of the ``n_echoes`` strongest
  echoes in increasing order; a missing echo is recorded at the end of the
  response window.
* ``imf_energy``: empirical mode decomposition of the pruned magnitude
  (Huang et al., 1998): the share of energy and the zero-crossing rate of
  each of the first ``max_imfs`` intrinsic mode functions, plus the energy
  share of the residual.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .echoes import pick_echoes
from .emd import CauchyStopping, count_zero_crossings, decompose_emd
from .identification import prune_small_taps


def decay_curve_db(response: ArrayLike, floor_db: float = -60.0) -> NDArray[np.float64]:
    """Schroeder backward-integrated energy decay curve in dB, 0 dB at tap 0.

    Parameters
    ----------
    response : array_like, shape (n_taps,)
        Impulse response.
    floor_db : float, optional
        Lowest value returned (a negative number).

    Returns
    -------
    numpy.ndarray, shape (n_taps,)
        Non-increasing values in ``[floor_db, 0]``; all ``floor_db`` for a
        silent response.

    References
    ----------
    .. [1] M. R. Schroeder, "New method of measuring reverberation time," The
       Journal of the Acoustical Society of America, vol. 37, no. 3,
       pp. 409-412, 1965.

    Examples
    --------
    >>> decay_curve_db([1.0, 0.0, 1.0, 0.0]).round(3).tolist()
    [0.0, -3.01, -3.01, -60.0]
    """
    if not floor_db < 0:
        raise ValueError(f"floor_db must be negative, got {floor_db!r}")
    energy = np.asarray(response, dtype=np.float64).ravel() ** 2
    remaining = np.cumsum(energy[::-1])[::-1]
    total = float(remaining[0]) if remaining.size else 0.0
    if total == 0.0:
        return np.full(energy.size, float(floor_db))
    with np.errstate(divide="ignore"):
        curve = 10.0 * np.log10(remaining / total)
    return np.maximum(curve, float(floor_db))


def echo_delay_features(
    response: ArrayLike,
    sample_rate: int,
    *,
    n_echoes: int = 9,
    method: str = "peaks",
    min_separation_taps: int = 3,
    relative_height: float = 0.02,
    interpolate: bool = False,
    compensate_spreading: bool = False,
) -> NDArray[np.float64]:
    """Arrival times of the strongest echoes in milliseconds, in increasing order.

    Missing echoes (fewer peaks than ``n_echoes``) are placed at the end of the
    response window, so the vector always has ``n_echoes`` values. The picking
    options are those of :func:`~room_geometry_inference.echoes.pick_echoes`.

    Examples
    --------
    >>> h = np.zeros(160)
    >>> h[[16, 48]] = [1.0, 0.5]
    >>> echo_delay_features(h, 16_000, n_echoes=3).tolist()
    [1.0, 3.0, 10.0]
    """
    values = np.asarray(response, dtype=np.float64).ravel()
    taps = pick_echoes(
        values,
        n_echoes,
        method=method,
        min_separation_taps=min_separation_taps,
        relative_height=relative_height,
        interpolate=interpolate,
        compensate_spreading=compensate_spreading,
    )
    padded = np.full(int(n_echoes), float(values.size))
    padded[: taps.size] = taps
    return 1000.0 * padded / float(sample_rate)


def imf_energy_features(
    response: ArrayLike, *, max_imfs: int = 6, tolerance: float = 0.3
) -> NDArray[np.float64]:
    """Energy share and zero-crossing rate of each intrinsic mode function.

    Parameters
    ----------
    response : array_like, shape (n_taps,)
        Response (typically the pruned magnitude).
    max_imfs : int, optional
        Number of IMFs described; missing ones are zeros.
    tolerance : float, optional
        Sifting stops when the normalized squared difference of successive
        candidates falls below it (Huang et al., 1998).

    Returns
    -------
    numpy.ndarray, shape (2 * max_imfs + 1,)
        ``max_imfs`` energy shares, ``max_imfs`` zero-crossing rates (per
        sample), then the residual's energy share.

    References
    ----------
    .. [1] N. E. Huang, Z. Shen, S. R. Long, M. C. Wu, H. H. Shih, Q. Zheng,
       N.-C. Yen, C. C. Tung, and H. H. Liu, "The empirical mode decomposition
       and the Hilbert spectrum for nonlinear and non-stationary time series
       analysis," Proceedings of the Royal Society of London A, vol. 454,
       no. 1971, pp. 903-995, 1998.

    Examples
    --------
    >>> t = np.linspace(0, 1, 400)
    >>> features = imf_energy_features(np.sin(2 * np.pi * 25 * t), max_imfs=2)
    >>> features.shape, bool(features[0] > 0.9)
    ((5,), True)
    """
    values = np.asarray(response, dtype=np.float64).ravel()
    if int(max_imfs) < 1:
        raise ValueError(f"max_imfs must be at least 1, got {max_imfs!r}")
    features = np.zeros(2 * int(max_imfs) + 1)
    total = float(np.sum(values**2))
    if total == 0.0 or values.size < 4:
        return features
    parts = decompose_emd(
        values, max_modes=int(max_imfs), stopping=CauchyStopping(tolerance=float(tolerance))
    )
    n_modes = parts.n_modes
    features[:n_modes] = np.sum(parts.imfs**2, axis=1) / total
    features[max_imfs : max_imfs + n_modes] = [
        count_zero_crossings(mode) / values.size for mode in parts.imfs
    ]
    features[-1] = float(np.sum(parts.residual**2)) / total
    return features


def represent_response(
    response: ArrayLike,
    representation: str = "magnitude",
    *,
    sample_rate: int = 16_000,
    relative_threshold: float = 1e-3,
    floor_db: float = -60.0,
    echo_options: dict[str, Any] | None = None,
    max_imfs: int = 6,
    emd_tolerance: float = 0.3,
) -> NDArray[np.float64]:
    """One feature vector of the chosen representation.

    Parameters
    ----------
    response : array_like, shape (n_taps,)
        Identified impulse response.
    representation : {"magnitude", "decay_curve", "echo_delays", "imf_energy"}, optional
        See the module documentation.
    sample_rate : int, optional
        Sampling rate in Hz (``echo_delays``).
    relative_threshold : float, optional
        Pruning threshold of the magnitude (``magnitude``, ``imf_energy``).
    floor_db : float, optional
        Floor of the decay curve.
    echo_options : dict, optional
        Keyword arguments of :func:`echo_delay_features`.
    max_imfs, emd_tolerance : int, float, optional
        Settings of :func:`imf_energy_features`.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> h = np.zeros(50)
    >>> h[[5, 20]] = [1.0, -0.5]
    >>> [represent_response(h, name).shape for name in REPRESENTATIONS]
    [(50,), (50,), (9,), (13,)]
    """
    if representation not in _REPRESENTERS:
        raise ValueError(
            f"representation={representation!r} is not supported; "
            f"choose one of {list(REPRESENTATIONS)}"
        )
    settings = {
        "sample_rate": sample_rate,
        "relative_threshold": relative_threshold,
        "floor_db": floor_db,
        "echo_options": echo_options or {},
        "max_imfs": max_imfs,
        "emd_tolerance": emd_tolerance,
    }
    return _REPRESENTERS[representation](response, settings)


def _magnitude(response: ArrayLike, settings: Mapping[str, Any]) -> NDArray[np.float64]:
    return prune_small_taps(response, settings["relative_threshold"])


def _decay_curve(response: ArrayLike, settings: Mapping[str, Any]) -> NDArray[np.float64]:
    return decay_curve_db(response, settings["floor_db"])


def _echo_delays(response: ArrayLike, settings: Mapping[str, Any]) -> NDArray[np.float64]:
    return echo_delay_features(response, settings["sample_rate"], **settings["echo_options"])


def _imf_energy(response: ArrayLike, settings: Mapping[str, Any]) -> NDArray[np.float64]:
    return imf_energy_features(
        prune_small_taps(response, settings["relative_threshold"]),
        max_imfs=settings["max_imfs"],
        tolerance=settings["emd_tolerance"],
    )


_REPRESENTERS: dict[str, Callable[[ArrayLike, Mapping[str, Any]], NDArray[np.float64]]] = {
    "magnitude": _magnitude,
    "decay_curve": _decay_curve,
    "echo_delays": _echo_delays,
    "imf_energy": _imf_energy,
}
#: Feature representations accepted by ``features.representation``.
REPRESENTATIONS: tuple[str, ...] = tuple(_REPRESENTERS)
