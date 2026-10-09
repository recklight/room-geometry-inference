"""Impulse-response identification from an excitation and its recording.

The loudspeaker plays a known excitation :math:`x[n]` and the microphone
records :math:`d[n] = (h * x)[n] + v[n]`. An adaptive FIR filter
:math:`\\mathbf{w}` of ``n_taps`` coefficients is driven by the regressor
:math:`\\mathbf{x}_n = (x[n], x[n-1], \\dots, x[n-N+1])` and converges to
:math:`h` (system identification; Haykin, 2014):

* NLMS (Nagumo & Noda, 1967):

  .. math::

      e[n] = d[n] - \\mathbf{w}_n^\\top \\mathbf{x}_n, \\qquad
      \\mathbf{w}_{n+1} = \\mathbf{w}_n
      + \\frac{\\mu\\, e[n]}{\\varepsilon + \\|\\mathbf{x}_n\\|^2}\\, \\mathbf{x}_n .

  The normalization makes the step independent of the excitation level;
  :math:`0 < \\mu < 2` is stable and :math:`\\mu = 1` is the fastest.
* LMS (Widrow & Hoff, 1960): the same update without normalization,
  stable only for :math:`\\mu < 2 / (N \\sigma_x^2)`, so its step size
  depends on the signal level.
* Least squares: the converged answer in one step, the Wiener solution
  :math:`R_{xx} \\mathbf{w} = \\mathbf{r}_{xd}` with the autocorrelation and
  cross-correlation estimated over the whole recording and the Toeplitz
  system solved by Levinson recursion. The adaptive filters approach this
  answer as they converge.

Only the final weights are kept, plus optional snapshots: the full history of a
2 s recording at 16 kHz with 1600 taps would be 51.2 million float64 values
(about 410 MB). The learning curve is the squared error averaged over a
sliding window, in dB; the normalized misalignment
:math:`20 \\log_{10} (\\|h - \\mathbf{w}\\| / \\|h\\|)` measures how close the
filter is to the true response when that is known.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class AdaptiveResult:
    """Outcome of an adaptive filter run.

    Attributes
    ----------
    weights : numpy.ndarray, shape (n_taps,)
        Final filter, tap 0 first: the identified impulse response.
    error : numpy.ndarray, shape (n_samples,)
        A-priori error :math:`e[n]` at every sample.
    snapshots : numpy.ndarray, shape (n_snapshots, n_taps)
        Weights every ``snapshot_every`` samples (empty when not requested).

    Examples
    --------
    >>> x = np.random.default_rng(0).standard_normal(500)
    >>> result = nlms(x, x, n_taps=4, snapshot_every=100)
    >>> result.weights.shape, result.error.shape, result.snapshots.shape
    ((4,), (500,), (5, 4))
    """

    weights: NDArray[np.float64]
    error: NDArray[np.float64]
    snapshots: NDArray[np.float64]


def _signals(
    excitation: ArrayLike, recording: ArrayLike, n_taps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(excitation, dtype=np.float64).ravel()
    d = np.asarray(recording, dtype=np.float64).ravel()
    if x.size != d.size:
        raise ValueError(
            f"excitation and recording must have the same length, got {x.size} and {d.size}"
        )
    if int(n_taps) < 1:
        raise ValueError(f"n_taps must be at least 1, got {n_taps!r}")
    if x.size <= int(n_taps):
        raise ValueError(f"the signals ({x.size} samples) must be longer than n_taps ({n_taps})")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(d))):
        raise ValueError("excitation and recording must be finite")
    if not np.any(x):
        raise ValueError("the excitation is silent")
    return x, d


def _adapt(
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    n_taps: int,
    step: float,
    regularization: float,
    normalized: bool,
    snapshot_every: int | None,
) -> AdaptiveResult:
    """Sample-by-sample (N)LMS. The weights are kept in reverse order, so the
    regressor is a plain slice of the zero-padded excitation."""
    padded = np.concatenate([np.zeros(n_taps - 1), x])
    reversed_weights = np.zeros(n_taps)
    error = np.empty(x.size)
    snapshots = []
    dot = np.dot
    for n in range(x.size):
        regressor = padded[n : n + n_taps]
        e = d[n] - dot(reversed_weights, regressor)
        error[n] = e
        gain = step * e / (regularization + dot(regressor, regressor)) if normalized else step * e
        reversed_weights += gain * regressor
        if snapshot_every and (n + 1) % snapshot_every == 0:
            snapshots.append(reversed_weights[::-1].copy())
    if not np.all(np.isfinite(reversed_weights)):
        raise ValueError("the adaptive filter diverged; lower the step size")
    stacked = np.asarray(snapshots) if snapshots else np.zeros((0, n_taps))
    return AdaptiveResult(weights=reversed_weights[::-1].copy(), error=error, snapshots=stacked)


def nlms(
    excitation: ArrayLike,
    recording: ArrayLike,
    n_taps: int = 1000,
    step: float = 0.8,
    regularization: float = 1e-10,
    *,
    snapshot_every: int | None = None,
) -> AdaptiveResult:
    """Normalized least-mean-squares identification of an FIR response.

    Parameters
    ----------
    excitation : array_like, shape (n_samples,)
        Signal sent to the loudspeaker.
    recording : array_like, shape (n_samples,)
        Microphone signal, sample-aligned with ``excitation``.
    n_taps : int, optional
        Filter length.
    step : float, optional
        Step size :math:`\\mu`, in ``(0, 2)``.
    regularization : float, optional
        :math:`\\varepsilon`, guards the division during silence.
    snapshot_every : int, optional
        Keep a copy of the weights every this many samples.

    Returns
    -------
    AdaptiveResult

    Raises
    ------
    ValueError
        For signals of different lengths, too short, silent or not finite, a
        step outside ``(0, 2)``, or a diverging filter.

    References
    ----------
    .. [1] J. Nagumo and A. Noda, "A learning method for system
       identification," IEEE Transactions on Automatic Control, vol. 12,
       no. 3, pp. 282-287, 1967.
    .. [2] S. Haykin, Adaptive Filter Theory, 5th ed. Upper Saddle River, NJ,
       USA: Pearson, 2014.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(4000)
    >>> h = np.array([0.0, 1.0, 0.0, -0.5, 0.25])
    >>> result = nlms(x, np.convolve(x, h)[:4000], n_taps=8)
    >>> (result.weights[:5].round(6) + 0.0).tolist(), result.snapshots.shape
    ([0.0, 1.0, 0.0, -0.5, 0.25], (0, 8))
    """
    x, d = _signals(excitation, recording, n_taps)
    if not 0 < step < 2:
        raise ValueError(f"the NLMS step must lie in (0, 2), got {step!r}")
    if regularization < 0:
        raise ValueError(f"regularization must not be negative, got {regularization!r}")
    return _adapt(x, d, int(n_taps), float(step), float(regularization), True, snapshot_every)


def lms(
    excitation: ArrayLike,
    recording: ArrayLike,
    n_taps: int = 1000,
    step: float = 2e-4,
    *,
    snapshot_every: int | None = None,
) -> AdaptiveResult:
    """Least-mean-squares identification of an FIR response.

    Parameters
    ----------
    excitation, recording : array_like, shape (n_samples,)
        Loudspeaker and microphone signals, sample-aligned.
    n_taps : int, optional
        Filter length.
    step : float, optional
        Step size :math:`\\mu`; it must stay below
        :math:`2 / (N \\sigma_x^2)` (checked, with a warning in the log when
        it is close).
    snapshot_every : int, optional
        Keep a copy of the weights every this many samples.

    Returns
    -------
    AdaptiveResult

    References
    ----------
    .. [1] B. Widrow and M. E. Hoff, "Adaptive switching circuits," in IRE
       WESCON Convention Record, Part 4, 1960, pp. 96-104.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> x = rng.standard_normal(6000)
    >>> result = lms(x, np.convolve(x, [0.5, 0.3])[:6000], n_taps=4, step=0.02)
    >>> (result.weights.round(3) + 0.0).tolist()
    [0.5, 0.3, 0.0, 0.0]
    """
    x, d = _signals(excitation, recording, n_taps)
    if not step > 0:
        raise ValueError(f"the LMS step must be positive, got {step!r}")
    bound = 2.0 / (int(n_taps) * float(np.mean(x**2)))
    if step >= bound:
        raise ValueError(
            f"the LMS step {step!r} exceeds the stability bound 2 / (n_taps * power) = {bound:.3g}"
        )
    if step > 0.5 * bound:
        _LOGGER.warning("the LMS step is above half the stability bound (%.3g)", bound)
    return _adapt(x, d, int(n_taps), float(step), 0.0, False, snapshot_every)


def least_squares_response(
    excitation: ArrayLike,
    recording: ArrayLike,
    n_taps: int = 1000,
    regularization: float = 1e-10,
) -> NDArray[np.float64]:
    """Wiener (least-squares) FIR estimate, solved by Levinson recursion.

    Parameters
    ----------
    excitation, recording : array_like, shape (n_samples,)
        Loudspeaker and microphone signals, sample-aligned.
    n_taps : int, optional
        Filter length.
    regularization : float, optional
        Added to the zero-lag autocorrelation, relative to it.

    Returns
    -------
    numpy.ndarray, shape (n_taps,)

    Notes
    -----
    The correlations are estimated over the whole zero-padded recording (the
    autocorrelation method), which keeps the normal equations Toeplitz; the
    edges bias the estimate slightly, by an amount that shrinks as the
    recording grows relative to ``n_taps``.

    Examples
    --------
    >>> rng = np.random.default_rng(2)
    >>> x = rng.standard_normal(3000)
    >>> h = np.array([0.2, 0.0, 1.0, 0.4])
    >>> estimate = least_squares_response(x, np.convolve(x, h)[:3000], n_taps=6)
    >>> (estimate.round(2) + 0.0).tolist()
    [0.2, 0.0, 1.0, 0.4, 0.0, 0.0]
    """
    x, d = _signals(excitation, recording, n_taps)
    n = int(n_taps)
    autocorrelation = fftconvolve(x, x[::-1])[x.size - 1 : x.size - 1 + n]
    cross = fftconvolve(d, x[::-1])[x.size - 1 : x.size - 1 + n]
    autocorrelation[0] *= 1.0 + float(regularization)
    return np.asarray(solve_toeplitz(autocorrelation, cross), dtype=np.float64)


def learning_curve_db(error: ArrayLike, window: int = 80) -> NDArray[np.float64]:
    """Squared error averaged over the last ``window`` samples, in dB.

    The average is causal; the first ``window - 1`` samples average over
    the samples available so far.

    Examples
    --------
    >>> learning_curve_db(np.full(5, 0.1), window=2).round(6).tolist()
    [-20.0, -20.0, -20.0, -20.0, -20.0]
    """
    squared = np.asarray(error, dtype=np.float64).ravel() ** 2
    if int(window) < 1:
        raise ValueError(f"window must be at least 1, got {window!r}")
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    index = np.arange(1, squared.size + 1)
    start = np.maximum(index - int(window), 0)
    mean = (cumulative[index] - cumulative[start]) / (index - start)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(mean, np.finfo(np.float64).tiny))


def normalized_misalignment_db(reference: ArrayLike, estimate: ArrayLike) -> float:
    """:math:`20 \\log_{10}(\\|h - \\hat h\\| / \\|h\\|)`; the shorter vector is zero-padded.

    Examples
    --------
    >>> round(normalized_misalignment_db([1.0, 0.5], [1.0, 0.5, 0.0]), 6)
    -inf
    >>> round(normalized_misalignment_db([1.0, 0.0], [0.9, 0.0]), 3)
    -20.0
    """
    h = np.asarray(reference, dtype=np.float64).ravel()
    w = np.asarray(estimate, dtype=np.float64).ravel()
    length = max(h.size, w.size)
    h = np.pad(h, (0, length - h.size))
    w = np.pad(w, (0, length - w.size))
    norm = float(np.linalg.norm(h))
    if norm == 0.0:
        raise ValueError("the reference response is zero")
    gap = float(np.linalg.norm(h - w))
    return float("-inf") if gap == 0.0 else 20.0 * float(np.log10(gap / norm))


def prune_small_taps(response: ArrayLike, relative_threshold: float = 1e-3) -> NDArray[np.float64]:
    """Peak-normalized magnitude with taps below a threshold set to zero.

    Parameters
    ----------
    response : array_like, shape (n_taps,)
        Identified response.
    relative_threshold : float, optional
        Taps whose magnitude is below this fraction of the peak become zero;
        0 keeps every tap.

    Returns
    -------
    numpy.ndarray, shape (n_taps,)
        Values in ``[0, 1]``; all zeros for a silent response.

    Examples
    --------
    >>> prune_small_taps([0.0005, -2.0, 0.5, 0.001], 0.01).tolist()
    [0.0, 1.0, 0.25, 0.0]
    """
    magnitude = np.abs(np.asarray(response, dtype=np.float64).ravel())
    if not 0 <= relative_threshold < 1:
        raise ValueError(f"relative_threshold must lie in [0, 1), got {relative_threshold!r}")
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak == 0.0:
        return np.zeros_like(magnitude)
    normalized = magnitude / peak
    normalized[normalized < relative_threshold] = 0.0
    return normalized


def identify_response(
    excitation: ArrayLike,
    recording: ArrayLike,
    *,
    method: str = "nlms",
    n_taps: int = 1000,
    step: float = 0.8,
    regularization: float = 1e-10,
) -> NDArray[np.float64]:
    """Identified impulse response with the chosen method (final weights only).

    Examples
    --------
    >>> rng = np.random.default_rng(3)
    >>> x = rng.standard_normal(2000)
    >>> d = np.convolve(x, [0.0, 0.0, 1.0])[:2000]
    >>> [int(np.argmax(np.abs(identify_response(x, d, method=m, n_taps=6, step=s))))
    ...  for m, s in (("nlms", 0.8), ("lms", 0.01), ("least_squares", 0.8))]
    [2, 2, 2]
    """
    if method not in _IDENTIFIERS:
        raise ValueError(
            f"method={method!r} is not supported; choose one of {list(IDENTIFICATION_METHODS)}"
        )
    return _IDENTIFIERS[method](excitation, recording, n_taps, step, regularization)


def _identify_nlms(
    x: ArrayLike, d: ArrayLike, n_taps: int, step: float, regularization: float
) -> NDArray[np.float64]:
    return nlms(x, d, n_taps, step, regularization).weights


def _identify_lms(
    x: ArrayLike, d: ArrayLike, n_taps: int, step: float, regularization: float
) -> NDArray[np.float64]:
    return lms(x, d, n_taps, step).weights


def _identify_least_squares(
    x: ArrayLike, d: ArrayLike, n_taps: int, step: float, regularization: float
) -> NDArray[np.float64]:
    return least_squares_response(x, d, n_taps, regularization)


_IDENTIFIERS: dict[
    str, Callable[[ArrayLike, ArrayLike, int, float, float], NDArray[np.float64]]
] = {
    "nlms": _identify_nlms,
    "lms": _identify_lms,
    "least_squares": _identify_least_squares,
}
#: Identification methods accepted by ``identification.method``.
IDENTIFICATION_METHODS: tuple[str, ...] = tuple(_IDENTIFIERS)
