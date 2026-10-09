"""Empirical mode decomposition (EMD): sifting, stopping rules and the decomposition.

EMD writes a signal as a sum of intrinsic mode functions (IMFs) and a
residual trend (Huang et al., 1998):

.. math::

    x(t) = \\sum_{k=1}^{K} c_k(t) + r(t).

One sifting step estimates the local mean of a signal and subtracts it. The
local mean is the average of two cubic-spline envelopes, one through the local
maxima and one through the local minima:

.. math::

    m(t) = \\tfrac12\\big(e_{\\max}(t) + e_{\\min}(t)\\big), \\qquad h(t) = x(t) - m(t).

The splines need support beyond the first and last extremum or they swing
wildly at the ends of the record. Following Rilling et al. (2003), a few
extrema near each end are mirrored about the end extremum (or about the end
sample itself when the signal ends beyond its last extremum), so the
envelopes are interpolated, not extrapolated, over the whole record.

Sifting refines an IMF candidate until a stopping criterion accepts it:

``CauchyStopping``
    A Cauchy-type test after Huang et al. (1998): the normalized squared
    difference between successive candidates, :math:`\\mathrm{SD} = \\sum
    (h_{k-1}-h_k)^2 / \\sum h_{k-1}^2`, falls below a tolerance (0.2-0.3 is
    customary), and the numbers of extrema and zero crossings differ by at
    most one. Huang et al. divide every sample by :math:`h_{k-1}^2(t)` before
    summing; dividing by the total energy keeps SD finite where the candidate
    crosses zero.
``ThreeThresholdStopping``
    Rilling et al. (2003): with :math:`\\sigma(t) = |m(t)| / a(t)` (envelope
    mean over envelope amplitude), stop when :math:`\\sigma < \\theta_1` on
    all but a fraction :math:`\\alpha` of the record and
    :math:`\\sigma < \\theta_2` everywhere.
``FixedIterationStopping``
    A fixed number of sifting steps (Wu & Huang, 2009).

Whatever the criterion, the number of sifting steps per IMF is capped
(``max_sift_iterations``) and so is the number of IMFs (``max_modes``), so a
pathological signal can never loop forever. Each IMF is subtracted and the
process repeats on the remainder until the remainder is a trend (fewer than
three extrema) or ``max_modes`` IMFs have been taken. The decomposition is
complete by construction, since the residual is whatever the IMFs leave
over; :func:`decompose_emd` still checks the sum and raises if it is off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.linalg import solve_banded

_LOGGER = logging.getLogger(__name__)

#: Number of extrema mirrored beyond each end of the record.
DEFAULT_MIRRORED_EXTREMA: int = 2


@dataclass(frozen=True, eq=False)
class Extrema:
    """Indices of the local maxima and minima of a signal.

    Plateaus count as a single extremum located at their center.

    Attributes
    ----------
    maxima, minima : numpy.ndarray of int64
        Sorted sample indices.

    Examples
    --------
    >>> Extrema(np.array([1, 5]), np.array([3])).count
    3
    """

    maxima: np.ndarray
    minima: np.ndarray

    @property
    def count(self) -> int:
        """Total number of extrema."""
        return int(self.maxima.size + self.minima.size)


def find_extrema(signal: np.ndarray) -> Extrema:
    """Locate local maxima and minima, treating plateaus as one extremum.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional real signal.

    Returns
    -------
    Extrema
        Sorted index arrays. End samples are never extrema.

    Examples
    --------
    >>> e = find_extrema(np.array([0.0, 2.0, 1.0, 1.0, 1.0, 3.0, 0.0]))
    >>> e.maxima.tolist(), e.minima.tolist()
    ([1, 5], [3])
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got shape {x.shape}")
    empty = np.zeros(0, dtype=np.int64)
    if x.size < 3:
        return Extrema(empty, empty)
    # Collapse every run of equal values to one point, located at the run's center.
    run_start = np.flatnonzero(np.r_[True, np.diff(x) != 0])
    run_end = np.r_[run_start[1:] - 1, x.size - 1]
    center = (run_start + run_end) // 2
    values = x[run_start]
    if values.size < 3:
        return Extrema(empty, empty)
    step = np.diff(values)
    rising = step[:-1] > 0
    falling = step[1:] < 0
    maxima = center[1:-1][rising & falling]
    minima = center[1:-1][~rising & ~falling]
    return Extrema(maxima.astype(np.int64), minima.astype(np.int64))


def count_zero_crossings(signal: np.ndarray) -> int:
    """Number of sign changes, ignoring exact zeros.

    Examples
    --------
    >>> count_zero_crossings(np.array([1.0, -1.0, 0.0, -2.0, 3.0]))
    2
    """
    sign = np.sign(np.asarray(signal, dtype=np.float64))
    sign = sign[sign != 0]
    return int(np.count_nonzero(sign[1:] != sign[:-1]))


def _mirror_start(
    t: np.ndarray, x: np.ndarray, maxima: np.ndarray, minima: np.ndarray, n_mirror: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirrored extrema before the start of the record: ``(t_max, x_max, t_min, x_min)``."""
    if maxima[0] < minima[0]:
        if x[0] > x[minima[0]]:
            sel_max, sel_min, axis = maxima[1 : n_mirror + 1], minima[:n_mirror], maxima[0]
        else:
            sel_max, sel_min, axis = maxima[:n_mirror], np.r_[0, minima[: n_mirror - 1]], 0
    else:
        if x[0] < x[maxima[0]]:
            sel_max, sel_min, axis = maxima[:n_mirror], minima[1 : n_mirror + 1], minima[0]
        else:
            sel_max, sel_min, axis = np.r_[0, maxima[: n_mirror - 1]], minima[:n_mirror], 0

    t_max = 2.0 * t[axis] - t[sel_max]
    t_min = 2.0 * t[axis] - t[sel_min]
    covered = t_max.size > 0 and t_min.size > 0 and t_max.min() <= t[0] and t_min.min() <= t[0]
    if not covered and axis != 0:
        # Mirroring about the end extremum does not reach past the record start:
        # mirror about the first sample instead.
        if axis == maxima[0]:
            sel_max = maxima[:n_mirror]
        else:
            sel_min = minima[:n_mirror]
        t_max = 2.0 * t[0] - t[sel_max]
        t_min = 2.0 * t[0] - t[sel_min]
    return t_max[::-1], x[sel_max][::-1], t_min[::-1], x[sel_min][::-1]


def _envelope_knots(
    t: np.ndarray, x: np.ndarray, extrema: Extrema, n_mirror: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Knot times and values of the upper and lower envelopes, mirrors included."""
    n = x.size
    left = _mirror_start(t, x, extrema.maxima, extrema.minima, n_mirror)
    # The end of the record is the start of the time-reversed record.
    t_rev = -t[::-1]
    right = _mirror_start(
        t_rev,
        x[::-1],
        (n - 1 - extrema.maxima)[::-1],
        (n - 1 - extrema.minima)[::-1],
        n_mirror,
    )
    t_max = np.concatenate([left[0], t[extrema.maxima], -right[0][::-1]])
    x_max = np.concatenate([left[1], x[extrema.maxima], right[1][::-1]])
    t_min = np.concatenate([left[2], t[extrema.minima], -right[2][::-1]])
    x_min = np.concatenate([left[3], x[extrema.minima], right[3][::-1]])
    return (*_strictly_increasing(t_max, x_max), *_strictly_increasing(t_min, x_min))


def _strictly_increasing(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times.size > 1 and np.all(np.diff(times) > 0):
        return times, values
    unique_t, first = np.unique(times, return_index=True)
    return unique_t, values[first]


def natural_cubic_spline(knots: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluate the natural cubic spline through ``(knots, values)`` at ``query``.

    The sifting loop builds two splines per step, for every IMF of every
    response of a dataset, so this is a lean direct implementation: one
    tridiagonal solve for the knot second derivatives :math:`M_i` (zero at both
    ends), then a vectorized evaluation of

    .. math::

        S(t) = M_i \\frac{(t_{i+1}-t)^3}{6h_i} + M_{i+1}\\frac{(t-t_i)^3}{6h_i}
             + \\Big(y_i - \\frac{M_i h_i^2}{6}\\Big)\\frac{t_{i+1}-t}{h_i}
             + \\Big(y_{i+1} - \\frac{M_{i+1} h_i^2}{6}\\Big)\\frac{t-t_i}{h_i}.

    The end conditions only act beyond the mirrored extrema, i.e. outside the
    record, so the natural condition is as good as any there.

    Parameters
    ----------
    knots : numpy.ndarray
        Strictly increasing knot positions (at least two).
    values : numpy.ndarray
        Values at the knots.
    query : numpy.ndarray
        Evaluation points.

    Returns
    -------
    numpy.ndarray
        Spline values at ``query``.

    Examples
    --------
    >>> k = np.linspace(0, 2 * np.pi, 40)
    >>> q = np.linspace(0.5, 5.5, 7)
    >>> bool(np.allclose(natural_cubic_spline(k, np.sin(k), q), np.sin(q), atol=1e-4))
    True
    """
    t = np.asarray(knots, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    m = t.size
    h = np.diff(t)
    second = np.zeros(m)
    if m > 2:
        slope = np.diff(y) / h
        rhs = 6.0 * np.diff(slope)
        bands = np.zeros((3, m - 2))
        bands[0, 1:] = h[1:-1]
        bands[1] = 2.0 * (h[:-1] + h[1:])
        bands[2, :-1] = h[1:-1]
        second[1:-1] = solve_banded((1, 1), bands, rhs, check_finite=False)
    points = np.asarray(query, dtype=np.float64)
    i = np.clip(np.searchsorted(t, points, side="right") - 1, 0, m - 2)
    width = h[i]
    left = t[i + 1] - points
    right = points - t[i]
    return (
        second[i] * left**3 / (6.0 * width)
        + second[i + 1] * right**3 / (6.0 * width)
        + (y[i] - second[i] * width**2 / 6.0) * left / width
        + (y[i + 1] - second[i + 1] * width**2 / 6.0) * right / width
    )


@dataclass(frozen=True, eq=False)
class EnvelopePair:
    """Upper and lower envelopes of a signal.

    Attributes
    ----------
    upper, lower : numpy.ndarray
        Envelope values at every sample.

    Examples
    --------
    >>> pair = EnvelopePair(np.array([2.0, 3.0]), np.array([0.0, -1.0]))
    >>> pair.mean.tolist(), pair.amplitude.tolist()
    ([1.0, 1.0], [1.0, 2.0])
    """

    upper: np.ndarray
    lower: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        """Local mean, the average of the two envelopes."""
        return 0.5 * (self.upper + self.lower)

    @property
    def amplitude(self) -> np.ndarray:
        """Local amplitude, half the distance between the envelopes."""
        return 0.5 * (self.upper - self.lower)


def extrema_envelopes(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
    extrema: Extrema | None = None,
) -> EnvelopePair | None:
    """Cubic-spline envelopes through the maxima and the minima.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional real signal.
    time : numpy.ndarray, optional
        Sample times (strictly increasing); defaults to sample indices.
    n_mirror : int, optional
        Extrema mirrored beyond each end.
    extrema : Extrema, optional
        Precomputed extrema of ``signal``.

    Returns
    -------
    EnvelopePair or None
        ``None`` when the signal has fewer than three extrema in total: it is
        then a trend, not an oscillation, and has no envelopes.

    Examples
    --------
    >>> t = np.linspace(0, 1, 500)
    >>> pair = extrema_envelopes(np.sin(2 * np.pi * 10 * t))
    >>> bool(np.allclose(pair.upper[50:-50], 1.0, atol=0.02)), extrema_envelopes(t) is None
    (True, True)
    """
    x = np.asarray(signal, dtype=np.float64)
    t = np.arange(x.size, dtype=np.float64) if time is None else np.asarray(time, np.float64)
    found = find_extrema(x) if extrema is None else extrema
    if found.count < 3 or found.maxima.size == 0 or found.minima.size == 0:
        return None
    t_max, x_max, t_min, x_min = _envelope_knots(t, x, found, max(1, int(n_mirror)))
    if t_max.size < 2 or t_min.size < 2:
        return None
    upper = natural_cubic_spline(t_max, x_max, t)
    lower = natural_cubic_spline(t_min, x_min, t)
    return EnvelopePair(upper=upper, lower=lower)


@dataclass(frozen=True, eq=False)
class SiftStep:
    """Outcome of one sifting step.

    Attributes
    ----------
    candidate : numpy.ndarray
        ``signal - mean``: the refined IMF candidate.
    envelopes : EnvelopePair
        The envelopes of the input.
    extrema : Extrema
        The extrema of the input.
    n_zero_crossings : int
        Zero crossings of the input.

    Examples
    --------
    >>> step = sift_once(np.sin(np.linspace(0, 12 * np.pi, 300)) + 1.0)
    >>> step.extrema.count, step.n_zero_crossings
    (12, 0)
    """

    candidate: np.ndarray
    envelopes: EnvelopePair
    extrema: Extrema
    n_zero_crossings: int


def sift_once(
    signal: np.ndarray,
    time: np.ndarray | None = None,
    *,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> SiftStep | None:
    """Subtract the envelope mean once.

    Returns
    -------
    SiftStep or None
        ``None`` when the signal has too few extrema to be sifted.

    Examples
    --------
    Sifting removes a slow offset riding under a fast oscillation:

    >>> t = np.linspace(0, 1, 400)
    >>> x = np.sin(2 * np.pi * 20 * t) + 0.5
    >>> step = sift_once(x)
    >>> bool(abs(step.candidate[50:-50].mean()) < 0.05)
    True
    """
    x = np.asarray(signal, dtype=np.float64)
    extrema = find_extrema(x)
    envelopes = extrema_envelopes(x, time, n_mirror=n_mirror, extrema=extrema)
    if envelopes is None:
        return None
    return SiftStep(
        candidate=x - envelopes.mean,
        envelopes=envelopes,
        extrema=extrema,
        n_zero_crossings=count_zero_crossings(x),
    )


#: Names accepted by :func:`stopping_from_name`.
STOPPING_CRITERIA: tuple[str, ...] = ("three_threshold", "cauchy", "fixed")


@dataclass(frozen=True, eq=False)
class SiftState:
    """Everything a stopping criterion may look at.

    Attributes
    ----------
    candidate : numpy.ndarray
        Current IMF candidate :math:`h_k`.
    previous : numpy.ndarray or None
        Previous candidate :math:`h_{k-1}`, ``None`` on the first step.
    mean_envelope : numpy.ndarray
        Envelope mean of ``candidate``.
    amplitude : numpy.ndarray
        Half the envelope gap of ``candidate``.
    iteration : int
        Sifting steps already applied to obtain ``candidate``.
    n_extrema : int
        Extrema of ``candidate``.
    n_zero_crossings : int
        Zero crossings of ``candidate``.

    Examples
    --------
    >>> h = np.zeros(4)
    >>> SiftState(h, None, h, h, 0, 5, 4).satisfies_imf_condition
    True
    """

    candidate: np.ndarray
    previous: np.ndarray | None
    mean_envelope: np.ndarray
    amplitude: np.ndarray
    iteration: int
    n_extrema: int
    n_zero_crossings: int

    @property
    def satisfies_imf_condition(self) -> bool:
        """Extrema and zero crossings differ by at most one."""
        return abs(self.n_extrema - self.n_zero_crossings) <= 1


@runtime_checkable
class StoppingCriterion(Protocol):
    """Interface of a sifting stopping criterion.

    Examples
    --------
    >>> isinstance(CauchyStopping(), StoppingCriterion)
    True
    """

    def should_stop(self, state: SiftState) -> bool:
        """Return True when ``state.candidate`` is accepted as an IMF."""
        ...


@dataclass(frozen=True)
class ThreeThresholdStopping:
    """Three-threshold criterion of Rilling, Flandrin and Goncalves (2003).

    Parameters
    ----------
    theta1 : float
        Threshold that must hold on all but a fraction ``alpha`` of samples.
    theta2 : float
        Threshold that must hold everywhere (conventionally ``10 * theta1``).
    alpha : float
        Tolerated fraction of samples above ``theta1``.

    References
    ----------
    .. [1] G. Rilling, P. Flandrin, and P. Goncalves, "On empirical mode
       decomposition and its algorithms," in Proc. IEEE-EURASIP Workshop on
       Nonlinear Signal and Image Processing (NSIP), 2003.

    Examples
    --------
    >>> t = np.linspace(0, 1, 200)
    >>> h = np.sin(2 * np.pi * 8 * t)
    >>> state = SiftState(h, None, np.zeros_like(h), np.ones_like(h), 0, 16, 15)
    >>> ThreeThresholdStopping().should_stop(state)
    True
    """

    theta1: float = 0.05
    theta2: float = 0.5
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.theta1 < self.theta2:
            raise ValueError(f"need 0 < theta1 < theta2, got {self.theta1}, {self.theta2}")
        if not 0 <= self.alpha < 1:
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")

    def should_stop(self, state: SiftState) -> bool:
        """Accept the candidate when the envelope mean is small relative to the amplitude."""
        if not state.satisfies_imf_condition:
            return False
        amplitude = np.abs(state.amplitude)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(amplitude > 0, np.abs(state.mean_envelope) / amplitude, np.inf)
        return bool(np.mean(sigma > self.theta1) < self.alpha and np.all(sigma < self.theta2))


@dataclass(frozen=True)
class CauchyStopping:
    """Cauchy-type convergence criterion on successive candidates (Huang et al., 1998).

    Parameters
    ----------
    tolerance : float
        Threshold on :math:`\\sum (h_{k-1}-h_k)^2 / \\sum h_{k-1}^2`, the
        total-energy normalization (Huang et al. normalize sample by sample).
    require_imf_condition : bool
        Also require the extrema / zero-crossing condition.

    References
    ----------
    .. [1] N. E. Huang, Z. Shen, S. R. Long, M. C. Wu, H. H. Shih, Q. Zheng,
       N.-C. Yen, C. C. Tung, and H. H. Liu, "The empirical mode decomposition
       and the Hilbert spectrum for nonlinear and non-stationary time series
       analysis," Proceedings of the Royal Society of London A, vol. 454,
       no. 1971, pp. 903-995, 1998.

    Examples
    --------
    >>> h = np.sin(np.linspace(0, 6, 50))
    >>> state = SiftState(h, 1.01 * h, h * 0, h * 0 + 1, 3, 2, 2)
    >>> CauchyStopping(tolerance=0.3).should_stop(state)
    True
    """

    tolerance: float = 0.2
    require_imf_condition: bool = True

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")

    def should_stop(self, state: SiftState) -> bool:
        """Accept the candidate once successive candidates stop changing."""
        if state.previous is None:
            return False
        if self.require_imf_condition and not state.satisfies_imf_condition:
            return False
        energy = float(np.sum(state.previous**2))
        if energy == 0.0:
            return True
        change = float(np.sum((state.previous - state.candidate) ** 2))
        return change / energy < self.tolerance


@dataclass(frozen=True)
class FixedIterationStopping:
    """Stop after a fixed number of sifting steps (Wu & Huang, 2009).

    Parameters
    ----------
    n_iterations : int
        Sifting steps per IMF.

    References
    ----------
    .. [1] Z. Wu and N. E. Huang, "Ensemble empirical mode decomposition: A
       noise-assisted data analysis method," Advances in Adaptive Data
       Analysis, vol. 1, no. 1, pp. 1-41, 2009.

    Examples
    --------
    >>> h = np.zeros(3)
    >>> FixedIterationStopping(2).should_stop(SiftState(h, h, h, h, 2, 0, 0))
    True
    """

    n_iterations: int = 10

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be at least 1, got {self.n_iterations}")

    def should_stop(self, state: SiftState) -> bool:
        """Accept the candidate once the iteration budget is spent."""
        return state.iteration >= self.n_iterations


def stopping_from_name(name: str, **parameters: float) -> StoppingCriterion:
    """Build a criterion from its configuration name.

    Parameters
    ----------
    name : {"three_threshold", "cauchy", "fixed"}
        Criterion name.
    **parameters
        Keyword arguments of the criterion's constructor.

    Returns
    -------
    StoppingCriterion
        The configured criterion.

    Examples
    --------
    >>> stopping_from_name("fixed", n_iterations=8)
    FixedIterationStopping(n_iterations=8)
    """
    if name == "three_threshold":
        return ThreeThresholdStopping(**parameters)
    if name == "cauchy":
        return CauchyStopping(**parameters)  # type: ignore[arg-type]
    if name == "fixed":
        return FixedIterationStopping(n_iterations=int(parameters.get("n_iterations", 10)))
    raise ValueError(f"unknown stopping criterion {name!r}; choose one of {STOPPING_CRITERIA}")


#: Hard cap on sifting steps per IMF, whatever the criterion says.
DEFAULT_MAX_SIFT_ITERATIONS: int = 1000


def default_max_modes(n_samples: int) -> int:
    """A practical bound on the number of IMFs, ``floor(log2(n)) - 1``.

    Each IMF roughly halves the dominant frequency, so a record of ``n``
    samples cannot hold many more dyadic modes than this.

    Examples
    --------
    >>> default_max_modes(256)
    7
    """
    return max(1, int(np.floor(np.log2(max(n_samples, 2)))) - 1)


@dataclass(frozen=True, eq=False)
class ModeDecomposition:
    """IMFs and residual of a decomposition.

    Attributes
    ----------
    signal : numpy.ndarray
        The decomposed signal.
    imfs : numpy.ndarray
        Shaped ``(n_modes, n_samples)``, highest frequency first.
    residual : numpy.ndarray
        The trend left over.
    sift_iterations : tuple of int
        Sifting steps spent on each IMF.

    Examples
    --------
    >>> parts = ModeDecomposition(np.ones(3), np.ones((1, 3)), np.zeros(3), (2,))
    >>> parts.n_modes, parts.reconstruct().tolist()
    (1, [1.0, 1.0, 1.0])
    """

    signal: np.ndarray
    imfs: np.ndarray
    residual: np.ndarray
    sift_iterations: tuple[int, ...] = ()

    @property
    def n_modes(self) -> int:
        """Number of IMFs."""
        return int(self.imfs.shape[0])

    def reconstruct(self) -> np.ndarray:
        """Sum of the IMFs and the residual."""
        return self.imfs.sum(axis=0) + self.residual


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got shape {x.shape}")
    if x.size < 4:
        raise ValueError(f"signal needs at least 4 samples, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or infinite values")
    return x


def extract_mode(
    signal: np.ndarray,
    *,
    stopping: StoppingCriterion | None = None,
    max_sift_iterations: int = DEFAULT_MAX_SIFT_ITERATIONS,
    time: np.ndarray | None = None,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> tuple[np.ndarray, int] | None:
    """Sift out the highest-frequency IMF of a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional signal.
    stopping : StoppingCriterion, optional
        Defaults to :class:`ThreeThresholdStopping`.
    max_sift_iterations : int, optional
        Hard cap on sifting steps.
    time : numpy.ndarray, optional
        Sample times; defaults to indices.
    n_mirror : int, optional
        Extrema mirrored at each end for the envelopes.

    Returns
    -------
    (numpy.ndarray, int) or None
        The IMF and the number of sifting steps used, or ``None`` when the
        signal is already a trend (fewer than three extrema).

    Examples
    --------
    >>> t = np.linspace(0, 1, 400)
    >>> mode, steps = extract_mode(np.sin(2 * np.pi * 30 * t) + t)
    >>> bool(np.corrcoef(mode, np.sin(2 * np.pi * 30 * t))[0, 1] > 0.99), steps >= 1
    (True, True)
    >>> extract_mode(t) is None
    True
    """
    criterion = ThreeThresholdStopping() if stopping is None else stopping
    if max_sift_iterations < 1:
        raise ValueError(f"max_sift_iterations must be at least 1, got {max_sift_iterations}")
    candidate = np.asarray(signal, dtype=np.float64)
    previous: np.ndarray | None = None
    for iteration in range(max_sift_iterations + 1):
        step = sift_once(candidate, time, n_mirror=n_mirror)
        if step is None:
            if iteration == 0:
                return None
            # Sifting flattened the candidate below three extrema: accept it.
            return candidate, iteration
        state = SiftState(
            candidate=candidate,
            previous=previous,
            mean_envelope=step.envelopes.mean,
            amplitude=step.envelopes.amplitude,
            iteration=iteration,
            n_extrema=step.extrema.count,
            n_zero_crossings=step.n_zero_crossings,
        )
        if iteration == max_sift_iterations or criterion.should_stop(state):
            if iteration == max_sift_iterations:
                _LOGGER.debug("sifting reached the %d-step cap", max_sift_iterations)
            return candidate, iteration
        previous = candidate
        candidate = step.candidate
    return candidate, max_sift_iterations  # pragma: no cover - loop always returns


def decompose_emd(
    signal: np.ndarray,
    *,
    max_modes: int | None = None,
    stopping: StoppingCriterion | None = None,
    max_sift_iterations: int = DEFAULT_MAX_SIFT_ITERATIONS,
    time: np.ndarray | None = None,
    n_mirror: int = DEFAULT_MIRRORED_EXTREMA,
) -> ModeDecomposition:
    """Decompose a signal into IMFs and a residual trend.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional, finite signal of at least four samples.
    max_modes : int, optional
        Maximum number of IMFs; defaults to :func:`default_max_modes`.
    stopping : StoppingCriterion, optional
        Sifting stopping criterion; defaults to :class:`ThreeThresholdStopping`.
    max_sift_iterations : int, optional
        Hard cap on sifting steps per IMF.
    time : numpy.ndarray, optional
        Sample times (strictly increasing); defaults to indices.
    n_mirror : int, optional
        Extrema mirrored at each end for the envelopes.

    Returns
    -------
    ModeDecomposition
        IMFs (highest frequency first) and residual, which sum exactly to the
        input.

    Raises
    ------
    ValueError
        If the signal is not one-dimensional, too short, or not finite.

    References
    ----------
    .. [1] N. E. Huang, Z. Shen, S. R. Long, M. C. Wu, H. H. Shih, Q. Zheng,
       N.-C. Yen, C. C. Tung, and H. H. Liu, "The empirical mode decomposition
       and the Hilbert spectrum for nonlinear and non-stationary time series
       analysis," Proceedings of the Royal Society of London A, vol. 454,
       no. 1971, pp. 903-995, 1998.
    .. [2] G. Rilling, P. Flandrin, and P. Goncalves, "On empirical mode
       decomposition and its algorithms," in Proc. IEEE-EURASIP Workshop on
       Nonlinear Signal and Image Processing (NSIP), 2003.

    Examples
    --------
    >>> t = np.linspace(0, 1, 512)
    >>> x = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 4 * t) + t
    >>> result = decompose_emd(x)
    >>> bool(np.allclose(result.reconstruct(), x))
    True
    >>> bool(np.corrcoef(result.imfs[0], np.sin(2 * np.pi * 40 * t))[0, 1] > 0.99)
    True
    """
    x = _validate_signal(signal)
    limit = default_max_modes(x.size) if max_modes is None else int(max_modes)
    if limit < 0:
        raise ValueError(f"max_modes must be non-negative, got {max_modes}")
    if time is not None and (np.shape(time) != x.shape or np.any(np.diff(time) <= 0)):
        raise ValueError("time must be strictly increasing with one entry per sample")

    scale = float(np.std(x))
    remainder = x.copy()
    imfs: list[np.ndarray] = []
    iterations: list[int] = []
    while len(imfs) < limit:
        if scale == 0.0 or float(np.std(remainder)) <= 1e-12 * scale:
            break
        if find_extrema(remainder).count < 3:
            break
        extracted = extract_mode(
            remainder,
            stopping=stopping,
            max_sift_iterations=max_sift_iterations,
            time=time,
            n_mirror=n_mirror,
        )
        if extracted is None:
            break
        mode, n_steps = extracted
        imfs.append(mode)
        iterations.append(n_steps)
        remainder = remainder - mode

    stacked = np.vstack(imfs) if imfs else np.zeros((0, x.size))
    residual = x - stacked.sum(axis=0)
    result = ModeDecomposition(
        signal=x, imfs=stacked, residual=residual, sift_iterations=tuple(iterations)
    )
    if not np.allclose(result.reconstruct(), x, rtol=1e-9, atol=1e-9 * max(scale, 1.0)):
        raise RuntimeError("EMD reconstruction check failed: IMFs + residual != signal")
    return result
