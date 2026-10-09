"""Room size and source distance from first-order echo arrivals.

The scene is planar: a rectangular room :math:`[0, W] \\times [0, D]`, a
microphone at :math:`(m_x, m_y)` and a source offset from it by
:math:`(a, b) = d_0 (\\cos\\theta, \\sin\\theta)`, where :math:`d_0` is the
direct-path distance and :math:`\\theta` the source bearing. With
:math:`u_x = 2m_x + a` and :math:`v_x = 2W - 2m_x - a` (and likewise
:math:`u_y, v_y`), the first-order image sources lie at distances

.. math::

    d_{x,\\min}^2 = u_x^2 + b^2, \\quad d_{x,\\max}^2 = v_x^2 + b^2, \\quad
    d_{y,\\min}^2 = a^2 + u_y^2, \\quad d_{y,\\max}^2 = a^2 + v_y^2,

and the four corner (two-reflection) images at
:math:`\\sqrt{u_x^2 + u_y^2}, \\sqrt{v_x^2 + u_y^2}, \\sqrt{u_x^2 + v_y^2}`
and :math:`\\sqrt{v_x^2 + v_y^2}`.

Given the bearing, the four wall equations decouple and have a closed form:

.. math::

    m_x = \\tfrac12\\big(\\sqrt{d_{x,\\min}^2 - b^2} - a\\big), \\qquad
    W = m_x + \\tfrac12\\big(a + \\sqrt{d_{x,\\max}^2 - b^2}\\big),

and the same for :math:`m_y, D`. Each square root has two signs, so there
are :math:`2^4 = 16` algebraic branches; only the positive one can be
physical, because :math:`u_x = m_x + s_x` and :math:`v_x` are sums of
coordinates inside the room. :func:`closed_form_first_order` can return all
16 to show this, and the feasibility test removes the others.

Arrival times do not say which wall produced them. The earliest echo is
taken as the direct sound and every ordered choice of four of the remaining
echoes is tried as (x_min, x_max, y_min, y_max): with eight echoes that is
:math:`8!/4! = 1680` assignments, solved at once as arrays. Candidates whose
room, microphone or source is impossible are dropped. The corner images are
the over-determined check: each predicted corner is matched with the nearest
echo and each unassigned echo with the nearest predicted corner, and the
root-mean-square mismatch is the candidate's residual. When fewer than eight
reflections are found (coinciding arrivals merge into one peak), one echo
may stand for two walls. This is the "hypothesize and test" idea of echo
labeling (Dokmanic et al., 2013); an optional least-squares fit then adjusts
the layout to all nine arrivals at once.

The bearing itself cannot be recovered from these arrivals. Every corner
distance follows from two wall distances and the direct path,

.. math::

    d_{x,\\min + y,\\min}^2 = d_{x,\\min}^2 + d_{y,\\min}^2 - d_0^2

(and likewise for the other three), so the corners check the labeling but
carry no information about :math:`\\theta`: for every bearing the closed
form reproduces all nine arrivals exactly. First-order arrivals therefore
fix the room only up to a one-parameter family,

.. math::

    W(\\theta) = \\tfrac12\\Big(\\sqrt{d_{x,\\min}^2 - b^2}
                 + \\sqrt{d_{x,\\max}^2 - b^2}\\Big),
    \\qquad b = d_0 \\sin\\theta,

and :math:`D(\\theta)` with :math:`a = d_0 \\cos\\theta`, so the bearing has to
be known or assumed (45 degrees by default). A wrong assumption costs little
when the source is close to the microphone compared with the walls, because
:math:`W \\approx \\tfrac12 (d_{x,\\min} + d_{x,\\max}) - \\tfrac{b^2}{4}
(d_{x,\\min}^{-1} + d_{x,\\max}^{-1})`; :func:`bearing_family` lists the whole
family for a set of arrivals.

Arrival times also cannot distinguish the 32 equivalent layouts of
:mod:`room_geometry_inference.geometry`. Candidates are deduplicated by their
canonical layout, and ``n_equivalent`` reports how many solutions collapsed
into each (eight under the 45-degree model). The room size and source
distance are identifiable, the absolute microphone position only up to these
symmetries.

Finally, :func:`rank_candidates` resynthesizes the impulse response of each
candidate with the image-source model and compares it with the observed
one, either by ``tap_match`` (the share of predicted arrivals without an
observed echo within a tolerance, and of observed echoes without a
predicted arrival; the tolerance absorbs the rounding of arrival times) or
by the L1 or L2 distance of the peak-normalized magnitudes. Ties are broken
by the residual.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .echoes import distances_to_taps, pick_echoes
from .errors import InversionError
from .geometry import canonical_layouts
from .image_source import SPEED_OF_SOUND_M_S, image_source_rir

_LOGGER = logging.getLogger(__name__)

#: Names of the nine first-order arrivals, in the column order of
#: :func:`first_order_distances`.
FIRST_ORDER_LABELS: tuple[str, ...] = (
    "direct",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "x_min+y_min",
    "x_max+y_min",
    "x_min+y_max",
    "x_max+y_max",
)
#: Candidate comparisons accepted by :func:`rank_candidates` and ``inversion.rank_metric``.
RANK_METRICS: tuple[str, ...] = ("tap_match", "l1", "l2")
#: Columns of the candidate tables returned by :func:`invert_echoes`.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "width_m",
    "depth_m",
    "mic_x_m",
    "mic_y_m",
    "source_x_m",
    "source_y_m",
    "source_distance_m",
    "bearing_deg",
    "residual_m",
    "n_equivalent",
    "assignment",
    "branch",
)

#: Rounding (decimals of a meter) used to recognize equivalent candidates.
_LAYOUT_DECIMALS = 3
#: Largest gap (m) a corner or an unassigned echo adds to the mismatch.
_MISMATCH_CAP_M = 0.5


@dataclass(frozen=True)
class RoomEstimate:
    """The selected room geometry.

    Attributes
    ----------
    width_m, depth_m : float
        Room sides (the longer one first).
    mic_xy, source_xy : tuple of float
        One of the equivalent microphone and source placements.
    source_distance_m : float
        Direct-path distance between source and microphone.
    bearing_deg : float
        Source bearing seen from the microphone.
    residual_m : float
        Root-mean-square mismatch of the matched first-order arrivals.
    score : float
        Ranking score (smaller is better; ``nan`` when not ranked).
    n_equivalent : int
        Equivalent solutions merged into this one.

    Examples
    --------
    >>> estimate = RoomEstimate(10.7, 7.5, (4.35, 2.75), (5.35, 3.75), 1.414, 45.0, 0.0, 0.0, 8)
    >>> estimate.size_m, estimate.as_dict()["mic_xy"]
    ((10.7, 7.5), [4.35, 2.75])
    """

    width_m: float
    depth_m: float
    mic_xy: tuple[float, float]
    source_xy: tuple[float, float]
    source_distance_m: float
    bearing_deg: float
    residual_m: float
    score: float
    n_equivalent: int

    @property
    def size_m(self) -> tuple[float, float]:
        """``(width_m, depth_m)``."""
        return (self.width_m, self.depth_m)

    def as_dict(self) -> dict[str, Any]:
        """Plain values, ready for JSON."""
        values = asdict(self)
        values["mic_xy"] = list(self.mic_xy)
        values["source_xy"] = list(self.source_xy)
        return values


def offsets_from_bearing(
    distance_m: ArrayLike, bearing_deg: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Source offset :math:`(a, b)` from the microphone for a distance and bearing.

    Examples
    --------
    >>> a, b = offsets_from_bearing(np.sqrt(2.0), 45.0)
    >>> round(float(a), 12), round(float(b), 12)
    (1.0, 1.0)
    """
    theta = np.deg2rad(np.asarray(bearing_deg, dtype=np.float64))
    distance = np.asarray(distance_m, dtype=np.float64)
    return distance * np.cos(theta), distance * np.sin(theta)


def first_order_distances(
    width_m: ArrayLike,
    depth_m: ArrayLike,
    mic_x_m: ArrayLike,
    mic_y_m: ArrayLike,
    offset_x_m: ArrayLike,
    offset_y_m: ArrayLike,
) -> NDArray[np.float64]:
    """Distances of the nine first-order arrivals of a planar layout.

    Parameters
    ----------
    width_m, depth_m, mic_x_m, mic_y_m, offset_x_m, offset_y_m : array_like
        Broadcastable layout parameters; the source sits at
        ``(mic_x + offset_x, mic_y + offset_y)``.

    Returns
    -------
    numpy.ndarray, shape (..., 9)
        Columns in the order of :data:`FIRST_ORDER_LABELS`.

    Examples
    --------
    >>> d = first_order_distances(10.7, 7.5, 5.35, 3.75, 1.0, 1.0)
    >>> np.floor(d * 16_000 / 343 + 0.5).astype(int).tolist()
    [66, 548, 455, 399, 307, 675, 602, 624, 545]
    """
    width, depth, mx, my, a, b = np.broadcast_arrays(
        *(
            np.asarray(v, dtype=np.float64)
            for v in (width_m, depth_m, mic_x_m, mic_y_m, offset_x_m, offset_y_m)
        )
    )
    near_x, far_x = 2.0 * mx + a, 2.0 * width - 2.0 * mx - a
    near_y, far_y = 2.0 * my + b, 2.0 * depth - 2.0 * my - b
    return np.stack(
        [
            np.hypot(a, b),
            np.hypot(near_x, b),
            np.hypot(far_x, b),
            np.hypot(a, near_y),
            np.hypot(a, far_y),
            np.hypot(near_x, near_y),
            np.hypot(far_x, near_y),
            np.hypot(near_x, far_y),
            np.hypot(far_x, far_y),
        ],
        axis=-1,
    )


def _branch_signs(all_branches: bool) -> NDArray[np.float64]:
    if not all_branches:
        return np.ones((1, 4))
    return np.asarray(list(itertools.product((1.0, -1.0), repeat=4)))


def closed_form_first_order(
    direct_m: ArrayLike,
    x_min_m: ArrayLike,
    x_max_m: ArrayLike,
    y_min_m: ArrayLike,
    y_max_m: ArrayLike,
    *,
    bearing_deg: ArrayLike = 45.0,
    all_branches: bool = False,
) -> NDArray[np.float64]:
    """Room size and microphone position from the direct path and four wall echoes.

    Parameters
    ----------
    direct_m : array_like
        Direct-path distance :math:`d_0`.
    x_min_m, x_max_m, y_min_m, y_max_m : array_like
        Distances of the echoes assigned to the four walls.
    bearing_deg : array_like, optional
        Source bearing seen from the microphone (45 degrees puts the source
        at :math:`(m_x + L, m_y + L)` with :math:`L = d_0/\\sqrt 2`).
    all_branches : bool, optional
        Return all :math:`2^4` sign branches of the square roots instead of
        only the positive, physical one.

    Returns
    -------
    numpy.ndarray, shape (..., n_branches, 4)
        ``(width, depth, mic_x, mic_y)`` for every branch; ``nan`` where a
        square root has a negative argument. Branch 0 is the positive one.

    References
    ----------
    .. [1] J. B. Allen and D. A. Berkley, "Image method for efficiently
       simulating small-room acoustics," The Journal of the Acoustical Society
       of America, vol. 65, no. 4, pp. 943-950, 1979.

    Examples
    --------
    >>> d = first_order_distances(10.7, 7.5, 5.35, 3.75, 1.0, 1.0)
    >>> solution = closed_form_first_order(d[0], d[1], d[2], d[3], d[4])
    >>> solution.round(6).tolist()
    [[10.7, 7.5, 5.35, 3.75]]
    >>> closed_form_first_order(*d[:5], all_branches=True).shape
    (16, 4)
    """
    a, b = offsets_from_bearing(direct_m, bearing_deg)
    with np.errstate(invalid="ignore"):
        roots = [
            np.sqrt(np.asarray(x_min_m, dtype=np.float64) ** 2 - b**2),
            np.sqrt(np.asarray(x_max_m, dtype=np.float64) ** 2 - b**2),
            np.sqrt(np.asarray(y_min_m, dtype=np.float64) ** 2 - a**2),
            np.sqrt(np.asarray(y_max_m, dtype=np.float64) ** 2 - a**2),
        ]
    signs = _branch_signs(all_branches)
    a, b, *roots = (np.asarray(v)[..., None] for v in np.broadcast_arrays(a, b, *roots))
    mic_x = 0.5 * (signs[:, 0] * roots[0] - a)
    width = mic_x + 0.5 * (a + signs[:, 1] * roots[1])
    mic_y = 0.5 * (signs[:, 2] * roots[2] - b)
    depth = mic_y + 0.5 * (b + signs[:, 3] * roots[3])
    return np.stack([width, depth, mic_x, mic_y], axis=-1)


def enumerate_assignments(
    n_echoes: int, n_walls: int = 4, *, allow_repeats: bool = False
) -> NDArray[np.int64]:
    """Every ordered choice of ``n_walls`` echoes out of ``n_echoes``.

    Parameters
    ----------
    n_echoes : int
        Number of reflections available.
    n_walls : int, optional
        Number of walls to label.
    allow_repeats : bool, optional
        Let one echo serve several walls, for arrivals that coincide (two
        reflections at the same distance merge into one peak).

    Returns
    -------
    numpy.ndarray of int64, shape (n_assignments, n_walls)

    Examples
    --------
    >>> enumerate_assignments(8).shape, enumerate_assignments(4)[1].tolist()
    ((1680, 4), [0, 1, 3, 2])
    >>> enumerate_assignments(3, allow_repeats=True).shape
    (81, 4)
    """
    if allow_repeats:
        if n_echoes < 1:
            raise InversionError("need at least one reflection")
        choices = itertools.product(range(n_echoes), repeat=n_walls)
    else:
        if n_echoes < n_walls:
            raise InversionError(f"need at least {n_walls} reflections, got {n_echoes}")
        choices = itertools.permutations(range(n_echoes), n_walls)
    return np.asarray(list(choices), dtype=np.int64)


def _mismatch(
    distances: NDArray[np.float64],
    direct: float,
    reflections: NDArray[np.float64],
    chosen: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Root-mean-square mismatch between predicted first-order arrivals and the echoes.

    The direct path and the walls are compared with the echoes assigned to
    them, every predicted corner with its nearest echo (a corner may coincide
    with a wall reflection), and every unassigned echo with its nearest
    predicted corner, so neither a corner without an echo nor an echo without
    an explanation goes unpenalized. Each of these gaps counts at most
    :data:`_MISMATCH_CAP_M`, so one spurious or missing echo cannot outweigh
    a layout that explains everything else.
    """
    n_rows = distances.shape[0]
    remaining = np.ones((n_rows, reflections.size), dtype=bool)
    remaining[np.arange(n_rows)[:, None], chosen] = False
    gaps = np.minimum(np.abs(distances[:, 5:, None] - reflections[None, None, :]), _MISMATCH_CAP_M)
    corner = gaps.min(axis=2)
    unexplained = np.where(remaining, gaps.min(axis=1), 0.0)
    total = (
        (distances[:, 0] - direct) ** 2
        + np.sum((distances[:, 1:5] - reflections[chosen]) ** 2, axis=1)
        + np.sum(corner**2, axis=1)
        + np.sum(unexplained**2, axis=1)
    )
    return np.sqrt(total / (9 + remaining.sum(axis=1)))


def _closed_form_candidates(
    direct: float,
    reflections: NDArray[np.float64],
    assignments: NDArray[np.int64],
    bearings: NDArray[np.float64],
    all_branches: bool,
) -> dict[str, NDArray[Any]]:
    """Feasible closed-form layouts for every assignment at every bearing."""
    walls = reflections[assignments]
    solution = closed_form_first_order(
        direct,
        walls[None, :, 0],
        walls[None, :, 1],
        walls[None, :, 2],
        walls[None, :, 3],
        bearing_deg=bearings[:, None],
        all_branches=all_branches,
    )  # (n_bearings, n_assignments, n_branches, 4)
    a, b = offsets_from_bearing(direct, bearings)
    a = np.broadcast_to(a[:, None, None], solution.shape[:3])
    b = np.broadcast_to(b[:, None, None], solution.shape[:3])
    width, depth, mic_x, mic_y = np.moveaxis(solution, -1, 0)
    src_x, src_y = mic_x + a, mic_y + b
    with np.errstate(invalid="ignore"):
        feasible = (
            np.isfinite(width)
            & np.isfinite(depth)
            & (mic_x > 0)
            & (mic_x < width)
            & (mic_y > 0)
            & (mic_y < depth)
            & (src_x > 0)
            & (src_x < width)
            & (src_y > 0)
            & (src_y < depth)
        )
    index = np.nonzero(feasible)
    distances = first_order_distances(
        width[index], depth[index], mic_x[index], mic_y[index], a[index], b[index]
    )
    return {
        "layout": np.stack(
            [width[index], depth[index], mic_x[index], mic_y[index], src_x[index], src_y[index]],
            axis=1,
        ),
        "bearing": bearings[index[0]],
        "assignment": index[1],
        "branch": index[2],
        "residual": _mismatch(distances, direct, reflections, assignments[index[1]]),
    }


def _refine(
    layout: NDArray[np.float64],
    bearing: float,
    chosen: NDArray[np.int64],
    direct: float,
    reflections: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Least-squares fit of a candidate to all nine first-order arrivals.

    Each predicted corner is paired with its nearest echo first; then width,
    depth, microphone position and source distance are fitted together, the
    bearing staying fixed.
    """
    width, depth, mic_x, mic_y = layout[:4]
    a, b = offsets_from_bearing(direct, bearing)
    corners = first_order_distances(width, depth, mic_x, mic_y, a, b)[5:]
    matched = reflections[np.argmin(np.abs(corners[:, None] - reflections[None, :]), axis=1)]
    target = np.concatenate([[direct], reflections[chosen], matched])

    def residuals(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        oa, ob = offsets_from_bearing(parameters[4], bearing)
        return first_order_distances(*parameters[:4], oa, ob) - target

    start = np.asarray([width, depth, mic_x, mic_y, direct], dtype=np.float64)
    fit = least_squares(residuals, start, method="lm")
    w, d, x, y, rho = fit.x
    oa, ob = offsets_from_bearing(rho, bearing)
    refined = np.asarray([w, d, x, y, x + oa, y + ob], dtype=np.float64)
    predicted = first_order_distances(w, d, x, y, oa, ob)[None, :]
    return refined, float(_mismatch(predicted, direct, reflections, chosen[None, :])[0])


def _feasible(layout: NDArray[np.float64]) -> bool:
    width, depth, mx, my, sx, sy = layout
    return bool(0 < mx < width and 0 < sx < width and 0 < my < depth and 0 < sy < depth)


def _observed(distances_m: ArrayLike) -> tuple[float, NDArray[np.float64]]:
    """Direct-path distance and the reflections, after validation."""
    observed = np.sort(np.asarray(distances_m, dtype=np.float64).ravel())
    if observed.size < 3:
        raise InversionError(
            f"need the direct sound and at least two reflections, got {observed.size} arrivals"
        )
    if not np.all(np.isfinite(observed)) or observed[0] <= 0:
        raise InversionError(
            f"arrival distances must be positive and finite, got {observed.tolist()}"
        )
    return float(observed[0]), observed[1:]


def invert_echoes(
    distances_m: ArrayLike,
    *,
    bearing_deg: float = 45.0,
    all_branches: bool = False,
    refine: bool = True,
    max_candidates: int = 50,
) -> pd.DataFrame:
    """Every room layout consistent with a set of echo distances, best first.

    Parameters
    ----------
    distances_m : array_like
        Travel distances of the picked arrivals (any order); the smallest is
        the direct path. With fewer than eight reflections, arrivals are
        assumed to have merged and one echo may serve two walls.
    bearing_deg : float, optional
        Assumed source bearing seen from the microphone (see the module
        documentation for why it cannot be estimated from the arrivals).
    all_branches : bool, optional
        Try all 16 sign branches of the closed form (only the positive one
        can be feasible).
    refine : bool, optional
        Fit each kept candidate to all nine arrivals by least squares
        (Levenberg-Marquardt); the fit is kept only when it stays feasible
        and does not increase the mismatch.
    max_candidates : int, optional
        Number of distinct candidates kept (smallest residual first).

    Returns
    -------
    pandas.DataFrame
        Columns :data:`CANDIDATE_COLUMNS`, sorted by ``residual_m``, the
        root-mean-square mismatch between predicted and observed arrivals.
        Sizes and positions are those of the canonical layout
        (:func:`~room_geometry_inference.geometry.canonical_layouts`); layouts
        that agree to the millimeter are merged. ``assignment`` lists the echo
        numbers (0 = direct sound, in order of arrival) taken as the x_min,
        x_max, y_min and y_max reflections.

    Raises
    ------
    InversionError
        With fewer than three arrivals or when no layout is feasible.

    References
    ----------
    .. [1] I. Dokmanic, R. Parhizkar, A. Walther, Y. M. Lu, and M. Vetterli,
       "Acoustic echoes reveal room shape," Proceedings of the National
       Academy of Sciences, vol. 110, no. 30, pp. 12186-12191, 2013.

    Examples
    --------
    >>> d = first_order_distances(10.7, 7.5, 5.35, 3.75, 1.0, 1.0)
    >>> best = invert_echoes(d).iloc[0]
    >>> [round(float(best[k]), 6) for k in ("width_m", "depth_m", "source_distance_m")]
    [10.7, 7.5, 1.414214]
    >>> int(best["n_equivalent"]), float(best["residual_m"]) < 1e-9
    (8, True)
    """
    direct, reflections = _observed(distances_m)
    if not np.isfinite(bearing_deg):
        raise InversionError(f"bearing_deg must be finite, got {bearing_deg!r}")
    # Fewer than eight reflections means some arrivals coincided and merged:
    # then one echo may stand for two walls.
    assignments = enumerate_assignments(reflections.size, allow_repeats=reflections.size < 8)
    found = _closed_form_candidates(
        direct, reflections, assignments, np.asarray([float(bearing_deg)]), all_branches
    )
    if found["residual"].size == 0:
        raise InversionError(
            f"no room layout is consistent with the {reflections.size + 1} arrivals "
            f"(direct path {direct:.3f} m, bearing {bearing_deg:g} degrees)"
        )
    order = np.lexsort((found["assignment"], found["residual"]))
    keys = canonical_layouts(found["layout"][order], _LAYOUT_DECIMALS)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    distinct = np.argsort(first, kind="stable")[: int(max_candidates)]
    rows = []
    for position, count in zip(order[first[distinct]], counts[distinct], strict=True):
        layout = found["layout"][position]
        residual = float(found["residual"][position])
        chosen = assignments[found["assignment"][position]]
        if refine:
            refined, fitted = _refine(layout, float(bearing_deg), chosen, direct, reflections)
            if _feasible(refined) and fitted <= residual + 1e-12:
                layout, residual = refined, fitted
        rows.append(
            {
                "layout": layout,
                "residual_m": residual,
                "n_equivalent": int(count),
                "assignment": ",".join(str(int(k) + 1) for k in chosen),
                "branch": int(found["branch"][position]),
            }
        )
    table = _deduplicate(rows)
    _LOGGER.debug(
        "%d feasible solutions, %d distinct layouts kept", found["residual"].size, len(table)
    )
    return table


def bearing_family(
    distances_m: ArrayLike,
    *,
    bearings_deg: ArrayLike | None = None,
    reference_bearing_deg: float = 45.0,
) -> pd.DataFrame:
    """The layouts that reproduce the arrivals exactly, one per source bearing.

    The echo labeling is taken from the best candidate of
    :func:`invert_echoes` at ``reference_bearing_deg``; because the corner
    check does not depend on the bearing, that labeling is valid at every
    bearing, and the closed form gives one layout per bearing.

    Parameters
    ----------
    distances_m : array_like
        Travel distances of the picked arrivals.
    bearings_deg : array_like, optional
        Bearings to evaluate (default: every degree from 0 to 359).
    reference_bearing_deg : float, optional
        Bearing used to choose the echo labeling.

    Returns
    -------
    pandas.DataFrame
        One row per bearing with a feasible layout: ``bearing_deg``,
        ``width_m``, ``depth_m``, ``mic_x_m``, ``mic_y_m``, ``source_x_m``,
        ``source_y_m`` and ``residual_m``.

    Examples
    --------
    >>> d = first_order_distances(10.7, 7.5, 5.35, 3.75, 1.0, 1.0)
    >>> family = bearing_family(d, bearings_deg=[30.0, 45.0, 60.0])
    >>> family["width_m"].round(3).tolist(), float(family["residual_m"].max()) < 1e-9
    ([10.724, 10.7, 10.676], True)
    """
    direct, reflections = _observed(distances_m)
    best = invert_echoes(
        distances_m, bearing_deg=reference_bearing_deg, refine=False, max_candidates=1
    ).iloc[0]
    chosen = np.asarray([int(k) - 1 for k in str(best["assignment"]).split(",")], dtype=np.int64)
    grid = np.arange(360.0) if bearings_deg is None else np.asarray(bearings_deg, np.float64)
    # Label the walls so the longer side comes out as the width: exchanging the
    # x and y walls of the labeling transposes the layout (bearing 90 - theta).
    width, depth = closed_form_first_order(
        direct, *reflections[chosen], bearing_deg=reference_bearing_deg
    )[0, :2]
    if depth > width:
        chosen = chosen[[2, 3, 0, 1]]
    found = _closed_form_candidates(direct, reflections, chosen[None, :], grid.ravel(), False)
    table = pd.DataFrame(found["layout"], columns=list(CANDIDATE_COLUMNS[:6]))
    table.insert(0, "bearing_deg", found["bearing"])
    table["residual_m"] = found["residual"]
    return table.reset_index(drop=True)


def _deduplicate(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Merge equivalent layouts: smallest residual kept, equivalence counts summed.

    Layouts are compared by their canonical form rounded to
    :data:`_LAYOUT_DECIMALS`; the table reports the canonical form unrounded,
    so a point a fraction of a millimeter from a wall is not moved onto it.
    """
    layouts = np.stack([row["layout"] for row in rows])
    keys = canonical_layouts(layouts, _LAYOUT_DECIMALS)
    values = canonical_layouts(layouts, _LAYOUT_DECIMALS, rounded=False)
    # Sides equal to the millimeter may still come out in either order: keep
    # the longer one first by transposing those layouts (an equivalent scene).
    narrow = values[:, 0] < values[:, 1]
    values[narrow] = values[narrow][:, [1, 0, 3, 2, 5, 4]]
    key_columns = [f"_key_{column}" for column in range(keys.shape[1])]
    table = pd.concat(
        [
            pd.DataFrame(values, columns=list(CANDIDATE_COLUMNS[:6])),
            pd.DataFrame(keys, columns=key_columns),
        ],
        axis=1,
    )
    table["source_distance_m"] = np.hypot(values[:, 4] - values[:, 2], values[:, 5] - values[:, 3])
    table["bearing_deg"] = (
        np.rad2deg(np.arctan2(values[:, 5] - values[:, 3], values[:, 4] - values[:, 2])) % 360.0
    )
    for name in ("residual_m", "n_equivalent", "assignment", "branch"):
        table[name] = [row[name] for row in rows]
    table["n_equivalent"] = table.groupby(key_columns)["n_equivalent"].transform("sum")
    table = table.sort_values(["residual_m", *key_columns], kind="stable")
    table = table.drop_duplicates(key_columns, keep="first")
    return table[list(CANDIDATE_COLUMNS)].reset_index(drop=True)


def rank_candidates(
    candidates: pd.DataFrame,
    observed_response: ArrayLike,
    *,
    sample_rate: int,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
    metric: str = "tap_match",
    tolerance_taps: float = 2.0,
    echo_taps: ArrayLike | None = None,
    max_order: int = 1,
    order_mode: str = "per_axis",
    reflection: float | Sequence[float] = 1.0,
    delay: str = "integer",
) -> pd.DataFrame:
    """Order candidates by how well their resynthesized response matches the observation.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Output of :func:`invert_echoes`.
    observed_response : array_like, shape (n_taps,)
        The measured or identified impulse response.
    sample_rate : int
        Sampling rate in Hz.
    speed_of_sound_m_s : float, optional
        Speed of sound used for the resynthesis.
    metric : {"tap_match", "l1", "l2"}, optional
        ``tap_match``: share of predicted arrivals farther than
        ``tolerance_taps`` from every observed echo and of observed echoes
        farther than that from every predicted arrival; ``l1``/``l2``:
        distance between the peak-normalized magnitudes, sample by sample.
    tolerance_taps : float, optional
        Tolerance of ``tap_match``.
    echo_taps : array_like, optional
        Observed echo arrivals for ``tap_match``; picked from the response
        when omitted.
    max_order, order_mode, reflection, delay
        Image-source settings of the resynthesis
        (see :func:`~room_geometry_inference.image_source.image_source_rir`).

    Returns
    -------
    pandas.DataFrame
        ``candidates`` with a ``score`` column (smaller is better), sorted by
        ``score`` and then ``residual_m``.

    Examples
    --------
    >>> from room_geometry_inference.image_source import image_source_rir
    >>> truth = image_source_rir([10.7, 7.5], [6.35, 4.75], [5.35, 3.75], sample_rate=16_000)
    >>> d = first_order_distances(10.7, 7.5, 5.35, 3.75, 1.0, 1.0)
    >>> ranked = rank_candidates(invert_echoes(d), truth.rir, sample_rate=16_000)
    >>> round(float(ranked["score"].iloc[0]), 6), float(ranked["width_m"].iloc[0])
    (0.0, 10.7)
    """
    if metric not in RANK_METRICS:
        raise ValueError(f"metric={metric!r} is not supported; choose one of {RANK_METRICS}")
    observed = np.abs(np.asarray(observed_response, dtype=np.float64).ravel())
    peak = float(observed.max()) if observed.size else 0.0
    if peak == 0.0:
        raise InversionError("the observed response is silent")
    observed = observed / peak
    if metric == "tap_match":
        reference = (
            pick_echoes(observed) if echo_taps is None else np.asarray(echo_taps, np.float64)
        )
    scores = []
    for row in candidates.itertuples(index=False):
        simulated = image_source_rir(
            [row.width_m, row.depth_m],
            [row.source_x_m, row.source_y_m],
            [row.mic_x_m, row.mic_y_m],
            sample_rate=sample_rate,
            n_taps=observed.size,
            speed_of_sound_m_s=speed_of_sound_m_s,
            max_order=max_order,
            order_mode=order_mode,
            reflection=reflection,
            delay=delay,
        )
        if metric == "tap_match":
            arrivals = distances_to_taps(simulated.distances_m, sample_rate, speed_of_sound_m_s)
            gaps = np.abs(arrivals[:, None] - reference[None, :])
            unmatched = np.sum(gaps.min(axis=1) > tolerance_taps) + np.sum(
                gaps.min(axis=0) > tolerance_taps
            )
            scores.append(float(unmatched) / float(arrivals.size + reference.size))
        else:
            difference = observed - np.abs(simulated.rir)
            value = np.sum(np.abs(difference)) if metric == "l1" else np.sqrt(np.sum(difference**2))
            scores.append(float(value))
    ranked = candidates.copy()
    ranked["score"] = scores
    return ranked.sort_values(["score", "residual_m"], kind="stable").reset_index(drop=True)


def best_estimate(ranked: pd.DataFrame) -> RoomEstimate:
    """The first row of a candidate table as a :class:`RoomEstimate`.

    Examples
    --------
    >>> d = first_order_distances(3.5, 3.0, 1.75, 1.5, 1.0, 1.0)
    >>> estimate = best_estimate(invert_echoes(d))
    >>> [round(v, 6) for v in estimate.size_m], round(estimate.source_distance_m, 6)
    ([3.5, 3.0], 1.414214)
    """
    if ranked.empty:
        raise InversionError("there is no candidate to choose from")
    row = ranked.iloc[0]
    width, depth = float(row["width_m"]), float(row["depth_m"])
    swap = depth > width
    mic = (float(row["mic_x_m"]), float(row["mic_y_m"]))
    source = (float(row["source_x_m"]), float(row["source_y_m"]))
    if swap:
        width, depth = depth, width
        mic, source = mic[::-1], source[::-1]
    bearing = float(np.rad2deg(np.arctan2(source[1] - mic[1], source[0] - mic[0])) % 360.0)
    return RoomEstimate(
        width_m=width,
        depth_m=depth,
        mic_xy=mic,
        source_xy=source,
        source_distance_m=float(row["source_distance_m"]),
        bearing_deg=bearing,
        residual_m=float(row["residual_m"]),
        score=float(row["score"]) if "score" in row.index else float("nan"),
        n_equivalent=int(row["n_equivalent"]),
    )
