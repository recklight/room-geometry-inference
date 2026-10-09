"""Planar room layouts: comparing sizes and folding away their symmetries.

A planar layout is the room size :math:`(W, D)` plus the microphone
:math:`\\mathbf{m}` and source :math:`\\mathbf{s}` positions, with the room
spanning :math:`[0, W] \\times [0, D]`. Echo arrival times cannot tell certain
layouts apart. An image distance is the hypotenuse of one x offset and one y
offset, and along an axis of length :math:`L` the image offsets are
:math:`(s - m) + 2kL` and :math:`-(s + m) + 2kL` for every integer
:math:`k`. That set of magnitudes is unchanged by four maps of the axis:

* the identity;
* mirroring, :math:`m \\to L - m` and :math:`s \\to L - s`;
* exchanging the source and microphone coordinates on that axis;
* mirroring and exchanging together, which keeps the sign of the source
  offset :math:`s - m`.

Applied independently to both axes, and combined with transposition
(exchanging the x and y axes), they give 32 layouts with identical image
distances. When every wall has the same reflection coefficient, their impulse
responses are identical as well. Eight of them keep the source bearing of the
45-degree model (the identity or mirror-and-exchange on each axis, with or
without transposition), which is why an echo inversion under that model
finds eight equally good answers. :func:`canonical_layouts` picks one
representative of each class (longer side first, then the smallest
coordinates) so candidate lists can be deduplicated, and
:func:`room_size_error` compares sizes as sorted side lengths, so an estimate
is not penalized for naming the sides the other way round.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Columns of a layout array: width, depth, microphone x and y, source x and y.
LAYOUT_COLUMNS: tuple[str, ...] = (
    "width_m",
    "depth_m",
    "mic_x_m",
    "mic_y_m",
    "source_x_m",
    "source_y_m",
)


def _layouts(values: ArrayLike) -> NDArray[np.float64]:
    layouts = np.asarray(values, dtype=np.float64)
    if layouts.ndim == 1:
        layouts = layouts[None, :]
    if layouts.ndim != 2 or layouts.shape[1] != len(LAYOUT_COLUMNS):
        raise ValueError(
            f"layouts need {len(LAYOUT_COLUMNS)} columns {list(LAYOUT_COLUMNS)}, "
            f"got shape {layouts.shape}"
        )
    return layouts


def _axis_maps(
    size: NDArray[np.float64], mic: NDArray[np.float64], src: NDArray[np.float64]
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """The four ``(mic, source)`` coordinate pairs of one axis, identity first."""
    return [(mic, src), (size - mic, size - src), (src, mic), (size - src, size - mic)]


def _variants(rows: NDArray[np.float64], transpose: bool) -> NDArray[np.float64]:
    """All equivalent variants of every row, shape ``(n, 16 or 32, 6)``, identity first."""
    width, depth, mic_x, mic_y, src_x, src_y = rows.T
    variants = [
        np.stack([width, depth, mx, my, sx, sy], axis=1)
        for mx, sx in _axis_maps(width, mic_x, src_x)
        for my, sy in _axis_maps(depth, mic_y, src_y)
    ]
    if transpose:
        variants += [variant[:, [1, 0, 3, 2, 5, 4]] for variant in variants]
    return np.stack(variants, axis=1)


def equivalent_layouts(layout: ArrayLike, *, transpose: bool = True) -> NDArray[np.float64]:
    """Every layout with the same image distances as ``layout``.

    Parameters
    ----------
    layout : array_like, shape (6,)
        ``(width, depth, mic_x, mic_y, source_x, source_y)`` in meters.
    transpose : bool, optional
        Include the axis exchange (16 layouts without it, 32 with it).

    Returns
    -------
    numpy.ndarray, shape (16, 6) or (32, 6)
        The identity comes first; duplicates are kept when the layout is
        itself symmetric.

    Examples
    --------
    >>> variants = equivalent_layouts([10.0, 6.0, 4.0, 2.0, 5.0, 3.0])
    >>> variants.shape, variants[1].tolist()
    ((32, 6), [10.0, 6.0, 4.0, 4.0, 5.0, 3.0])
    """
    return _variants(_layouts(layout)[:1], transpose)[0]


def canonical_layouts(
    layouts: ArrayLike, decimals: int = 6, *, rounded: bool = True
) -> NDArray[np.float64]:
    """One representative per symmetry class, row by row.

    The representative is the one of the 32 equivalent layouts with the
    longer side as width and, among those, the lexicographically smallest
    coordinates (after rounding to ``decimals``), so two rows describe
    indistinguishable scenes exactly when their canonical rows are equal.

    Parameters
    ----------
    layouts : array_like, shape (n, 6) or (6,)
        Rows of ``(width, depth, mic_x, mic_y, source_x, source_y)``.
    decimals : int, optional
        Rounding applied before the comparisons.
    rounded : bool, optional
        Return the representative rounded to ``decimals`` (the comparison
        key) or with its exact values.

    Returns
    -------
    numpy.ndarray, shape (n, 6)

    Examples
    --------
    >>> rows = [[10.0, 6.0, 4.0, 2.0, 5.0, 3.0], [10.0, 6.0, 5.0, 2.0, 6.0, 3.0],
    ...         [6.0, 10.0, 2.0, 4.0, 3.0, 5.0]]
    >>> np.unique(canonical_layouts(rows), axis=0).tolist()
    [[10.0, 6.0, 4.0, 2.0, 5.0, 3.0]]
    >>> exact = canonical_layouts([4.0, 3.0, 0.0004, 1.0, 1.0, 2.0], 3, rounded=False)
    >>> float(exact[0, 2])  # the rounded key would put the microphone on the wall
    0.0004
    """
    rows = _layouts(layouts)
    exact = _variants(rows, transpose=True)
    variants = np.round(exact, decimals)
    keep = np.ones(variants.shape[:2], dtype=bool)
    for column in range(variants.shape[2]):
        # The width is maximized (longer side first), every other column minimized.
        sign = -1.0 if column == 0 else 1.0
        values = np.where(keep, sign * variants[:, :, column], np.inf)
        keep &= values == values.min(axis=1, keepdims=True)
    choice = np.argmax(keep, axis=1)
    return (variants if rounded else exact)[np.arange(rows.shape[0]), choice]


def room_size_error(estimated: ArrayLike, reference: ArrayLike) -> float:
    """Sum of absolute side-length errors, sides sorted so naming does not matter.

    Absolute errors are summed, so a width 0.2 m too long and a depth 0.2 m
    too short count as 0.4 m instead of cancelling.

    Parameters
    ----------
    estimated, reference : array_like, shape (2,) or (3,)
        Side lengths in meters.

    Returns
    -------
    float
        :math:`\\sum_i |\\hat L_{(i)} - L_{(i)}|` in meters.

    Examples
    --------
    >>> room_size_error([7.5, 10.7], [10.7, 7.5])
    0.0
    >>> round(room_size_error([10.9, 7.3], [10.7, 7.5]), 6)
    0.4
    """
    first = np.sort(np.asarray(estimated, dtype=np.float64).ravel())[::-1]
    second = np.sort(np.asarray(reference, dtype=np.float64).ravel())[::-1]
    if first.shape != second.shape:
        raise ValueError(f"cannot compare {first.size} sides with {second.size}")
    return float(np.abs(first - second).sum())
