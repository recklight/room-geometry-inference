"""Impulse responses of rectangular rooms by the image-source method.

A point source in a rectangular room with specular walls is replaced by a
lattice of mirror images of itself (Allen & Berkley, 1979). Along an axis of
length :math:`L` with the source at coordinate :math:`s`, the image with index
:math:`k \\in \\mathbb{Z}` sits at

.. math::

    x_k = (-1)^k s + \\big(k + (k \\bmod 2)\\big) L

and reaches the receiver after :math:`|k|` reflections on that axis:
:math:`\\lfloor (|k| + [k < 0]) / 2 \\rfloor` on the wall at 0 and the rest on
the wall at :math:`L`. An image with indices :math:`(k_x, k_y[, k_z])` lies at
distance :math:`d` from the microphone; its sound arrives after
:math:`\\tau = d / c` with amplitude

.. math::

    a = \\frac{1}{d} \\prod_{w} \\beta_w^{n_w},

where :math:`n_w` counts the reflections on wall :math:`w` and
:math:`\\beta_w` is that wall's pressure reflection coefficient. The constant
:math:`1/4\\pi` is dropped because responses are normalized to a peak of one.

Two-dimensional rooms are simulated in two dimensions, not as a
three-dimensional room of zero height, which would count every planar image
three times. A first-order planar room therefore has :math:`3^2 = 9`
arrivals: the direct sound, four wall reflections and four corner
(two-reflection) images.

Options added to the classic method:

* ``order_mode = "per_axis"`` keeps :math:`|k_i| \\le N` on every axis, the
  classic lattice; ``"total"`` keeps :math:`\\sum_i |k_i| \\le N`, i.e. every
  reflection path of at most :math:`N` bounces.
* ``delay = "integer"`` rounds every arrival to the nearest sample (half-up,
  so an arrival exactly between two samples goes to the later one);
  ``"fractional"`` spreads it over neighboring samples with a Hann-windowed
  sinc (Peterson, 1986), which keeps sub-sample timing.
* Reflection coefficients are given once for all walls or per wall, and the
  speed of sound can be derived from the air temperature
  (:func:`speed_of_sound`; Kinsler et al., 2000).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import GeometryError

_LOGGER = logging.getLogger(__name__)

#: Speed of sound in air at about 20 degrees Celsius, in m/s.
SPEED_OF_SOUND_M_S: float = 343.0
#: Image lattices accepted by ``acoustics.order_mode``.
ORDER_MODES: tuple[str, ...] = ("per_axis", "total")
#: Arrival-time models accepted by ``acoustics.delay``.
DELAY_MODES: tuple[str, ...] = ("integer", "fractional")
#: Wall names in the order of :attr:`ImageSourceResult.wall_hits`.
WALL_NAMES: tuple[str, ...] = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

#: Half width, in samples, of the windowed sinc of fractional delays.
_SINC_HALF_WIDTH = 16


def speed_of_sound(temperature_c: float) -> float:
    """Speed of sound in dry air at a given temperature.

    .. math::

        c = 331.3 \\sqrt{1 + T / 273.15} \\ \\mathrm{m/s}

    Parameters
    ----------
    temperature_c : float
        Air temperature in degrees Celsius, above absolute zero.

    Returns
    -------
    float
        Speed of sound in m/s.

    References
    ----------
    .. [1] L. E. Kinsler, A. R. Frey, A. B. Coppens, and J. V. Sanders,
       Fundamentals of Acoustics, 4th ed. New York, NY, USA: Wiley, 2000.

    Examples
    --------
    >>> round(speed_of_sound(20.0), 1), round(speed_of_sound(0.0), 1)
    (343.2, 331.3)
    """
    temperature = float(temperature_c)
    if not temperature > -273.15:
        raise ValueError(f"temperature_c must be above -273.15, got {temperature_c!r}")
    return 331.3 * float(np.sqrt(1.0 + temperature / 273.15))


@dataclass(frozen=True, eq=False)
class ImageSourceResult:
    """A simulated impulse response and the images that build it.

    Only images that arrive inside the response window are kept.

    Attributes
    ----------
    rir : numpy.ndarray, shape (n_taps,)
        The impulse response; tap 0 is the emission time.
    sample_rate : int
        Sampling rate in Hz.
    image_indices : numpy.ndarray of int64, shape (n_images, n_dims)
        Lattice index :math:`k` of every image on every axis; all zeros is
        the direct sound.
    wall_hits : numpy.ndarray of int64, shape (n_images, 2 * n_dims)
        Reflections of every image on every wall, in the order of
        :data:`WALL_NAMES`.
    distances_m : numpy.ndarray, shape (n_images,)
        Image-to-microphone distances.
    delays_s : numpy.ndarray, shape (n_images,)
        Arrival times.
    amplitudes : numpy.ndarray, shape (n_images,)
        Arrival amplitudes, on the same scale as ``rir``.

    Examples
    --------
    Nine images, but this symmetric placement makes several arrive together:

    >>> result = image_source_rir([4.0, 3.0], [1.0, 1.0], [2.5, 2.0], sample_rate=8000)
    >>> result.orders.tolist(), result.arrival_taps().size
    ([0, 1, 1, 1, 1, 2, 2, 2, 2], 5)
    """

    rir: NDArray[np.float64]
    sample_rate: int
    image_indices: NDArray[np.int64]
    wall_hits: NDArray[np.int64]
    distances_m: NDArray[np.float64]
    delays_s: NDArray[np.float64]
    amplitudes: NDArray[np.float64]

    @property
    def orders(self) -> NDArray[np.int64]:
        """Number of reflections of every image."""
        return np.abs(self.image_indices).sum(axis=1)

    @property
    def taps(self) -> NDArray[np.int64]:
        """Arrival of every image rounded half-up to a sample index."""
        return np.floor(self.delays_s * self.sample_rate + 0.5).astype(np.int64)

    def arrival_taps(self) -> NDArray[np.int64]:
        """Distinct arrival samples, in increasing order."""
        return np.unique(self.taps)

    def labels(self) -> list[str]:
        """Readable name of every image: ``direct``, ``x_min``, ``x_max+y_min`` ...

        Images that hit a wall more than once carry the count (``x_min*2``).
        """
        names = WALL_NAMES[: self.wall_hits.shape[1]]
        result = []
        for hits in self.wall_hits:
            parts = [
                name if count == 1 else f"{name}*{count}"
                for name, count in zip(names, hits, strict=True)
                if count
            ]
            result.append("+".join(parts) if parts else "direct")
        return result


def _as_point(values: ArrayLike, name: str, n_dims: int) -> NDArray[np.float64]:
    point = np.asarray(values, dtype=np.float64).ravel()
    if point.size != n_dims:
        raise GeometryError(f"{name} needs {n_dims} coordinates like the room, got {point.size}")
    if not np.all(np.isfinite(point)):
        raise GeometryError(f"{name} must be finite, got {point.tolist()}")
    return point


def validate_scene(
    room_m: ArrayLike, source_m: ArrayLike, microphone_m: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Check that a room is 2-D or 3-D and that source and microphone lie inside it.

    Parameters
    ----------
    room_m : array_like, shape (2,) or (3,)
        Side lengths in meters (width, depth[, height]).
    source_m, microphone_m : array_like, shape like ``room_m``
        Positions in meters, with the room spanning ``[0, side]`` on every axis.

    Returns
    -------
    tuple of numpy.ndarray
        ``(room, source, microphone)`` as float arrays.

    Raises
    ------
    GeometryError
        If a side is not positive, the dimensions disagree, a point lies on
        or outside a wall, or source and microphone coincide.

    Examples
    --------
    >>> room, source, mic = validate_scene([4.0, 3.0], [1.0, 1.0], [2.0, 2.0])
    >>> room.tolist(), float(np.linalg.norm(source - mic)) > 0
    ([4.0, 3.0], True)
    >>> try:
    ...     validate_scene([4.0, 3.0], [1.0, 1.0], [5.0, 2.0])
    ... except GeometryError as exc:
    ...     print(exc)
    microphone at [5.0, 2.0] is not strictly inside the 4 x 3 m room
    """
    room = np.asarray(room_m, dtype=np.float64).ravel()
    if room.size not in (2, 3):
        raise GeometryError(f"a room has 2 or 3 side lengths, got {room.size}")
    if not np.all(np.isfinite(room)) or np.any(room <= 0):
        raise GeometryError(f"room sides must be positive, got {room.tolist()}")
    source = _as_point(source_m, "source", room.size)
    microphone = _as_point(microphone_m, "microphone", room.size)
    size = " x ".join(f"{side:g}" for side in room)
    for name, point in (("source", source), ("microphone", microphone)):
        if np.any(point <= 0) or np.any(point >= room):
            raise GeometryError(
                f"{name} at {point.tolist()} is not strictly inside the {size} m room"
            )
    if np.allclose(source, microphone):
        raise GeometryError("source and microphone coincide")
    return room, source, microphone


def lattice_indices(max_order: int, n_dims: int, order_mode: str = "per_axis") -> NDArray[np.int64]:
    """Image indices :math:`(k_x, k_y[, k_z])` of a truncated lattice.

    Parameters
    ----------
    max_order : int
        Truncation order :math:`N \\ge 0`.
    n_dims : int
        2 or 3.
    order_mode : {"per_axis", "total"}, optional
        ``per_axis`` keeps :math:`|k_i| \\le N` on every axis
        (:math:`(2N+1)^{n}` images); ``total`` keeps
        :math:`\\sum_i |k_i| \\le N`.

    Returns
    -------
    numpy.ndarray of int64, shape (n_images, n_dims)
        Ordered by total reflection count, then lexicographically.

    Examples
    --------
    >>> lattice_indices(1, 2).shape, lattice_indices(1, 2, "total").shape
    ((9, 2), (5, 2))
    >>> lattice_indices(2, 3, "per_axis").shape[0]
    125
    """
    if order_mode not in ORDER_MODES:
        raise ValueError(f"order_mode={order_mode!r} is not supported; choose one of {ORDER_MODES}")
    if max_order < 0:
        raise ValueError(f"max_order must not be negative, got {max_order!r}")
    span = range(-max_order, max_order + 1)
    grid = np.asarray(list(itertools.product(span, repeat=n_dims)), dtype=np.int64)
    total = np.abs(grid).sum(axis=1)
    if order_mode == "total":
        grid, total = grid[total <= max_order], total[total <= max_order]
    order = np.lexsort((*grid.T[::-1], total))
    return grid[order]


def _reflection_table(reflection: float | Sequence[float], n_dims: int) -> NDArray[np.float64]:
    """One reflection coefficient per wall, in the order of :data:`WALL_NAMES`."""
    values = np.atleast_1d(np.asarray(reflection, dtype=np.float64)).ravel()
    if values.size == 1:
        values = np.full(2 * n_dims, values[0])
    if values.size != 2 * n_dims:
        raise ValueError(
            f"reflection needs one value or one per wall ({2 * n_dims}), got {values.size}"
        )
    if np.any(np.abs(values) > 1) or not np.all(np.isfinite(values)):
        raise ValueError(f"reflection coefficients must lie in [-1, 1], got {values.tolist()}")
    return values


def image_source_rir(
    room_m: ArrayLike,
    source_m: ArrayLike,
    microphone_m: ArrayLike,
    *,
    sample_rate: int,
    n_taps: int = 1000,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
    max_order: int = 1,
    order_mode: str = "per_axis",
    reflection: float | Sequence[float] = 1.0,
    delay: str = "integer",
    normalize: bool = True,
) -> ImageSourceResult:
    """Simulate the impulse response from a source to a microphone in a rectangular room.

    Parameters
    ----------
    room_m : array_like, shape (2,) or (3,)
        Side lengths in meters; two sides give a planar room.
    source_m, microphone_m : array_like
        Positions in meters, strictly inside the room.
    sample_rate : int
        Sampling rate in Hz.
    n_taps : int, optional
        Length of the response; later arrivals are left out.
    speed_of_sound_m_s : float, optional
        Speed of sound.
    max_order : int, optional
        Image lattice truncation (see :func:`lattice_indices`).
    order_mode : {"per_axis", "total"}, optional
        How ``max_order`` truncates the lattice.
    reflection : float or sequence of float, optional
        Pressure reflection coefficient of all walls, or one per wall in the
        order of :data:`WALL_NAMES` (four for a planar room, six otherwise).
    delay : {"integer", "fractional"}, optional
        Round arrivals to whole samples, or keep sub-sample timing with a
        Hann-windowed sinc.
    normalize : bool, optional
        Scale the response (and the amplitudes) to a peak magnitude of one.

    Returns
    -------
    ImageSourceResult

    Raises
    ------
    GeometryError
        If the room or a position is impossible (see :func:`validate_scene`).
    ValueError
        For an unknown mode or invalid numeric argument.

    References
    ----------
    .. [1] J. B. Allen and D. A. Berkley, "Image method for efficiently
       simulating small-room acoustics," The Journal of the Acoustical Society
       of America, vol. 65, no. 4, pp. 943-950, 1979.
    .. [2] P. M. Peterson, "Simulating the response of multiple microphones to
       a single acoustic source in a reverberant room," The Journal of the
       Acoustical Society of America, vol. 80, no. 5, pp. 1527-1529, 1986.

    Examples
    --------
    The direct sound of a source 1 m away arrives after 1 / 343 s, i.e. at
    sample 47 of a 16 kHz response, and four wall plus four corner images
    follow:

    >>> result = image_source_rir([4.0, 3.0], [2.0, 1.0], [2.0, 2.0], sample_rate=16_000)
    >>> int(result.taps[0]), len(result.taps), result.labels()[:3]
    (47, 9, ['direct', 'x_min', 'y_min'])
    >>> float(result.rir.max()), int(np.argmax(result.rir))
    (1.0, 47)
    """
    room, source, microphone = validate_scene(room_m, source_m, microphone_m)
    if delay not in DELAY_MODES:
        raise ValueError(f"delay={delay!r} is not supported; choose one of {DELAY_MODES}")
    if int(sample_rate) <= 0 or int(n_taps) <= 0:
        raise ValueError(f"sample_rate and n_taps must be positive, got {sample_rate}, {n_taps}")
    if not speed_of_sound_m_s > 0:
        raise ValueError(f"speed_of_sound_m_s must be positive, got {speed_of_sound_m_s!r}")
    n_dims = room.size
    betas = _reflection_table(reflection, n_dims)
    indices = lattice_indices(int(max_order), n_dims, order_mode)

    odd = np.mod(indices, 2)
    images = np.where(odd == 1, -source, source) + (indices + odd) * room
    distances = np.linalg.norm(images - microphone, axis=1)
    near = (np.abs(indices) + (indices < 0)) // 2
    far = np.abs(indices) - near
    hits = np.empty((indices.shape[0], 2 * n_dims), dtype=np.int64)
    hits[:, 0::2], hits[:, 1::2] = near, far
    amplitudes = np.prod(betas**hits, axis=1) / distances
    delays = distances / float(speed_of_sound_m_s)

    rate = int(sample_rate)
    positions = delays * rate
    keep = np.floor(positions + 0.5) < n_taps
    if not np.all(keep):
        _LOGGER.debug("%d images arrive after the %d-tap window", int(np.sum(~keep)), n_taps)
    indices, hits, distances, delays, amplitudes, positions = (
        array[keep] for array in (indices, hits, distances, delays, amplitudes, positions)
    )

    rir = np.zeros(int(n_taps))
    if delay == "integer":
        np.add.at(rir, np.floor(positions + 0.5).astype(np.int64), amplitudes)
    else:
        offsets = np.arange(-_SINC_HALF_WIDTH + 1, _SINC_HALF_WIDTH + 1)
        samples = np.floor(positions)[:, None].astype(np.int64) + offsets
        lag = samples - positions[:, None]
        window = 0.5 * (1.0 + np.cos(np.pi * lag / _SINC_HALF_WIDTH))
        window[np.abs(lag) >= _SINC_HALF_WIDTH] = 0.0
        values = amplitudes[:, None] * window * np.sinc(lag)
        inside = (samples >= 0) & (samples < n_taps)
        np.add.at(rir, samples[inside], values[inside])

    if normalize:
        peak = float(np.max(np.abs(rir))) if rir.size else 0.0
        if peak > 0:
            rir /= peak
            amplitudes = amplitudes / peak
    return ImageSourceResult(
        rir=rir,
        sample_rate=rate,
        image_indices=indices,
        wall_hits=hits,
        distances_m=distances,
        delays_s=delays,
        amplitudes=amplitudes,
    )
