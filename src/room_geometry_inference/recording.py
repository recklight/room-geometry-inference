"""Measurement recordings: reverberation, additive noise and two-channel WAV files.

A measurement is stored as one two-channel WAV file: channel 0 is the
excitation as it was sent to the loudspeaker (ideally a loopback capture,
so playback latency cancels) and channel 1 is the microphone. Keeping both
in one file keeps them sample-aligned, which identification requires: any
offset between the channels would shift every echo and bias every distance.

Noise is added at an exact signal-to-noise ratio. With signal power
:math:`P_s` and noise power :math:`P_n`, the noise is scaled by

.. math::

    g = \\sqrt{\\frac{P_s}{P_n \\, 10^{\\mathrm{SNR}/10}}},

so :func:`measured_snr_db` of the result equals the requested value.

Measurement files go through :mod:`scipy.io.wavfile`: :func:`speechdsp.read_wav`
averages the channels to mono, which would mix the excitation into the
microphone signal, and :func:`speechdsp.write_wav` writes mono only. Reading
accepts 8-, 16- and 32-bit integer or floating-point PCM, scales it to
``[-1, 1]`` and resamples with a polyphase filter when a different rate is
requested. A mono excitation file (for example a recording of speech to play
instead of a synthetic signal) goes through :func:`speechdsp.read_wav`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import speechdsp
from numpy.typing import ArrayLike, NDArray
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly

from .errors import DatasetError
from .excitation import make_excitation

_LOGGER = logging.getLogger(__name__)

#: Interference types accepted by :func:`add_noise` and the ``noise`` settings.
NOISE_KINDS: tuple[str, ...] = ("white", "pink", "babble", "sine")

#: Full-scale values of the integer PCM formats.
_PCM_SCALE = {"int16": 32768.0, "int32": 2147483648.0}
#: Peak level, relative to full scale, of the channels written by :func:`write_recording`.
_WRITE_PEAK = 0.9


@dataclass(frozen=True, eq=False)
class Recording:
    """An excitation and its microphone recording, sample-aligned.

    Attributes
    ----------
    excitation : numpy.ndarray, shape (n_samples,)
        Signal sent to the loudspeaker.
    microphone : numpy.ndarray, shape (n_samples,)
        Microphone signal.
    sample_rate : int
        Sampling rate in Hz.

    Examples
    --------
    >>> Recording(np.ones(8000), np.zeros(8000), 16_000).duration_s
    0.5
    """

    excitation: NDArray[np.float64]
    microphone: NDArray[np.float64]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        """Length in seconds."""
        return self.excitation.size / self.sample_rate


def reverberate(excitation: ArrayLike, response: ArrayLike) -> NDArray[np.float64]:
    """The excitation filtered by an impulse response, cut to the excitation's length.

    Examples
    --------
    >>> (reverberate([1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.25]).round(12) + 0.0).tolist()
    [0.0, 0.5, 0.25, 0.0]
    """
    x = np.asarray(excitation, dtype=np.float64).ravel()
    h = np.asarray(response, dtype=np.float64).ravel()
    if x.size == 0 or h.size == 0:
        raise ValueError("excitation and response must not be empty")
    return np.asarray(fftconvolve(x, h)[: x.size], dtype=np.float64)


def measured_snr_db(clean: ArrayLike, noisy: ArrayLike) -> float:
    """Signal-to-noise ratio of ``noisy`` with respect to ``clean``, in dB.

    Examples
    --------
    >>> round(measured_snr_db([1.0, -1.0], [1.1, -0.9]), 3)
    20.0
    """
    signal = np.asarray(clean, dtype=np.float64).ravel()
    noise = np.asarray(noisy, dtype=np.float64).ravel() - signal
    noise_power = float(np.sum(noise**2))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(np.sum(signal**2) / noise_power))


def add_noise(
    signal: ArrayLike,
    snr_db: float | None,
    *,
    kind: str = "white",
    sample_rate: int = 16_000,
    rng: np.random.Generator | int | None = None,
    frequency_hz: float = 800.0,
) -> NDArray[np.float64]:
    """Add interference at an exact signal-to-noise ratio.

    Parameters
    ----------
    signal : array_like, shape (n_samples,)
        Clean signal.
    snr_db : float or None
        Target ratio in dB; ``None`` returns the signal unchanged.
    kind : {"white", "pink", "babble", "sine"}, optional
        Interference type (see :mod:`room_geometry_inference.excitation`).
    sample_rate : int, optional
        Sampling rate in Hz (shapes ``pink``, ``babble`` and ``sine``).
    rng : numpy.random.Generator or int, optional
        Random generator or seed.
    frequency_hz : float, optional
        Tone frequency of ``sine``.

    Returns
    -------
    numpy.ndarray, shape (n_samples,)

    Examples
    --------
    >>> clean = np.sin(np.arange(4000) / 7.0)
    >>> noisy = add_noise(clean, 10.0, kind="pink", rng=0)
    >>> round(measured_snr_db(clean, noisy), 6)
    10.0
    """
    rng = np.random.default_rng(rng)
    clean = np.asarray(signal, dtype=np.float64).ravel()
    if snr_db is None:
        return clean.copy()
    if kind not in NOISE_KINDS:
        raise ValueError(f"kind={kind!r} is not supported; choose one of {list(NOISE_KINDS)}")
    if not np.isfinite(snr_db):
        raise ValueError(f"snr_db must be finite or None, got {snr_db!r}")
    signal_power = float(np.mean(clean**2))
    if signal_power == 0.0:
        raise ValueError("cannot set a signal-to-noise ratio for a silent signal")
    noise = make_excitation(kind, clean.size, sample_rate, rng=rng, frequency_hz=frequency_hz)
    gain = np.sqrt(signal_power / (float(np.mean(noise**2)) * 10.0 ** (float(snr_db) / 10.0)))
    return clean + gain * noise


def simulate_recording(
    excitation: ArrayLike,
    response: ArrayLike,
    *,
    snr_db: float | None,
    noise_kind: str = "white",
    sample_rate: int = 16_000,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Microphone signal: the excitation through the response, plus noise.

    Examples
    --------
    >>> x = np.random.default_rng(0).standard_normal(1000)
    >>> mic = simulate_recording(x, [0.0, 1.0], snr_db=None)
    >>> bool(np.allclose(mic[1:], x[:-1]))
    True
    """
    rng = np.random.default_rng(rng)
    clean = reverberate(excitation, response)
    return add_noise(clean, snr_db, kind=noise_kind, sample_rate=sample_rate, rng=rng)


def _to_float(data: NDArray[np.generic], source: Path) -> NDArray[np.float64]:
    kind = data.dtype.name
    if kind == "uint8":
        return (data.astype(np.float64) - 128.0) / 128.0
    if kind in _PCM_SCALE:
        return data.astype(np.float64) / _PCM_SCALE[kind]
    if kind in ("float32", "float64"):
        return data.astype(np.float64)
    raise DatasetError(f"{source}: unsupported WAV sample format {kind}")


def _resample(signal: NDArray[np.float64], rate: int, target: int) -> NDArray[np.float64]:
    if rate == target:
        return signal
    divisor = gcd(rate, target)
    return np.asarray(resample_poly(signal, target // divisor, rate // divisor), dtype=np.float64)


def read_recording(path: str | Path, sample_rate: int | None = None) -> Recording:
    """Read a two-channel measurement WAV file.

    Parameters
    ----------
    path : str or pathlib.Path
        WAV file with the excitation in channel 0 and the microphone in
        channel 1 (further channels are ignored).
    sample_rate : int, optional
        Resample both channels to this rate (polyphase filter).

    Returns
    -------
    Recording

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DatasetError
        If the file is not a readable WAV file, has fewer than two channels,
        or the excitation channel is silent.

    Examples
    --------
    >>> import tempfile
    >>> x = np.random.default_rng(0).standard_normal(800)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = write_recording(Path(folder) / "take.wav", x, 0.5 * x, 8000)
    ...     loaded = read_recording(path, sample_rate=16_000)
    >>> loaded.sample_rate, loaded.excitation.shape, loaded.duration_s
    (16000, (1600,), 0.1)
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"recording not found: {source}")
    try:
        rate, data = wavfile.read(source)
    except (ValueError, OSError) as exc:
        raise DatasetError(f"{source} is not a readable WAV file: {exc}") from exc
    if data.ndim != 2 or data.shape[1] < 2:
        raise DatasetError(
            f"{source} has 1 channel; a measurement needs two (0: excitation, 1: microphone)"
        )
    values = _to_float(np.asarray(data), source)
    excitation, microphone = values[:, 0], values[:, 1]
    if not np.any(excitation):
        raise DatasetError(f"{source}: the excitation channel (channel 0) is silent")
    target = int(rate) if sample_rate is None else int(sample_rate)
    if target != int(rate):
        _LOGGER.debug("resampling %s from %d Hz to %d Hz", source.name, rate, target)
    return Recording(
        excitation=_resample(excitation, int(rate), target),
        microphone=_resample(microphone, int(rate), target),
        sample_rate=target,
    )


def write_recording(
    path: str | Path, excitation: ArrayLike, microphone: ArrayLike, sample_rate: int
) -> Path:
    """Write a two-channel 16-bit measurement WAV file and return its path.

    Each channel is scaled to a peak of 0.9 of full scale on its own: the
    absolute gain of a measurement chain is unknown anyway, and the
    identified response is normalized to its peak.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = write_recording(Path(folder) / "a" / "take.wav", [0.0, 1.0], [0.5, 0.0], 8000)
    ...     rate, data = wavfile.read(path)
    >>> rate, data.tolist()
    (8000, [[0, 29490], [29490, 0]])
    """
    x = np.asarray(excitation, dtype=np.float64).ravel()
    d = np.asarray(microphone, dtype=np.float64).ravel()
    if x.size != d.size or x.size == 0:
        raise ValueError(f"channels must be non-empty and equally long, got {x.size} and {d.size}")
    columns = []
    for channel in (x, d):
        peak = float(np.max(np.abs(channel)))
        scaled = channel * (_WRITE_PEAK / peak) if peak > 0 else channel
        columns.append(np.round(scaled * 32767.0).astype(np.int16))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, int(sample_rate), np.stack(columns, axis=1))
    return destination


def load_excitation(path: str | Path, sample_rate: int) -> NDArray[np.float64]:
    """Read a mono excitation WAV file (via :func:`speechdsp.read_wav`) at a given rate.

    Multi-channel files are averaged to mono; other rates are resampled.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     target = Path(folder) / "tone.wav"
    ...     speechdsp.write_wav(target, np.full(10, 0.5), 8000)
    ...     signal = load_excitation(target, 16_000)
    >>> signal.shape
    (20,)
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"excitation file not found: {source}")
    signal, rate = speechdsp.read_wav(source)
    if not np.any(signal):
        raise DatasetError(f"{source}: the excitation is silent")
    return _resample(np.asarray(signal, dtype=np.float64), int(rate), int(sample_rate))
