"""Synthetic test signals: excitations for identification and interfering noise.

Every signal has unit root-mean-square value and is drawn from the given
random generator, so a seed reproduces it.

* ``white``: Gaussian white noise, the ideal excitation. Its flat spectrum
  excites every frequency and its autocorrelation is a delta, which gives the
  adaptive filter its fastest convergence.
* ``pink``: noise with a :math:`1/f` power spectrum, shaped in the frequency
  domain; a colored but broadband excitation.
* ``speech_like``: white noise through three random formant resonances (an
  all-pole, autoregressive spectrum envelope like a vowel's) with a
  syllable-rate (3-6 Hz) amplitude modulation. It stands in for running
  speech: strongly colored and non-stationary, the hard case for NLMS.
* ``babble``: the sum of six independent ``speech_like`` talkers, used as
  multi-talker interference.
* ``sine``: a pure tone (800 Hz by default), used as narrowband interference.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

#: Signal kinds accepted by :func:`make_excitation`.
EXCITATION_KINDS: tuple[str, ...] = ("white", "pink", "speech_like", "babble", "sine")

#: Nominal formant frequencies (Hz) of the speech-like signal, each jittered by 15 %.
_FORMANTS_HZ = (500.0, 1500.0, 2500.0)
#: Pole radius of the formant resonators.
_FORMANT_RADIUS = 0.97
#: Talkers summed into the babble signal.
_BABBLE_TALKERS = 6


def _unit_rms(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    rms = float(np.sqrt(np.mean(signal**2)))
    return signal / rms if rms > 0 else signal


def _white(n: int, rate: int, rng: np.random.Generator, frequency_hz: float) -> NDArray[np.float64]:
    return rng.standard_normal(n)


def _pink(n: int, rate: int, rng: np.random.Generator, frequency_hz: float) -> NDArray[np.float64]:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    frequencies = np.fft.rfftfreq(n, d=1.0 / rate)
    scale = np.zeros_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    return np.fft.irfft(spectrum * scale, n)


def _speech_like(
    n: int, rate: int, rng: np.random.Generator, frequency_hz: float
) -> NDArray[np.float64]:
    denominator = np.ones(1)
    for nominal in _FORMANTS_HZ:
        center = min(nominal * rng.uniform(0.85, 1.15), 0.45 * rate)
        angle = 2.0 * np.pi * center / rate
        pole_pair = [1.0, -2.0 * _FORMANT_RADIUS * np.cos(angle), _FORMANT_RADIUS**2]
        denominator = np.convolve(denominator, pole_pair)
    shaped = lfilter([1.0], denominator, rng.standard_normal(n))
    time = np.arange(n) / rate
    syllable_rate = rng.uniform(3.0, 6.0)
    envelope = np.sin(np.pi * syllable_rate * time + rng.uniform(0.0, np.pi)) ** 2
    return shaped * (0.05 + envelope)


def _babble(
    n: int, rate: int, rng: np.random.Generator, frequency_hz: float
) -> NDArray[np.float64]:
    talkers = [_unit_rms(_speech_like(n, rate, rng, frequency_hz)) for _ in range(_BABBLE_TALKERS)]
    return np.sum(talkers, axis=0)


def _sine(n: int, rate: int, rng: np.random.Generator, frequency_hz: float) -> NDArray[np.float64]:
    if not 0 < frequency_hz < rate / 2:
        raise ValueError(
            f"frequency_hz must lie between 0 and the Nyquist frequency {rate / 2:g}, "
            f"got {frequency_hz!r}"
        )
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return np.sin(2.0 * np.pi * frequency_hz * np.arange(n) / rate + phase)


_GENERATORS: dict[str, Callable[[int, int, np.random.Generator, float], NDArray[np.float64]]] = {
    "white": _white,
    "pink": _pink,
    "speech_like": _speech_like,
    "babble": _babble,
    "sine": _sine,
}


def make_excitation(
    kind: str,
    n_samples: int,
    sample_rate: int,
    *,
    rng: np.random.Generator | int | None = None,
    frequency_hz: float = 800.0,
) -> NDArray[np.float64]:
    """A synthetic signal of unit root-mean-square value.

    Parameters
    ----------
    kind : {"white", "pink", "speech_like", "babble", "sine"}
        Signal type (see the module documentation).
    n_samples : int
        Length in samples.
    sample_rate : int
        Sampling rate in Hz.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.
    frequency_hz : float, optional
        Tone frequency of ``sine``.

    Returns
    -------
    numpy.ndarray, shape (n_samples,)

    Examples
    --------
    >>> x = make_excitation("pink", 16_000, 16_000, rng=0)
    >>> x.shape, round(float(np.sqrt(np.mean(x**2))), 6)
    ((16000,), 1.0)
    >>> sorted(EXCITATION_KINDS) == sorted(_GENERATORS)
    True
    """
    rng = np.random.default_rng(rng)
    if kind not in _GENERATORS:
        raise ValueError(f"kind={kind!r} is not supported; choose one of {list(EXCITATION_KINDS)}")
    if int(n_samples) < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples!r}")
    if int(sample_rate) <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    signal = _GENERATORS[kind](int(n_samples), int(sample_rate), rng, float(frequency_hz))
    return _unit_rms(np.asarray(signal, dtype=np.float64))
