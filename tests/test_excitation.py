"""Synthetic excitations: level, spectra, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from room_geometry_inference import EXCITATION_KINDS, make_excitation


@pytest.mark.parametrize("kind", EXCITATION_KINDS)
def test_every_kind_has_unit_rms_and_is_reproducible(kind):
    first = make_excitation(kind, 8000, 16_000, rng=3)
    second = make_excitation(kind, 8000, 16_000, rng=3)
    assert first.shape == (8000,)
    assert np.sqrt(np.mean(first**2)) == pytest.approx(1.0)
    assert np.array_equal(first, second)


def test_white_noise_has_a_flat_spectrum_and_pink_falls_by_3_db_per_octave():
    rate = 16_000
    frequencies, white = welch(make_excitation("white", 2**16, rate, rng=0), rate, nperseg=1024)
    _, pink = welch(make_excitation("pink", 2**16, rate, rng=0), rate, nperseg=1024)
    low, high = (
        (frequencies > 400) & (frequencies < 600),
        (frequencies > 3200) & (frequencies < 4800),
    )
    assert abs(10 * np.log10(white[low].mean() / white[high].mean())) < 1.0
    # 500 Hz -> 4000 Hz is three octaves: about 9 dB for 1/f.
    assert 10 * np.log10(pink[low].mean() / pink[high].mean()) == pytest.approx(9.0, abs=1.5)


def test_the_tone_sits_at_its_frequency():
    tone = make_excitation("sine", 16_000, 16_000, rng=0, frequency_hz=800.0)
    spectrum = np.abs(np.fft.rfft(tone))
    assert int(np.argmax(spectrum)) == 800
    with pytest.raises(ValueError, match="Nyquist"):
        make_excitation("sine", 100, 1000, frequency_hz=600.0)


def test_speech_like_signal_is_colored_and_modulated():
    signal = make_excitation("speech_like", 32_000, 16_000, rng=1)
    frequencies, power = welch(signal, 16_000, nperseg=1024)
    assert power[frequencies < 3000].sum() > 10 * power[frequencies > 5000].sum()
    envelope = np.sqrt(np.convolve(signal**2, np.ones(800) / 800, mode="valid"))
    assert envelope.max() > 4 * envelope.min()


@pytest.mark.parametrize(
    ("kind", "n_samples", "rate", "fragment"),
    [("brown", 10, 100, "kind="), ("white", 1, 100, "n_samples"), ("white", 10, 0, "sample_rate")],
)
def test_invalid_requests_are_rejected(kind, n_samples, rate, fragment):
    with pytest.raises(ValueError, match=fragment):
        make_excitation(kind, n_samples, rate)
