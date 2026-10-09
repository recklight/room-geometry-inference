"""Empirical mode decomposition: completeness, mode separation and stopping rules."""

from __future__ import annotations

import numpy as np
import pytest

from room_geometry_inference import decompose_emd
from room_geometry_inference.emd import (
    CauchyStopping,
    FixedIterationStopping,
    ThreeThresholdStopping,
    count_zero_crossings,
    default_max_modes,
    extract_mode,
    find_extrema,
    natural_cubic_spline,
    stopping_from_name,
)


def test_imfs_plus_residual_rebuild_the_signal_exactly(rng):
    signal = rng.standard_normal(400).cumsum()
    parts = decompose_emd(signal, stopping=CauchyStopping(tolerance=0.3))
    assert np.allclose(parts.imfs.sum(axis=0) + parts.residual, signal, atol=1e-10)


def test_two_tones_and_a_trend_are_separated():
    t = np.linspace(0.0, 1.0, 1000)
    fast, slow = np.sin(2 * np.pi * 50 * t), 0.8 * np.sin(2 * np.pi * 5 * t)
    parts = decompose_emd(fast + slow + 2.0 * t)
    assert np.corrcoef(parts.imfs[0], fast)[0, 1] > 0.99
    best_slow = max(np.corrcoef(mode, slow)[0, 1] for mode in parts.imfs[1:])
    assert best_slow > 0.9


def test_the_number_of_modes_and_sifts_is_capped(rng):
    parts = decompose_emd(rng.standard_normal(256), max_modes=2, max_sift_iterations=3)
    assert parts.n_modes == 2
    assert max(parts.sift_iterations) <= 3
    assert default_max_modes(256) == 7


def test_a_trend_has_no_mode():
    assert extract_mode(np.linspace(0.0, 1.0, 50)) is None
    assert decompose_emd(np.linspace(0.0, 1.0, 50)).n_modes == 0


def test_extrema_and_zero_crossings():
    extrema = find_extrema(np.array([0.0, 2.0, 1.0, 1.0, 1.0, 3.0, 0.0]))
    assert extrema.maxima.tolist() == [1, 5] and extrema.minima.tolist() == [3]
    assert count_zero_crossings(np.array([1.0, -1.0, 0.0, -2.0, 3.0])) == 2


def test_natural_spline_interpolates_its_knots():
    knots = np.linspace(0.0, 3.0, 7)
    values = knots**2
    assert np.allclose(natural_cubic_spline(knots, values, knots), values)


def test_stopping_rules_by_name():
    assert stopping_from_name("cauchy", tolerance=0.3) == CauchyStopping(tolerance=0.3)
    assert stopping_from_name("fixed", n_iterations=4) == FixedIterationStopping(4)
    assert stopping_from_name("three_threshold") == ThreeThresholdStopping()
    with pytest.raises(ValueError, match="unknown stopping criterion"):
        stopping_from_name("forever")


@pytest.mark.parametrize(
    ("signal", "fragment"),
    [
        (np.ones((2, 5)), "one-dimensional"),
        (np.ones(3), "at least 4"),
        (np.r_[1.0, np.nan, 1, 1], "NaN"),
    ],
)
def test_invalid_signals_are_rejected(signal, fragment):
    with pytest.raises(ValueError, match=fragment):
        decompose_emd(signal)
