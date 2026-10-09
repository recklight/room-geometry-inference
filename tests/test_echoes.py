"""Echo picking, distance conversion and arrival comparison."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import LARGE_ROOM, simulate
from room_geometry_inference import pick_echoes, tap_shift_error, taps_to_distances
from room_geometry_inference.echoes import distances_to_taps


def test_peaks_find_every_first_order_arrival():
    result = simulate(LARGE_ROOM)
    assert pick_echoes(result.rir, 9).tolist() == result.arrival_taps().tolist()


def test_minimum_separation_stops_a_smeared_arrival_counting_twice():
    response = np.zeros(200)
    response[[50, 51, 52]] = [1.0, 0.95, 0.9]
    response[[120, 160]] = [0.5, 0.4]
    assert pick_echoes(response, 3, min_separation_taps=3).tolist() == [50.0, 120.0, 160.0]
    assert pick_echoes(response, 3, method="top_k").tolist() == [50.0, 51.0, 52.0]


def test_top_k_equals_the_largest_magnitudes():
    response = np.random.default_rng(0).standard_normal(300)
    expected = np.sort(np.argsort(-np.abs(response))[:5])
    assert pick_echoes(response, 5, method="top_k").tolist() == expected.tolist()


def test_relative_height_hides_weak_peaks():
    response = np.zeros(100)
    response[[10, 60]] = [1.0, 0.01]
    assert pick_echoes(response, 5, relative_height=0.05).tolist() == [10.0]
    assert pick_echoes(response, 5, relative_height=0.001).tolist() == [10.0, 60.0]


def test_spreading_compensation_prefers_distant_reflections_of_equal_strength():
    """With 1/d amplitudes, a far echo is weaker than a near one but equally important."""
    taps = np.array([40, 80, 120, 400, 800])
    response = np.zeros(1000)
    response[taps] = 40.0 / taps
    assert pick_echoes(response, 4).tolist() == [40.0, 80.0, 120.0, 400.0]
    compensated = pick_echoes(response, 5, compensate_spreading=True, relative_height=0.01)
    assert compensated.tolist() == taps.astype(float).tolist()


def test_parabolic_interpolation_finds_the_vertex():
    positions = np.arange(50)
    response = np.maximum(0.0, 1.0 - 0.1 * (positions - 20.3) ** 2)
    assert pick_echoes(response, 1, interpolate=True)[0] == pytest.approx(20.3, abs=1e-9)
    assert pick_echoes(response, 1)[0] == 20.0


def test_taps_and_distances_convert_both_ways():
    distances = taps_to_distances([0, 160, 320], 16_000, 343.0)
    assert distances.tolist() == pytest.approx([0.0, 3.43, 6.86])
    assert distances_to_taps(distances, 16_000, 343.0).tolist() == pytest.approx([0, 160, 320])
    with pytest.raises(ValueError, match="positive"):
        taps_to_distances([1], 0)


def test_tap_shift_pairs_arrivals_optimally():
    assert tap_shift_error([10, 31, 50], [10, 30, 50, 90]) == pytest.approx(1 / 3)
    assert tap_shift_error([90, 10], [10, 90]) == 0.0
    assert np.isnan(tap_shift_error([], [1, 2]))


@pytest.mark.parametrize(
    ("response", "kwargs", "fragment"),
    [
        (np.zeros((3, 3)), {}, "one-dimensional"),
        (np.r_[np.ones(4), np.inf], {}, "NaN or infinite"),
        (np.ones(10), {"method": "energy"}, "method="),
        (np.ones(10), {"n_echoes": 0}, "n_echoes"),
    ],
)
def test_invalid_inputs_are_rejected(response, kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        pick_echoes(response, **kwargs)


def test_a_silent_response_has_no_echo():
    assert pick_echoes(np.zeros(10)).size == 0
