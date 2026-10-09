"""Adaptive identification: NLMS, LMS, least squares, learning curve and misalignment."""

from __future__ import annotations

import numpy as np
import pytest

from room_geometry_inference import (
    learning_curve_db,
    least_squares_response,
    lms,
    nlms,
    normalized_misalignment_db,
    prune_small_taps,
)
from room_geometry_inference.identification import identify_response


@pytest.fixture
def system(rng):
    """White excitation through a known 32-tap response, with a little noise."""
    response = np.zeros(32)
    response[[2, 9, 20, 31]] = [1.0, -0.6, 0.35, 0.2]
    excitation = rng.standard_normal(6000)
    recording = np.convolve(excitation, response)[: excitation.size]
    recording += 1e-3 * rng.standard_normal(excitation.size)
    return excitation, recording, response


def test_nlms_identifies_a_known_response_below_minus_30_db(system):
    excitation, recording, response = system
    result = nlms(excitation, recording, n_taps=32, step=0.8)
    assert normalized_misalignment_db(response, result.weights) < -30.0
    assert result.error.shape == excitation.shape


def test_nlms_is_independent_of_the_excitation_level(system):
    excitation, recording, _ = system
    quiet = nlms(excitation, recording, n_taps=32).weights
    loud = nlms(100.0 * excitation, 100.0 * recording, n_taps=32).weights
    assert np.allclose(quiet, loud, atol=1e-8)


def test_lms_converges_with_a_stable_step(system):
    excitation, recording, response = system
    result = lms(excitation, recording, n_taps=32, step=0.01)
    assert normalized_misalignment_db(response, result.weights) < -20.0


def test_lms_rejects_a_step_beyond_the_stability_bound(system):
    excitation, recording, _ = system
    with pytest.raises(ValueError, match="stability bound"):
        lms(excitation, recording, n_taps=32, step=0.5)


def test_least_squares_is_the_converged_answer(system):
    excitation, recording, response = system
    estimate = least_squares_response(excitation, recording, n_taps=32)
    assert normalized_misalignment_db(response, estimate) < -30.0
    assert identify_response(excitation, recording, method="least_squares", n_taps=32).shape == (
        32,
    )


def test_snapshots_track_the_convergence(system):
    excitation, recording, response = system
    result = nlms(excitation, recording, n_taps=32, snapshot_every=40)
    assert result.snapshots.shape == (150, 32)
    errors = [normalized_misalignment_db(response, weights) for weights in result.snapshots]
    assert errors[-1] < errors[0]


def test_learning_curve_is_a_causal_moving_average_in_db():
    error = np.r_[np.ones(10), np.zeros(10) + 0.1]
    curve = learning_curve_db(error, window=5)
    assert curve.shape == (20,)
    assert curve[:10] == pytest.approx(np.zeros(10))
    assert curve[-1] == pytest.approx(-20.0)
    assert curve[11] > curve[-1]  # the average still remembers the large errors


def test_learning_curve_decreases_while_nlms_converges(system):
    excitation, recording, _ = system
    curve = learning_curve_db(nlms(excitation, recording, n_taps=32).error, window=80)
    assert curve[-500:].mean() < curve[:100].mean() - 20.0


def test_misalignment_handles_unequal_lengths_and_zero_reference():
    assert normalized_misalignment_db([1.0, 0.0, 0.0], [1.0]) == float("-inf")
    assert normalized_misalignment_db([1.0], [0.0, 0.1]) == pytest.approx(0.0, abs=0.05)
    with pytest.raises(ValueError, match="zero"):
        normalized_misalignment_db([0.0], [1.0])


def test_pruning_normalizes_and_zeroes_small_taps():
    pruned = prune_small_taps([0.0001, -4.0, 2.0, 0.01], 0.01)
    assert pruned.tolist() == [0.0, 1.0, 0.5, 0.0]
    assert prune_small_taps(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="relative_threshold"):
        prune_small_taps([1.0], 1.0)


@pytest.mark.parametrize(
    ("excitation", "recording", "fragment"),
    [
        (np.ones(100), np.ones(99), "same length"),
        (np.ones(10), np.ones(10), "longer than n_taps"),
        (np.zeros(100), np.ones(100), "silent"),
        (np.r_[np.ones(99), np.nan], np.ones(100), "finite"),
    ],
)
def test_invalid_signals_are_rejected(excitation, recording, fragment):
    with pytest.raises(ValueError, match=fragment):
        nlms(excitation, recording, n_taps=16)


def test_invalid_settings_are_rejected(system):
    excitation, recording, _ = system
    with pytest.raises(ValueError, match=r"\(0, 2\)"):
        nlms(excitation, recording, n_taps=8, step=2.5)
    with pytest.raises(ValueError, match="method="):
        identify_response(excitation, recording, method="rls", n_taps=8)
