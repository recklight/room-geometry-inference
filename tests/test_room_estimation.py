"""The estimation chain: simulated scene, identification, inversion."""

from __future__ import annotations

import numpy as np
import pytest

from room_geometry_inference import Config, InversionError, invert_response, simulate_scene
from room_geometry_inference.room_estimation import identify_recording, simulate_truth


@pytest.fixture
def config():
    return Config().override({"synthetic.duration_s": 1.0})


def test_a_noisy_white_noise_measurement_gives_the_room_within_centimeters(config):
    scene = simulate_scene(config, rng=0)
    assert scene.misalignment_db < -25.0
    assert scene.room_error_m < 0.02
    assert scene.tap_shift == pytest.approx(0.0)
    assert scene.inversion.estimate.source_distance_m == pytest.approx(np.sqrt(2.0), abs=0.01)
    assert scene.learning_curve().shape == scene.recording.excitation.shape


def test_the_scene_is_reproducible(config):
    first = simulate_scene(config, rng=4)
    second = simulate_scene(config, rng=4)
    assert np.array_equal(first.weights, second.weights)
    assert first.inversion.candidates.equals(second.inversion.candidates)


def test_least_squares_identification_runs_the_same_chain(config):
    scene = simulate_scene(config.override({"identification.method": "least_squares"}), rng=0)
    assert scene.error.size == 0 and scene.learning_curve().size == 0
    assert scene.room_error_m < 0.02


def test_a_given_excitation_and_a_noise_free_recording(config, rng):
    excitation = rng.standard_normal(12_000)
    scene = simulate_scene(config, excitation=excitation, snr_db=None)
    assert np.array_equal(scene.recording.excitation, excitation)
    assert scene.misalignment_db < -40.0


def test_identify_recording_uses_the_configured_method(config):
    scene = simulate_scene(config, rng=1)
    weights = identify_recording(scene.recording, config)
    assert weights.shape == (config.identification.n_taps,)
    assert int(np.argmax(np.abs(weights))) == 66


def test_truth_follows_the_room_section(config):
    moved = config.override({"room.source_m": [6.35, 3.75]})
    assert simulate_truth(moved).taps[0] == round(16_000 / 343.0)


def test_an_empty_response_cannot_be_inverted(config):
    response = np.zeros(1600)
    response[100] = 1.0
    with pytest.raises(InversionError):
        invert_response(response, config)
