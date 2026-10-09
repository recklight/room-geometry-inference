"""Feature representations of identified responses."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import LARGE_ROOM, SMALL_ROOM, simulate
from room_geometry_inference import (
    REPRESENTATIONS,
    Config,
    decay_curve_db,
    represent_response,
    response_features,
)
from room_geometry_inference.rir_features import echo_delay_features, imf_energy_features


def test_decay_curve_is_non_increasing_and_starts_at_zero_db():
    curve = decay_curve_db(simulate(LARGE_ROOM).rir)
    assert curve[0] == pytest.approx(0.0)
    assert np.all(np.diff(curve) <= 1e-12)
    assert curve.min() >= -60.0


def test_decay_curve_steps_by_each_echo_energy():
    response = np.zeros(10)
    response[[2, 6]] = [1.0, 1.0]
    curve = decay_curve_db(response, floor_db=-30.0)
    assert curve[:3].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert curve[3:7].tolist() == pytest.approx([-3.0103] * 4, abs=1e-4)
    assert curve[7:].tolist() == [-30.0] * 3


def test_larger_rooms_decay_later():
    large = decay_curve_db(simulate(LARGE_ROOM).rir)
    small = decay_curve_db(simulate(SMALL_ROOM).rir)
    assert np.argmax(large <= -10.0) > np.argmax(small <= -10.0)


def test_echo_delays_are_in_milliseconds_and_padded():
    response = simulate(SMALL_ROOM).rir
    delays = echo_delay_features(response, 16_000, n_echoes=9)
    assert delays[0] == pytest.approx(66 / 16.0)
    padded = echo_delay_features(response[:150], 16_000, n_echoes=9)
    assert padded[-1] == pytest.approx(150 / 16.0)


def test_echo_delays_follow_the_interpolate_setting():
    response = np.zeros(200)
    response[[40, 41, 42]] = [0.5, 1.0, 0.9]  # parabola vertex: 1/3 sample after tap 41
    base = Config().override({"features.representation": "echo_delays"})
    whole = response_features(response, base)
    refined = response_features(response, base.override({"echoes.interpolate": True}))
    assert whole[0] == pytest.approx(41 / 16.0)
    assert refined[0] == pytest.approx((41 + 1 / 3) / 16.0)


def test_imf_features_describe_energy_shares():
    t = np.linspace(0.0, 1.0, 500)
    features = imf_energy_features(np.sin(2 * np.pi * 40 * t), max_imfs=3)
    assert features.shape == (7,)
    assert features[:3].sum() + features[-1] == pytest.approx(1.0, abs=0.05)
    assert features[3] == pytest.approx(80 / 500, abs=0.01)
    assert imf_energy_features(np.zeros(20), max_imfs=2).tolist() == [0.0] * 5


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_every_representation_has_a_fixed_length(representation):
    first = represent_response(simulate(LARGE_ROOM).rir, representation)
    second = represent_response(simulate(SMALL_ROOM).rir, representation)
    assert first.shape == second.shape and np.all(np.isfinite(first))


def test_unknown_representations_are_rejected():
    with pytest.raises(ValueError, match="representation="):
        represent_response(np.ones(10), "cepstrum")
    with pytest.raises(ValueError, match="floor_db"):
        decay_curve_db(np.ones(3), floor_db=0.0)
