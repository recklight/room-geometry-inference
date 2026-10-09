"""Image-source simulation: arrival times, amplitudes, lattices and reflection coefficients."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import LARGE_ROOM, SMALL_ROOM, simulate
from room_geometry_inference import GeometryError, image_source_rir, speed_of_sound
from room_geometry_inference.image_source import lattice_indices, validate_scene


@pytest.mark.parametrize(
    ("scene", "taps"),
    [
        (LARGE_ROOM, [66, 307, 399, 455, 545, 548, 602, 624, 675]),
        (SMALL_ROOM, [66, 104, 126, 149, 192, 215, 220, 230, 281]),
    ],
)
def test_reference_scenes_arrive_at_the_known_samples(scene, taps):
    assert simulate(scene).arrival_taps().tolist() == taps


def test_the_direct_sound_arrives_after_distance_over_speed():
    result = image_source_rir([5.0, 4.0], [3.0, 2.0], [1.0, 2.0], sample_rate=16_000)
    direct = result.labels().index("direct")
    assert result.distances_m[direct] == pytest.approx(2.0)
    assert result.taps[direct] == round(16_000 * 2.0 / 343.0)


def test_amplitudes_fall_as_one_over_distance():
    result = image_source_rir([5.0, 4.0], [3.0, 2.0], [1.0, 2.0], sample_rate=8000, normalize=False)
    assert np.allclose(result.amplitudes * result.distances_m, 1.0)


def test_reflection_coefficients_multiply_per_wall_hit():
    betas = [0.9, 0.8, 0.7, 0.6]
    result = image_source_rir(
        [5.0, 4.0], [3.0, 2.0], [1.0, 2.0], sample_rate=8000, reflection=betas, normalize=False
    )
    expected = np.prod(np.asarray(betas) ** result.wall_hits, axis=1) / result.distances_m
    assert np.allclose(result.amplitudes, expected)
    corner = result.labels().index("x_max+y_min")
    assert result.amplitudes[corner] * result.distances_m[corner] == pytest.approx(0.8 * 0.7)


def test_lattice_sizes_follow_the_truncation_rule():
    for order in range(4):
        assert lattice_indices(order, 2).shape[0] == (2 * order + 1) ** 2
        assert lattice_indices(order, 3).shape[0] == (2 * order + 1) ** 3
    assert lattice_indices(2, 2, "total").shape[0] == 13
    assert np.abs(lattice_indices(3, 3, "total")).sum(axis=1).max() == 3


def test_a_planar_room_matches_the_mid_height_slice_of_a_tall_room():
    """In-plane images of a 3-D room with a far ceiling are exactly the 2-D images."""
    planar = image_source_rir([4.0, 3.0], [1.0, 1.0], [2.5, 2.0], sample_rate=16_000)
    tall = image_source_rir(
        [4.0, 3.0, 1000.0], [1.0, 1.0, 500.0], [2.5, 2.0, 500.0], sample_rate=16_000
    )
    assert np.array_equal(planar.rir, tall.rir)


def test_image_positions_satisfy_the_mirror_formula():
    result = image_source_rir([10.0, 6.0], [2.0, 1.0], [5.0, 3.0], sample_rate=8000, max_order=3)
    for (kx, ky), distance in zip(result.image_indices.tolist(), result.distances_m, strict=True):
        x = (1 - 2 * (kx % 2)) * 2.0 + (kx + kx % 2) * 10.0
        y = (1 - 2 * (ky % 2)) * 1.0 + (ky + ky % 2) * 6.0
        assert distance == pytest.approx(np.hypot(x - 5.0, y - 3.0))


def test_fractional_delays_keep_sub_sample_timing():
    """The windowed sinc puts the energy centroid at the exact arrival time."""
    result = image_source_rir(
        [40.0, 30.0], [10.0, 10.0], [11.0, 10.0], sample_rate=16_000, delay="fractional", n_taps=200
    )
    exact = 16_000 / 343.0
    window = slice(int(exact) - 6, int(exact) + 8)
    taps = np.arange(200)[window]
    weights = result.rir[window]
    assert np.sum(taps * weights) / np.sum(weights) == pytest.approx(exact, abs=0.05)


def test_late_images_are_left_out_of_a_short_window():
    result = image_source_rir(
        [10.7, 7.5], [6.35, 4.75], [5.35, 3.75], sample_rate=16_000, n_taps=400
    )
    assert result.arrival_taps().tolist() == [66, 307, 399]
    assert result.rir.size == 400


def test_speed_of_sound_rises_with_temperature():
    assert speed_of_sound(20.0) == pytest.approx(343.2, abs=0.1)
    assert speed_of_sound(30.0) > speed_of_sound(0.0)
    with pytest.raises(ValueError, match="temperature_c"):
        speed_of_sound(-300.0)


@pytest.mark.parametrize(
    ("room", "source", "microphone", "fragment"),
    [
        ([4.0], [1.0], [2.0], "2 or 3 side lengths"),
        ([4.0, -3.0], [1.0, 1.0], [2.0, 2.0], "must be positive"),
        ([4.0, 3.0], [1.0, 1.0, 1.0], [2.0, 2.0], "needs 2 coordinates"),
        ([4.0, 3.0], [1.0, 1.0], [5.0, 2.0], "microphone at"),
        ([4.0, 3.0], [0.0, 1.0], [2.0, 2.0], "source at"),
        ([4.0, 3.0], [2.0, 2.0], [2.0, 2.0], "coincide"),
    ],
)
def test_impossible_scenes_are_rejected(room, source, microphone, fragment):
    with pytest.raises(GeometryError, match=fragment):
        validate_scene(room, source, microphone)


def test_invalid_modes_are_rejected():
    with pytest.raises(ValueError, match="delay="):
        image_source_rir([4.0, 3.0], [1.0, 1.0], [2.0, 2.0], sample_rate=8000, delay="exact")
    with pytest.raises(ValueError, match="order_mode="):
        lattice_indices(1, 2, "manhattan")
    with pytest.raises(ValueError, match="one per wall"):
        image_source_rir([4.0, 3.0], [1.0, 1.0], [2.0, 2.0], sample_rate=8000, reflection=[1, 1])
