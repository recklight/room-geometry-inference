"""Layout symmetries and size comparison."""

from __future__ import annotations

import numpy as np
import pytest

from room_geometry_inference import (
    canonical_layouts,
    equivalent_layouts,
    image_source_rir,
    room_size_error,
)


def _response(layout):
    width, depth, mx, my, sx, sy = layout
    return image_source_rir(
        [width, depth], [sx, sy], [mx, my], sample_rate=16_000, max_order=2, n_taps=1500
    ).rir


def test_every_equivalent_layout_has_the_same_impulse_response():
    layout = [7.0, 5.0, 2.1, 1.3, 3.0, 2.2]
    reference = _response(layout)
    variants = equivalent_layouts(layout)
    assert variants.shape == (32, 6)
    for variant in variants:
        assert np.allclose(_response(variant), reference, atol=1e-12)


def test_without_transposition_there_are_sixteen():
    assert equivalent_layouts([7.0, 5.0, 2.1, 1.3, 3.0, 2.2], transpose=False).shape == (16, 6)


def test_canonical_layout_is_shared_by_a_whole_class():
    layout = [7.0, 5.0, 2.1, 1.3, 3.0, 2.2]
    keys = canonical_layouts(equivalent_layouts(layout))
    assert np.unique(keys, axis=0).shape[0] == 1
    assert keys[0][0] >= keys[0][1]


def test_different_rooms_keep_different_canonical_layouts():
    keys = canonical_layouts([[7.0, 5.0, 2.1, 1.3, 3.0, 2.2], [7.0, 5.0, 2.1, 1.3, 3.0, 2.3]])
    assert not np.array_equal(keys[0], keys[1])


def test_room_size_error_ignores_side_order_and_never_cancels():
    assert room_size_error([7.5, 10.7], [10.7, 7.5]) == 0.0
    assert room_size_error([10.9, 7.3], [10.7, 7.5]) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="cannot compare"):
        room_size_error([1.0, 2.0, 3.0], [1.0, 2.0])


def test_layout_arrays_need_six_columns():
    with pytest.raises(ValueError, match="6 columns"):
        canonical_layouts([[1.0, 2.0]])
