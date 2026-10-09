"""Echo inversion: closed form, labeling, symmetry, refinement, ranking and the bearing family."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import LARGE_ROOM, SMALL_ROOM, simulate
from room_geometry_inference import (
    InversionError,
    bearing_family,
    best_estimate,
    closed_form_first_order,
    first_order_distances,
    invert_echoes,
    rank_candidates,
    room_size_error,
)
from room_geometry_inference.inversion import enumerate_assignments, offsets_from_bearing


def _distances(width, depth, mic, source):
    return first_order_distances(
        width, depth, mic[0], mic[1], source[0] - mic[0], source[1] - mic[1]
    )


def test_first_order_distances_equal_the_image_source_distances():
    result = simulate(LARGE_ROOM)
    expected = _distances(10.7, 7.5, LARGE_ROOM["microphone_m"], LARGE_ROOM["source_m"])
    assert np.allclose(np.sort(result.distances_m), np.sort(expected))


def test_corners_follow_from_walls_and_direct_path():
    """d_corner^2 = d_wall_x^2 + d_wall_y^2 - d_0^2: the corners carry no bearing information."""
    d = _distances(6.0, 4.0, (2.2, 1.3), (3.0, 2.5))
    assert d[5] ** 2 == pytest.approx(d[1] ** 2 + d[3] ** 2 - d[0] ** 2)
    assert d[8] ** 2 == pytest.approx(d[2] ** 2 + d[4] ** 2 - d[0] ** 2)


def test_closed_form_recovers_the_layout_exactly():
    a, b = offsets_from_bearing(1.3, 30.0)
    d = first_order_distances(6.0, 4.0, 2.2, 1.3, a, b)
    solution = closed_form_first_order(d[0], d[1], d[2], d[3], d[4], bearing_deg=30.0)
    assert np.allclose(solution[0], [6.0, 4.0, 2.2, 1.3])


def test_only_the_positive_branch_is_feasible():
    d = _distances(10.7, 7.5, (5.35, 3.75), (6.35, 4.75))
    branches = closed_form_first_order(*d[:5], all_branches=True)
    width, depth, mic_x, mic_y = branches.T
    feasible = (mic_x > 0) & (mic_x < width) & (mic_y > 0) & (mic_y < depth)
    assert branches.shape == (16, 4) and feasible.tolist() == [True] + [False] * 15
    with_all = invert_echoes(d, all_branches=True).iloc[0]
    assert int(with_all["branch"]) == 0


def test_assignments_are_all_ordered_choices():
    assignments = enumerate_assignments(8)
    assert assignments.shape == (1680, 4)
    assert len({tuple(row) for row in assignments.tolist()}) == 1680
    assert all(len(set(row)) == 4 for row in assignments.tolist())
    with pytest.raises(InversionError, match="at least 4"):
        enumerate_assignments(3)


@pytest.mark.parametrize("scene", [LARGE_ROOM, SMALL_ROOM], ids=["large", "small"])
def test_exact_arrivals_give_the_true_room_with_eight_equivalent_answers(scene):
    d = _distances(*scene["room_m"], scene["microphone_m"], scene["source_m"])
    best = invert_echoes(d).iloc[0]
    assert room_size_error((best["width_m"], best["depth_m"]), scene["room_m"]) < 1e-9
    assert best["source_distance_m"] == pytest.approx(np.sqrt(2.0))
    assert int(best["n_equivalent"]) == 8 and best["residual_m"] < 1e-9


def test_rounded_small_room_arrivals_give_three_point_five_by_three():
    """Arrivals rounded to 16 kHz samples: about 3.504 x 2.992 m without refinement."""
    taps = simulate(SMALL_ROOM).arrival_taps()
    estimate = best_estimate(invert_echoes(taps * 343.0 / 16_000, refine=False))
    assert estimate.size_m == pytest.approx((3.504, 2.992), abs=1e-3)


def test_refinement_does_not_increase_the_mismatch():
    taps = simulate(LARGE_ROOM).arrival_taps()
    plain = invert_echoes(taps * 343.0 / 16_000, refine=False).iloc[0]
    refined = invert_echoes(taps * 343.0 / 16_000, refine=True).iloc[0]
    assert refined["residual_m"] <= plain["residual_m"]
    assert room_size_error((refined["width_m"], refined["depth_m"]), (10.7, 7.5)) < 0.01


def test_infeasible_candidates_are_filtered():
    d = _distances(10.7, 7.5, (5.35, 3.75), (6.35, 4.75))
    table = invert_echoes(d, max_candidates=50)
    assert np.all(table["mic_x_m"] > 0) and np.all(table["mic_x_m"] < table["width_m"])
    assert np.all(table["source_y_m"] > 0) and np.all(table["source_y_m"] < table["depth_m"])
    assert np.all(table["width_m"] >= table["depth_m"])
    assert table["residual_m"].is_monotonic_increasing


def test_a_microphone_next_to_a_wall_is_not_rounded_onto_it():
    """Candidates are merged at millimeter precision but reported exactly."""
    mic, source = (0.0003, 1.0), (0.5003, 1.5)
    d = _distances(6.0, 4.0, mic, source)
    candidates = invert_echoes(d)
    best = candidates.iloc[0]
    assert (best["width_m"], best["depth_m"]) == pytest.approx((6.0, 4.0))
    assert 0 < best["mic_x_m"] < 0.001
    truth = simulate({"room_m": (6.0, 4.0), "microphone_m": mic, "source_m": source})
    ranked = rank_candidates(candidates, truth.rir, sample_rate=16_000)
    assert best_estimate(ranked).size_m == pytest.approx((6.0, 4.0))


def test_coinciding_arrivals_may_share_one_echo():
    """x_min and y_max of this layout are equally far; the merged peak serves both walls."""
    d = _distances(6.0, 4.0, (2.0, 1.0), (3.0, 2.0))
    merged = np.unique(np.round(d, 9))
    assert merged.size == 7
    best = invert_echoes(merged).iloc[0]
    assert room_size_error((best["width_m"], best["depth_m"]), (6.0, 4.0)) < 1e-6


def test_too_few_or_invalid_arrivals_raise():
    with pytest.raises(InversionError, match="at least two reflections"):
        invert_echoes([1.0, 2.0])
    with pytest.raises(InversionError, match="positive and finite"):
        invert_echoes([0.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(InversionError, match="no room layout"):
        invert_echoes([1.0, 1.0, 1.0, 1.0, 1.0])  # echoes no later than the direct sound


@pytest.mark.parametrize("metric", ["tap_match", "l1", "l2"])
def test_ranking_by_resynthesis_puts_the_true_room_first(metric):
    result = simulate(LARGE_ROOM)
    candidates = invert_echoes(result.distances_m)
    ranked = rank_candidates(candidates, result.rir, sample_rate=16_000, metric=metric)
    first = ranked.iloc[0]
    assert room_size_error((first["width_m"], first["depth_m"]), (10.7, 7.5)) < 0.01
    assert first["score"] == pytest.approx(0.0, abs=1e-9)
    assert ranked["score"].is_monotonic_increasing


def test_tap_matching_survives_one_sample_rounding_where_l1_fails():
    """At 10 kHz, rounding shifts arrivals by a sample: L1 on spikes picks a wrong room."""
    result = simulate(SMALL_ROOM, sample_rate=10_000)
    distances = result.arrival_taps() * 343.0 / 10_000
    candidates = invert_echoes(distances, refine=False)
    by_taps = best_estimate(rank_candidates(candidates, result.rir, sample_rate=10_000))
    by_l1 = best_estimate(rank_candidates(candidates, result.rir, sample_rate=10_000, metric="l1"))
    assert room_size_error(by_taps.size_m, (3.5, 3.0)) < 0.02
    assert room_size_error(by_l1.size_m, (3.5, 3.0)) > 0.5


def test_ranking_rejects_unknown_metrics_and_silence():
    candidates = invert_echoes(_distances(10.7, 7.5, (5.35, 3.75), (6.35, 4.75)))
    with pytest.raises(ValueError, match="metric="):
        rank_candidates(candidates, np.ones(10), sample_rate=16_000, metric="cosine")
    with pytest.raises(InversionError, match="silent"):
        rank_candidates(candidates, np.zeros(10), sample_rate=16_000)


def test_bearing_family_reproduces_every_arrival_at_every_bearing():
    d = _distances(10.7, 7.5, (5.35, 3.75), (6.35, 4.75))
    family = bearing_family(d, bearings_deg=np.arange(10.0, 90.0, 10.0))
    assert len(family) == 8 and float(family["residual_m"].max()) < 1e-9
    true = family[family["bearing_deg"] == 40.0]
    assert np.ptp(family["width_m"]) < 0.1  # the room barely depends on the bearing
    assert float(true["width_m"].iloc[0]) != pytest.approx(10.7, abs=1e-6)


def test_best_estimate_names_the_longer_side_width():
    estimate = best_estimate(invert_echoes(_distances(4.0, 9.0, (1.0, 2.0), (2.0, 3.0))))
    assert estimate.size_m == pytest.approx((9.0, 4.0))
    assert estimate.as_dict()["mic_xy"] == list(estimate.mic_xy)
    with pytest.raises(InversionError, match="no candidate"):
        best_estimate(invert_echoes(_distances(4.0, 9.0, (1.0, 2.0), (2.0, 3.0))).iloc[0:0])
