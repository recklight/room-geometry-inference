"""Echo-pattern template matching with a tolerance and an unknown answer."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from room_geometry_inference import TemplateMatcher
from room_geometry_inference.template_matching import UNKNOWN_LABEL, match_rate


def _response(taps, length=400):
    response = np.zeros(length)
    response[list(taps)] = 1.0 / (1.0 + np.arange(len(taps)))
    return response


@pytest.fixture
def matcher():
    X = np.stack([_response([20, 80, 140]), _response([22, 81, 139]), _response([20, 200, 330])])
    return TemplateMatcher(n_echoes=3, tolerance_taps=2, min_score=0.7).fit(
        X, ["small", "small", "large"]
    )


def test_match_rate_counts_arrivals_within_the_tolerance():
    assert match_rate([10, 50, 90, 130], [11, 52, 200], tolerance_taps=1) == 0.25
    assert match_rate([10, 50], [10, 50], tolerance_taps=0) == 1.0
    assert match_rate([], [1]) == 0.0


def test_exact_matching_fails_on_a_one_sample_shift_but_the_tolerance_does_not():
    shifted = _response([21, 81, 141])[None, :]
    exact = TemplateMatcher(n_echoes=3, tolerance_taps=0).fit(
        _response([20, 80, 140])[None, :], ["small"]
    )
    tolerant = TemplateMatcher(n_echoes=3, tolerance_taps=1).fit(
        _response([20, 80, 140])[None, :], ["small"]
    )
    assert exact.class_scores(shifted)[0, 0] == 0.0
    assert tolerant.class_scores(shifted)[0, 0] == 1.0


def test_predictions_and_probabilities(matcher):
    X = np.stack([_response([21, 79, 141]), _response([19, 201, 331])])
    assert matcher.predict(X).tolist() == ["small", "large"]
    probabilities = matcher.predict_proba(X)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert matcher.classes_.tolist() == ["large", "small"]


def test_poor_matches_are_unknown(matcher):
    X = np.stack([_response([20, 120, 260]), _response([21, 80, 140])])
    assert matcher.predict_with_unknown(X).tolist() == [UNKNOWN_LABEL, "small"]


def test_no_match_at_all_gives_uniform_probabilities(matcher):
    probabilities = matcher.predict_proba(_response([60, 250, 390])[None, :])
    assert probabilities.tolist() == [[0.5, 0.5]]


def test_the_matcher_clones_and_validates_input(matcher):
    assert clone(matcher).get_params()["tolerance_taps"] == 2
    with pytest.raises(ValueError, match="two-dimensional"):
        matcher.predict(np.zeros(10))
    with pytest.raises(ValueError, match="labels"):
        TemplateMatcher().fit(np.zeros((2, 10)), ["a"])
