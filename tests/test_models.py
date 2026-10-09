"""Classifier registry and model files."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from room_geometry_inference import (
    MODEL_NAMES,
    Config,
    ModelFileError,
    build_model,
    load_model,
    save_model,
)
from room_geometry_inference.models import MODEL_FORMAT, fit_estimator


def _config(name):
    representation = "magnitude" if name == "template" else "decay_curve"
    return Config().override({"model.name": name, "features.representation": representation})


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_is_a_pipeline_with_probabilities(name, rng):
    model = build_model(_config(name))
    assert hasattr(model, "steps")
    X = np.zeros((12, 200))
    labels = np.repeat(["large", "small"], 6)
    for row, label in enumerate(labels):
        taps = [10, 60, 120] if label == "small" else [10, 90, 180]
        X[row, taps] = [1.0, 0.5, 0.4]
        X[row] += 0.001 * rng.random(200)
    fitted = fit_estimator(model, X, labels)
    probabilities = fitted.predict_proba(X)
    assert probabilities.shape == (12, 2) and np.allclose(probabilities.sum(axis=1), 1.0)
    assert (fitted.predict(X) == labels).all()


def test_the_template_matcher_takes_every_echo_setting():
    config = _config("template").override(
        {"echoes.interpolate": True, "echoes.min_separation_taps": 5}
    )
    matcher = build_model(config)[-1]
    assert matcher.interpolate is True and matcher.min_separation_taps == 5
    assert matcher.compensate_spreading == config.echoes.compensate_spreading


def test_the_seed_reaches_the_estimators():
    model = build_model(Config().override({"seed": 9}))
    assert model[-1].random_state == 9


def test_model_files_round_trip(tmp_path, rng):
    config = _config("knn")
    X, y = rng.standard_normal((10, 3)), np.repeat(["large", "small"], 5)
    estimator = fit_estimator(build_model(config), X, y)
    path = save_model(tmp_path / "out" / "model.joblib", estimator, config)
    bundle = load_model(path)
    assert bundle["format"] == MODEL_FORMAT and bundle["classes"] == ["large", "small"]
    assert Config.from_dict(bundle["config"]) == config
    assert np.array_equal(bundle["estimator"].predict(X), estimator.predict(X))


def test_foreign_and_incomplete_model_files_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'train' first"):
        load_model(tmp_path / "absent.joblib")
    joblib.dump({"format": "other/model-v1"}, tmp_path / "foreign.joblib")
    with pytest.raises(ModelFileError, match="not written by this package"):
        load_model(tmp_path / "foreign.joblib")
    joblib.dump({"format": MODEL_FORMAT}, tmp_path / "partial.joblib")
    with pytest.raises(ModelFileError, match="incomplete"):
        load_model(tmp_path / "partial.joblib")
    (tmp_path / "garbage.joblib").write_bytes(b"not a pickle")
    with pytest.raises(ModelFileError, match="not a readable model file"):
        load_model(tmp_path / "garbage.joblib")
