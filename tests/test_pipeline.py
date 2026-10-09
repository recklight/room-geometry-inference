"""The high-level steps: prepare, train, predict, invert and the denoising pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from room_geometry_inference import (
    Config,
    DatasetError,
    build_model,
    denoising_pairs,
    invert_measurements,
    load_dataset_index,
    load_model,
    measurement_response,
    predict_inputs,
    prepare_features,
    save_model,
    train_model,
    write_recording,
)
from room_geometry_inference.pipeline import ESTIMATE_COLUMNS, TRUTH_COLUMNS


def test_prepare_describes_every_measurement_in_index_order(quick_features, quick_dataset):
    index = load_dataset_index(quick_dataset)
    assert quick_features.n_items == len(index)
    assert quick_features.paths.tolist() == index["source"].tolist()
    assert quick_features.X.shape == (16, 1600)
    assert quick_features.groups.tolist() == index["group"].tolist()


def test_decay_curves_separate_the_room_classes(quick_features):
    labels = quick_features.y
    level = quick_features.X[:, :800].mean(axis=1)  # mean decay over the first 50 ms
    assert level[labels == "large"].min() > level[labels == "small"].max()


def test_train_and_predict_use_the_stored_settings(
    quick_features, quick_dataset, fast_config, tmp_path
):
    estimator = train_model(quick_features, fast_config)
    bundle = load_model(save_model(tmp_path / "model.joblib", estimator, fast_config))
    inputs = [
        quick_dataset.parent / "large" / "large_000.wav",
        quick_dataset.parent / "small" / "small_000.wav",
    ]
    table = predict_inputs(inputs, bundle)
    assert table.columns.tolist() == ["path", "predicted", "score_large", "score_small"]
    assert table["predicted"].tolist() == ["large", "small"]
    with pytest.raises(DatasetError, match="no input"):
        predict_inputs([], bundle)


def test_the_template_matcher_may_answer_unknown(quick_dataset, tmp_path):
    config = Config().override(
        {
            "synthetic.duration_s": 1.0,
            "features.representation": "magnitude",
            "model.name": "template",
            "model.template_min_score": 1.0,
        }
    )
    features = prepare_features(quick_dataset, config)
    first_rooms = np.isin(features.groups, ["large_room_00", "small_room_00"])
    subset = type(features)(
        X=features.X[first_rooms],
        y=features.y[first_rooms],
        groups=features.groups[first_rooms],
        paths=features.paths[first_rooms],
        config=features.config,
    )
    bundle = {
        "config": config.to_dict(),
        "classes": subset.classes,
        "estimator": train_model(subset, config),
    }
    seen = quick_dataset.parent / "large" / "large_000.wav"
    unseen = quick_dataset.parent / "large" / "large_002.wav"
    table = predict_inputs([seen, unseen], bundle)
    assert table["predicted"].tolist() == ["large", "unknown"]


def test_training_needs_two_classes(quick_features, fast_config):
    one_class = type(quick_features)(
        X=quick_features.X[:4],
        y=np.array(["large"] * 4),
        groups=quick_features.groups[:4],
        paths=quick_features.paths[:4],
        config=quick_features.config,
    )
    with pytest.raises(DatasetError, match="two classes"):
        train_model(one_class, fast_config)
    assert build_model(fast_config) is not None


def test_inversion_of_the_dataset_matches_the_true_rooms(quick_dataset, fast_config):
    index = load_dataset_index(quick_dataset, require_label=False)
    table = invert_measurements(index["path"].tolist(), fast_config, index=index)
    assert table.columns.tolist() == [*ESTIMATE_COLUMNS, *TRUTH_COLUMNS]
    assert (table["status"] == "ok").all()
    assert table["room_error_m"].median() < 0.02
    assert table["path"].tolist() == index["source"].tolist()


def test_inversion_of_bare_files_has_no_truth_columns(quick_dataset, fast_config):
    files = [quick_dataset.parent / "small" / "small_001.wav"]
    table = invert_measurements(files, fast_config)
    assert table.columns.tolist() == list(ESTIMATE_COLUMNS)
    assert table.loc[0, "width_m"] == pytest.approx(3.7, abs=0.5)


def test_a_measurement_without_echoes_is_reported_not_raised(tmp_path, fast_config, rng):
    excitation = rng.standard_normal(16_000)
    path = write_recording(tmp_path / "dry.wav", excitation, np.roll(excitation, 30), 16_000)
    table = invert_measurements([path], fast_config)
    assert table["status"].tolist() == ["no_solution"] and np.isnan(table.loc[0, "width_m"])


def test_denoising_pairs_need_the_geometry(quick_dataset, fast_config, tmp_path):
    noisy, clean, groups = denoising_pairs(quick_dataset, fast_config)
    assert noisy.shape == clean.shape == (16, 1600)
    assert np.allclose(clean.max(axis=1), 1.0) and np.count_nonzero(clean[0]) == 9
    assert groups.tolist() == pd.read_csv(quick_dataset)["group"].tolist()
    stripped = pd.read_csv(quick_dataset)[["path", "label", "group"]]
    stripped.to_csv(quick_dataset.parent / "plain.csv", index=False)
    with pytest.raises(DatasetError, match="no geometry columns"):
        denoising_pairs(quick_dataset.parent / "plain.csv", fast_config)


def test_short_measurements_are_rejected(tmp_path, fast_config, rng):
    path = write_recording(
        tmp_path / "short.wav", rng.standard_normal(1000), rng.standard_normal(1000), 16_000
    )
    with pytest.raises(DatasetError, match=r"not more than identification\.n_taps"):
        measurement_response(path, fast_config)
