"""Cross-validation: grouping, out-of-fold coverage, metrics and reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from room_geometry_inference import Config, ConfigError, DatasetError, cross_validate
from room_geometry_inference.evaluation import classification_metrics, summarize


@pytest.fixture
def config():
    return Config().override({"model.name": "knn", "evaluation.n_splits": 3})


def test_folds_never_split_a_room(make_toy, rng, config):
    result = cross_validate(make_toy(rng), config)
    folds_per_room = result.predictions.groupby("group")["fold"].nunique()
    assert (folds_per_room == 1).all()


def test_every_item_gets_exactly_one_out_of_fold_prediction(make_toy, rng, config):
    data = make_toy(rng)
    result = cross_validate(data, config)
    assert len(result.predictions) == data.n_items
    assert result.predictions["path"].tolist() == data.paths.tolist()
    assert sorted(result.predictions["fold"].unique().tolist()) == [0, 1, 2]
    assert result.folds["n_test"].sum() == data.n_items
    assert result.predictions.columns.tolist() == [
        "path",
        "group",
        "fold",
        "label",
        "predicted",
        "score_large",
        "score_small",
    ]


def test_metrics_agree_with_scikit_learn(make_toy, rng, config):
    result = cross_validate(make_toy(rng, separation=1.0), config)
    table = result.predictions
    pooled = result.metrics["pooled"]
    assert pooled["uar"] == pytest.approx(
        balanced_accuracy_score(table["label"], table["predicted"])
    )
    assert pooled["f1_macro"] == pytest.approx(
        f1_score(table["label"], table["predicted"], average="macro")
    )
    assert pooled["roc_auc"] == pytest.approx(
        roc_auc_score(table["label"] == "large", table["score_large"])
    )
    recall_large = np.mean(table.loc[table["label"] == "large", "predicted"] == "large")
    assert pooled["sensitivity"] == pytest.approx(recall_large)
    assert np.array(pooled["confusion_matrix"]).sum() == len(table)


def test_summary_uses_the_sample_standard_deviation(make_toy, rng, config):
    result = cross_validate(make_toy(rng, separation=1.0), config)
    uar = result.folds["uar"].to_numpy()
    assert result.metrics["summary"]["uar"]["std"] == pytest.approx(np.std(uar, ddof=1))
    assert summarize([0.7, 0.8, 0.9, float("nan")]) == {"mean": 0.8, "std": 0.1}


def test_metrics_json_keys_are_fixed(make_toy, rng, config):
    metrics = cross_validate(make_toy(rng), config).metrics
    assert list(metrics) == ["task", "n_items", "n_splits", "seed", "folds", "summary", "pooled"]
    assert list(metrics["summary"]) == [
        "uar",
        "accuracy",
        "f1_macro",
        "sensitivity",
        "specificity",
        "roc_auc",
    ]
    assert set(metrics["pooled"]) >= {"classes", "confusion_matrix", "per_class"}


def test_the_same_seed_gives_the_same_folds(make_toy, rng, config):
    data = make_toy(rng)
    first = cross_validate(data, config.override({"model.name": "svm", "seed": 3}))
    second = cross_validate(data, config.override({"model.name": "svm", "seed": 3}))
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


def test_without_groups_the_split_is_stratified_by_item(make_toy, rng, config):
    result = cross_validate(make_toy(rng, with_groups=False), config)
    per_fold = result.predictions.groupby("fold")["label"].value_counts().unstack()
    assert (per_fold.min(axis=1) > 0).all()


def test_three_classes_report_no_binary_metrics(make_toy, rng, config):
    data = make_toy(rng, classes=("a", "b", "c"))
    metrics = cross_validate(data, config).metrics
    assert list(metrics["summary"]) == ["uar", "accuracy", "f1_macro"]


def test_impossible_splits_and_wrong_positive_class_are_rejected(make_toy, rng, config):
    with pytest.raises(DatasetError, match="groups"):
        cross_validate(make_toy(rng, rooms_per_class=1), config)
    with pytest.raises(DatasetError, match="two classes"):
        cross_validate(make_toy(rng, classes=("large",)), config)
    with pytest.raises(ConfigError, match="positive_class"):
        cross_validate(make_toy(rng, classes=("a", "b")), config)


def test_binary_metrics_are_undefined_for_a_single_class_set():
    metrics = classification_metrics(
        ["a", "a"], ["a", "b"], [[0.6, 0.4], [0.4, 0.6]], classes=["a", "b"], positive_class="a"
    )
    assert np.isnan(metrics["roc_auc"]) and np.isnan(metrics["specificity"])
