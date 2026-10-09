"""Grouped, stratified k-fold cross-validation of the room classifier.

Folds are drawn with :class:`~sklearn.model_selection.StratifiedGroupKFold`
when the index has a ``group`` column (the room) and with
:class:`~sklearn.model_selection.StratifiedKFold` otherwise, both shuffled
with ``random_state = seed`` (Kohavi, 1995). With groups, every measurement
of a room stays on one side of each split, so the classifier is always
tested on rooms it has never seen, which is how it would be used. A fresh
model is fitted on the training folds only, so its scaler never sees the
test fold, and every measurement receives exactly one out-of-fold
prediction.

The main metric is the unweighted average recall (UAR; Schuller et al.,
2009), the mean of the per-class recalls. Unlike accuracy, it is not
inflated by predicting the larger class. Accuracy and macro F1 are reported
alongside; with two classes also the sensitivity (recall of
``evaluation.positive_class``), the specificity and the ROC-AUC (Fawcett,
2006). Fold scores are summarized by their mean and sample standard
deviation (``ddof=1``); the confusion matrix and per-class figures are
computed on the pooled out-of-fold predictions.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import speechdsp
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .errors import ConfigError, DatasetError
from .models import build_model, fit_estimator

if TYPE_CHECKING:
    from .config import Config
    from .dataset import FeatureSet

_LOGGER = logging.getLogger(__name__)

#: Scalar metrics of every classification, in report order.
CLASSIFICATION_METRICS: tuple[str, ...] = ("uar", "accuracy", "f1_macro")
#: Extra scalar metrics of two-class problems.
BINARY_METRICS: tuple[str, ...] = ("sensitivity", "specificity", "roc_auc")


@dataclass(frozen=True, eq=False)
class CrossValidationResult:
    """Out-of-fold predictions, per-fold metrics and their summary.

    Attributes
    ----------
    predictions : pandas.DataFrame
        One row per measurement in input order (``predictions.csv``): ``path``,
        ``group``, ``fold``, ``label``, ``predicted`` and ``score_<class>``.
    folds : pandas.DataFrame
        One row per fold (``folds.csv``).
    metrics : dict
        Contents of ``metrics.json``.
    """

    predictions: pd.DataFrame
    folds: pd.DataFrame
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """The ``metrics.json`` mapping."""
        return self.metrics

    @property
    def classes(self) -> list[str]:
        """Class names in sorted order."""
        return list(self.metrics["pooled"]["classes"])


def metric_names(*, binary: bool) -> tuple[str, ...]:
    """Scalar metric names, in report order.

    Examples
    --------
    >>> metric_names(binary=True)
    ('uar', 'accuracy', 'f1_macro', 'sensitivity', 'specificity', 'roc_auc')
    """
    return CLASSIFICATION_METRICS + (BINARY_METRICS if binary else ())


def score_columns(classes: Sequence[str]) -> list[str]:
    """Names of the probability columns, ``score_<class>``.

    Examples
    --------
    >>> score_columns(["large", "small"])
    ['score_large', 'score_small']
    """
    return [f"score_{name}" for name in classes]


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    scores: ArrayLike | None = None,
    *,
    classes: Sequence[str],
    positive_class: str | None = None,
) -> dict[str, float]:
    """UAR, accuracy and macro F1, plus sensitivity, specificity and ROC-AUC for two classes.

    Parameters
    ----------
    y_true, y_pred : array_like of str, shape (n,)
        True and predicted class names.
    scores : array_like, shape (n, n_classes), optional
        Class probabilities in the order of ``classes`` (needed for ROC-AUC).
    classes : sequence of str
        All class names, sorted.
    positive_class : str, optional
        The class whose recall is the sensitivity (two classes only).

    Returns
    -------
    dict
        Keys of :func:`metric_names`; the UAR is :func:`speechdsp.uar`, the
        sensitivity and specificity come from
        :func:`speechdsp.sensitivity_specificity`; ``roc_auc`` is ``nan``
        without scores or when the set holds a single class.

    References
    ----------
    .. [1] B. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009
       Emotion Challenge," in Proc. Interspeech 2009, pp. 312-315.
    .. [2] T. Fawcett, "An introduction to ROC analysis," Pattern Recognition
       Letters, vol. 27, no. 8, pp. 861-874, 2006.

    Examples
    --------
    >>> truth = ["large", "large", "small", "small"]
    >>> guess = ["large", "small", "small", "small"]
    >>> probabilities = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]]
    >>> metrics = classification_metrics(
    ...     truth, guess, probabilities, classes=["large", "small"], positive_class="large")
    >>> [round(value, 3) for value in metrics.values()]
    [0.75, 0.75, 0.733, 0.5, 1.0, 1.0]
    """
    truth = np.asarray(y_true).astype(str)
    predicted = np.asarray(y_pred).astype(str)
    labels = [str(name) for name in classes]
    present = np.unique(truth)
    with warnings.catch_warnings():
        # scikit-learn warns about a single-class confusion matrix; the fold
        # warning in cross_validate already reports that situation.
        warnings.simplefilter("ignore", UserWarning)
        uar = float(speechdsp.uar(truth, predicted))
    result = {
        "uar": uar,
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }
    if len(labels) == 2 and positive_class is not None:
        if positive_class not in labels:
            raise ConfigError(
                f"evaluation.positive_class={positive_class!r} is not one of the classes {labels}"
            )
        sensitivity = specificity = auc = math.nan
        if present.size == 2:
            sensitivity, specificity = speechdsp.sensitivity_specificity(
                truth, predicted, pos_label=positive_class
            )
            if scores is not None:
                probabilities = np.asarray(scores, dtype=np.float64)
                auc = float(
                    roc_auc_score(
                        truth == positive_class, probabilities[:, labels.index(positive_class)]
                    )
                )
        result.update(sensitivity=float(sensitivity), specificity=float(specificity), roc_auc=auc)
    return result


def per_class_table(
    y_true: ArrayLike, y_pred: ArrayLike, classes: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    """Recall, precision, F1 and support of every class.

    Examples
    --------
    >>> per_class_table(["a", "b", "b"], ["a", "b", "a"], ["a", "b"])["b"]
    {'recall': 0.5, 'precision': 1.0, 'f1': 0.6666666666666666, 'support': 2}
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        np.asarray(y_true).astype(str),
        np.asarray(y_pred).astype(str),
        labels=list(classes),
        zero_division=0,
    )
    return {
        name: {
            "recall": float(recall[i]),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(classes)
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample standard deviation (``ddof=1``) of the finite values.

    Examples
    --------
    >>> summarize([0.7, 0.8, 0.9, float("nan")])
    {'mean': 0.8, 'std': 0.1}
    """
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan}
    mean = round(float(finite.mean()), 12)
    std = round(float(finite.std(ddof=1)), 12) if finite.size > 1 else math.nan
    return {"mean": mean, "std": std}


def class_scores(estimator: Any, X: ArrayLike, classes: Sequence[str]) -> NDArray[np.float64]:
    """Class probabilities aligned to ``classes`` (zero for a class the model never saw)."""
    probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
    fitted = [str(name) for name in estimator.classes_]
    aligned = np.zeros((probabilities.shape[0], len(classes)))
    for column, name in enumerate(classes):
        if name in fitted:
            aligned[:, column] = probabilities[:, fitted.index(name)]
    return aligned


def _folds(
    labels: NDArray[np.str_], groups: NDArray[np.str_], n_splits: int, seed: int
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        raise DatasetError(f"cross-validation needs at least two classes, found {classes.tolist()}")
    grouped = bool(np.any(groups != ""))
    if grouped:
        frame = pd.DataFrame({"label": labels, "group": groups}).drop_duplicates()
        per_class = frame.groupby("label")["group"].nunique()
        n_groups = int(np.unique(groups).size)
        if n_groups < n_splits:
            raise DatasetError(
                f"only {n_groups} groups for evaluation.n_splits={n_splits}; every fold needs at "
                "least one group, so lower evaluation.n_splits or add rooms"
            )
        sparse = per_class[per_class < n_splits]
        if not sparse.empty:
            _LOGGER.warning(
                "classes with fewer rooms than folds will be missing from some test folds: %s",
                ", ".join(f"{name} ({count})" for name, count in sparse.items()),
            )
        splitter: Any = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(labels.size), labels, groups))
    if counts.min() < n_splits:
        raise DatasetError(
            f"the smallest class has {int(counts.min())} items, fewer than "
            f"evaluation.n_splits={n_splits}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(labels.size), labels))


def cross_validate(features: FeatureSet, config: Config) -> CrossValidationResult:
    """Stratified (and, with groups, grouped) k-fold cross-validation of the configured model.

    Parameters
    ----------
    features : FeatureSet
        Measurements written by ``prepare``.
    config : Config
        Complete configuration; ``evaluation``, ``seed`` and ``model`` are used.

    Returns
    -------
    CrossValidationResult

    Raises
    ------
    DatasetError
        With fewer than two classes, or too few groups or items for the folds.
    ConfigError
        If ``evaluation.positive_class`` is not one of two classes.

    References
    ----------
    .. [1] R. Kohavi, "A study of cross-validation and bootstrap for accuracy
       estimation and model selection," in Proc. 14th International Joint
       Conference on Artificial Intelligence (IJCAI), 1995, pp. 1137-1143.
    """
    X = np.asarray(features.X, dtype=np.float64)
    truth = np.asarray(features.y).astype(str)
    groups = np.asarray(features.groups).astype(str)
    classes = sorted(np.unique(truth).tolist())
    binary = len(classes) == 2
    positive = config.evaluation.positive_class if binary else None
    if binary and positive not in classes:
        raise ConfigError(
            f"evaluation.positive_class={positive!r} is not one of the classes {classes}"
        )
    folds = _folds(truth, groups, config.evaluation.n_splits, config.seed)
    template = build_model(config)
    n_items = truth.size
    fold_of = np.empty(n_items, dtype=np.int64)
    predicted = np.empty(n_items, dtype=object)
    scores = np.zeros((n_items, len(classes)))
    rows: list[dict[str, Any]] = []
    for k, (train, test) in enumerate(folds):
        model = fit_estimator(clone(template), X[train], truth[train])
        predicted[test] = np.asarray(model.predict(X[test])).astype(str)
        scores[test] = class_scores(model, X[test], classes)
        fold_of[test] = k
        if np.unique(truth[test]).size < len(classes):
            _LOGGER.warning("fold %d lacks a class; its class-dependent scores are undefined", k)
        rows.append(
            {"fold": k, "n_train": int(train.size), "n_test": int(test.size)}
            | classification_metrics(
                truth[test],
                predicted[test].astype(str),
                scores[test],
                classes=classes,
                positive_class=positive,
            )
        )
        _LOGGER.info(
            "fold %d/%d: trained on %d, tested on %d", k + 1, len(folds), train.size, test.size
        )
    names = metric_names(binary=binary)
    labels_out = predicted.astype(str)
    pooled: dict[str, Any] = classification_metrics(
        truth, labels_out, scores, classes=classes, positive_class=positive
    )
    pooled["classes"] = classes
    pooled["confusion_matrix"] = confusion_matrix(truth, labels_out, labels=classes).tolist()
    pooled["per_class"] = per_class_table(truth, labels_out, classes)
    predictions = pd.DataFrame(
        {
            "path": features.paths,
            "group": groups,
            "fold": fold_of,
            "label": truth,
            "predicted": labels_out,
        }
    )
    for column, name in enumerate(score_columns(classes)):
        predictions[name] = scores[:, column]
    fold_table = pd.DataFrame(rows, columns=["fold", "n_train", "n_test", *names])
    metrics = {
        "task": "classification",
        "n_items": int(n_items),
        "n_splits": len(folds),
        "seed": int(config.seed),
        "folds": fold_table.to_dict(orient="records"),
        "summary": {name: summarize(fold_table[name].tolist()) for name in names},
        "pooled": pooled,
    }
    _LOGGER.info("pooled UAR %.3f over %d measurements", pooled["uar"], n_items)
    return CrossValidationResult(predictions=predictions, folds=fold_table, metrics=metrics)
