"""Room recognition by matching echo arrival patterns against stored templates.

A room leaves a characteristic pattern of echo arrival times in its impulse
response. :class:`TemplateMatcher` stores the arrival pattern of every
training response; a new response is compared with each stored pattern of a
class, and the class score is the best match rate, i.e. the share of the new
response's echoes that lie within ``tolerance_taps`` of an echo of the
template:

.. math::

    s(\\mathcal{A}, \\mathcal{B}) = \\frac{1}{|\\mathcal{A}|}
    \\sum_{a \\in \\mathcal{A}} \\big[\\min_{b \\in \\mathcal{B}} |a - b|
    \\le \\tau\\big].

Requiring exact equality of arrival samples (a tolerance of zero) fails as
soon as noise or the identification shifts one echo by a single sample; the
tolerance makes the comparison usable on identified responses. When even the
best class matches poorly (score below ``min_score``),
:meth:`TemplateMatcher.predict_with_unknown` answers ``unknown``, so such
responses can be passed to the echo inversion instead of being forced into a
known room. The matcher follows the scikit-learn estimator interface, so it
is cross-validated like any other model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .echoes import pick_echoes

#: Label returned by :meth:`TemplateMatcher.predict_with_unknown` for poor matches.
UNKNOWN_LABEL: str = "unknown"


def match_rate(arrivals: ArrayLike, template: ArrayLike, tolerance_taps: float = 2.0) -> float:
    """Share of ``arrivals`` within ``tolerance_taps`` of some arrival of ``template``.

    Examples
    --------
    >>> match_rate([10, 50, 90, 130], [11, 52, 200], tolerance_taps=1)
    0.25
    >>> match_rate([10, 50, 90, 130], [11, 52, 200], tolerance_taps=2)
    0.5
    """
    first = np.asarray(arrivals, dtype=np.float64).ravel()
    second = np.asarray(template, dtype=np.float64).ravel()
    if first.size == 0 or second.size == 0:
        return 0.0
    gaps = np.min(np.abs(first[:, None] - second[None, :]), axis=1)
    return float(np.mean(gaps <= float(tolerance_taps)))


class TemplateMatcher(ClassifierMixin, BaseEstimator):
    """Nearest-template classifier on echo arrival patterns.

    Parameters
    ----------
    n_echoes : int, optional
        Echoes picked from every response.
    tolerance_taps : float, optional
        Two arrivals match when they are at most this many samples apart.
    min_score : float, optional
        Below this best-class score, :meth:`predict_with_unknown` answers
        :data:`UNKNOWN_LABEL`.
    pick_method : {"peaks", "top_k"}, optional
        Echo picker (see :func:`~room_geometry_inference.echoes.pick_echoes`).
    min_separation_taps : int, optional
        Minimum distance between picked peaks.
    relative_height : float, optional
        Peaks below this fraction of the largest one are ignored.
    interpolate : bool, optional
        Refine every arrival to sub-sample precision with a parabola.
    compensate_spreading : bool, optional
        Weight every tap by its delay before picking.

    Examples
    --------
    >>> def response(taps):
    ...     h = np.zeros(300)
    ...     h[taps] = 1.0
    ...     return h
    >>> X = np.stack([response([10, 60, 90]), response([11, 61, 92]),
    ...               response([10, 150, 240]), response([12, 151, 238])])
    >>> model = TemplateMatcher(n_echoes=3).fit(X, ["small", "small", "large", "large"])
    >>> model.predict(np.stack([response([11, 59, 91]), response([10, 149, 241])])).tolist()
    ['small', 'large']
    >>> model.predict_with_unknown(response([10, 120, 200])[None, :]).tolist()
    ['unknown']
    """

    def __init__(
        self,
        n_echoes: int = 9,
        tolerance_taps: float = 2.0,
        min_score: float = 0.7,
        pick_method: str = "peaks",
        min_separation_taps: int = 3,
        relative_height: float = 0.02,
        interpolate: bool = False,
        compensate_spreading: bool = False,
    ) -> None:
        self.n_echoes = n_echoes
        self.tolerance_taps = tolerance_taps
        self.min_score = min_score
        self.pick_method = pick_method
        self.min_separation_taps = min_separation_taps
        self.relative_height = relative_height
        self.interpolate = interpolate
        self.compensate_spreading = compensate_spreading

    def _arrivals(self, X: ArrayLike) -> list[NDArray[np.float64]]:
        rows = np.asarray(X, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError(f"X must be two-dimensional (one response per row), got {rows.shape}")
        return [
            pick_echoes(
                row,
                self.n_echoes,
                method=self.pick_method,
                min_separation_taps=self.min_separation_taps,
                relative_height=self.relative_height,
                interpolate=self.interpolate,
                compensate_spreading=self.compensate_spreading,
            )
            for row in rows
        ]

    def fit(self, X: ArrayLike, y: ArrayLike) -> TemplateMatcher:
        """Store the echo pattern of every training response with its class."""
        labels = np.asarray(y).astype(str)
        arrivals = self._arrivals(X)
        if labels.size != len(arrivals):
            raise ValueError(f"X has {len(arrivals)} rows but y has {labels.size} labels")
        self.classes_ = np.unique(labels)
        self.templates_: dict[str, list[NDArray[np.float64]]] = {
            name: [
                pattern for pattern, label in zip(arrivals, labels, strict=True) if label == name
            ]
            for name in self.classes_
        }
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def class_scores(self, X: ArrayLike) -> NDArray[np.float64]:
        """Best match rate against the templates of every class, shape ``(n, n_classes)``."""
        check_is_fitted(self, "templates_")
        arrivals = self._arrivals(X)
        scores = np.zeros((len(arrivals), self.classes_.size))
        for row, pattern in enumerate(arrivals):
            for column, name in enumerate(self.classes_):
                scores[row, column] = max(
                    match_rate(pattern, template, self.tolerance_taps)
                    for template in self.templates_[name]
                )
        return scores

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Class scores normalized to sum to one (uniform when nothing matches)."""
        scores = self.class_scores(X)
        totals = scores.sum(axis=1, keepdims=True)
        uniform = np.full_like(scores, 1.0 / scores.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(totals > 0, scores / np.where(totals > 0, totals, 1.0), uniform)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """The best-matching class of every response."""
        return self.classes_[np.argmax(self.class_scores(X), axis=1)]

    def predict_with_unknown(self, X: ArrayLike) -> NDArray[Any]:
        """Like :meth:`predict`, but :data:`UNKNOWN_LABEL` below ``min_score``."""
        scores = self.class_scores(X)
        best = self.classes_[np.argmax(scores, axis=1)].astype(object)
        best[scores.max(axis=1) < float(self.min_score)] = UNKNOWN_LABEL
        return best.astype(str)
