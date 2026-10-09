"""Cleaning identified impulse responses, compared by cross-validation.

An identified response :math:`|\\hat h|` carries an estimation-noise floor
that the true sparse response :math:`h` (a few echoes, zeros elsewhere) does
not. A denoiser learns the map from the first to the second on pairs where
the truth is known (simulated rooms) and is scored per response by

.. math::

    \\mathrm{MSE_{dB}} = 10 \\log_{10} \\frac{1}{N} \\sum_{t} \\big(h[t] - \\tilde h[t]\\big)^2 .

The candidates, from trivial to deep:

* ``identity``: the noisy response itself, the baseline every other method
  has to beat.
* ``linear``: one gain and one offset for all taps, fitted by ordinary
  least squares, i.e. a global calibration of the noise floor.
* ``pca``: projection of the noisy response onto the leading principal
  components of the clean training responses (Pearson, 1901), which keeps
  only what clean responses can look like.
* ``mlp``: a multilayer perceptron (scikit-learn, Adam) with per-tap
  standardization of inputs and targets. It is a denoising autoencoder
  (Vincent et al., 2008; Lu et al., 2013) small enough for a CPU.
* ``deep_autoencoder``: the same idea in PyTorch
  (:mod:`room_geometry_inference.autoencoder`, ``dl`` extra), for larger
  networks and GPU training.

:func:`compare_denoisers` evaluates every method with the same k-fold split,
grouped by room when a group is given, so a denoiser is always scored on
rooms it has never seen; a lookup table that memorized the training rooms
would fail here. ``mlp`` and ``deep_autoencoder`` are evaluated once per
entry of ``architectures``, each entry as its own method.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .autoencoder import DeepDenoisingAutoencoder
from .errors import DatasetError, require_module
from .models import fit_estimator

_LOGGER = logging.getLogger(__name__)

#: Columns of the table returned by :func:`compare_denoisers`.
SCORE_COLUMNS: tuple[str, ...] = ("item", "group", "fold", "method", "mse_db")

#: Smallest mean squared error reported (-300 dB), so a perfect estimate stays finite.
_MSE_FLOOR = 1e-30


class IdentityDenoiser(RegressorMixin, BaseEstimator):
    """Returns its input: the do-nothing baseline.

    Examples
    --------
    >>> IdentityDenoiser().fit([[1.0, 2.0]], [[0.0, 0.0]]).predict([[3.0, 4.0]]).tolist()
    [[3.0, 4.0]]
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> IdentityDenoiser:
        """Nothing to learn."""
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """A copy of ``X``."""
        return np.array(X, dtype=np.float64)


class LinearCalibration(RegressorMixin, BaseEstimator):
    """One gain and one offset shared by all taps, fitted by least squares.

    Examples
    --------
    >>> noisy = np.array([[0.1, 1.0, 0.1], [0.2, 0.6, 0.2]])
    >>> model = LinearCalibration().fit(noisy, 2.0 * noisy - 0.2)
    >>> round(model.gain_, 6), round(model.offset_, 6)
    (2.0, -0.2)
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearCalibration:
        """Solve ``y ~ offset + gain * X`` over every tap of every row."""
        noisy = np.asarray(X, dtype=np.float64).ravel()
        clean = np.asarray(y, dtype=np.float64).ravel()
        if noisy.size != clean.size:
            raise ValueError(f"X and y must have the same shape, got {noisy.size} and {clean.size}")
        design = np.column_stack([np.ones_like(noisy), noisy])
        (offset, gain), *_ = np.linalg.lstsq(design, clean, rcond=None)
        self.offset_, self.gain_ = float(offset), float(gain)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """``offset + gain * X``."""
        check_is_fitted(self, "gain_")
        return self.offset_ + self.gain_ * np.asarray(X, dtype=np.float64)


class PCADenoiser(RegressorMixin, BaseEstimator):
    """Projection onto the principal subspace of the clean training responses.

    Parameters
    ----------
    n_components : int, optional
        Principal components kept (at most the number of training rows).

    References
    ----------
    .. [1] K. Pearson, "On lines and planes of closest fit to systems of points
       in space," The London, Edinburgh, and Dublin Philosophical Magazine and
       Journal of Science, vol. 2, no. 11, pp. 559-572, 1901.

    Examples
    --------
    >>> clean = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    >>> model = PCADenoiser(n_components=2).fit(clean + 0.1, clean)
    >>> model.predict([[0.5, 0.5, 0.3]]).round(6).tolist()
    [[0.5, 0.5, 0.0]]
    """

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components

    def fit(self, X: ArrayLike, y: ArrayLike) -> PCADenoiser:
        """Fit the principal components of the clean targets ``y``.

        The exact SVD is used: with few long responses the automatic solver
        choice would be a randomized SVD, whose result changes from run to run.
        """
        clean = np.asarray(y, dtype=np.float64)
        components = max(1, min(int(self.n_components), clean.shape[0], clean.shape[1]))
        self.pca_ = PCA(n_components=components, svd_solver="full").fit(clean)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project ``X`` onto the clean subspace and back."""
        check_is_fitted(self, "pca_")
        return np.asarray(
            self.pca_.inverse_transform(self.pca_.transform(np.asarray(X, dtype=np.float64))),
            dtype=np.float64,
        )


def build_denoiser(
    name: str,
    *,
    hidden_units: Sequence[int] = (300, 300, 300),
    activation: str = "logistic",
    n_epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 100,
    weight_decay: float = 1e-4,
    n_components: int = 10,
    device: str = "cpu",
    random_state: int = 0,
) -> BaseEstimator:
    """An untrained denoiser by name.

    Parameters
    ----------
    name : {"identity", "linear", "pca", "mlp", "deep_autoencoder"}
        Method (see the module documentation).
    hidden_units, activation, n_epochs, learning_rate, batch_size, weight_decay
        Network settings of ``mlp`` and ``deep_autoencoder``.
    n_components : int, optional
        Principal components of ``pca``.
    device : {"cpu", "cuda"}, optional
        Device of ``deep_autoencoder``.
    random_state : int, optional
        Seed of the network initialization and batch order.

    Returns
    -------
    sklearn.base.BaseEstimator

    Examples
    --------
    >>> type(build_denoiser("mlp")).__name__, type(build_denoiser("pca")).__name__
    ('TransformedTargetRegressor', 'PCADenoiser')
    """
    if name not in _DENOISERS:
        raise ValueError(f"unknown denoiser {name!r}; choose one of {list(DENOISER_NAMES)}")
    settings = {
        "hidden_units": tuple(int(width) for width in hidden_units),
        "activation": activation,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "n_components": n_components,
        "device": device,
        "random_state": random_state,
    }
    return _DENOISERS[name](settings)


def _identity(settings: Mapping[str, Any]) -> BaseEstimator:
    return IdentityDenoiser()


def _linear(settings: Mapping[str, Any]) -> BaseEstimator:
    return LinearCalibration()


def _pca(settings: Mapping[str, Any]) -> BaseEstimator:
    return PCADenoiser(n_components=settings["n_components"])


def _mlp(settings: Mapping[str, Any]) -> BaseEstimator:
    network = MLPRegressor(
        hidden_layer_sizes=settings["hidden_units"],
        activation=settings["activation"],
        solver="adam",
        alpha=settings["weight_decay"],
        batch_size=settings["batch_size"],
        learning_rate_init=settings["learning_rate"],
        max_iter=settings["n_epochs"],
        random_state=settings["random_state"],
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("standardize", StandardScaler()), ("network", network)]),
        transformer=StandardScaler(),
    )


def _deep_autoencoder(settings: Mapping[str, Any]) -> BaseEstimator:
    require_module("torch", extra="dl", feature="the deep denoising autoencoder")
    return DeepDenoisingAutoencoder(
        hidden_units=settings["hidden_units"],
        activation=settings["activation"],
        n_epochs=settings["n_epochs"],
        learning_rate=settings["learning_rate"],
        batch_size=settings["batch_size"],
        weight_decay=settings["weight_decay"],
        device=settings["device"],
        random_state=settings["random_state"],
    )


_DENOISERS: dict[str, Callable[[Mapping[str, Any]], BaseEstimator]] = {
    "identity": _identity,
    "linear": _linear,
    "pca": _pca,
    "mlp": _mlp,
    "deep_autoencoder": _deep_autoencoder,
}
#: Denoisers accepted by :func:`build_denoiser` and ``denoising.methods``.
DENOISER_NAMES: tuple[str, ...] = tuple(_DENOISERS)


def per_sample_mse_db(clean: ArrayLike, estimate: ArrayLike) -> NDArray[np.float64]:
    """Mean squared error of every row in dB, :math:`10\\log_{10}` of the mean over taps.

    Examples
    --------
    >>> per_sample_mse_db([[1.0, 0.0], [1.0, 0.0]], [[0.9, 0.1], [1.0, 0.0]]).round(3).tolist()
    [-20.0, -300.0]
    """
    truth = np.atleast_2d(np.asarray(clean, dtype=np.float64))
    guess = np.atleast_2d(np.asarray(estimate, dtype=np.float64))
    if truth.shape != guess.shape:
        raise ValueError(f"shapes differ: {truth.shape} and {guess.shape}")
    mse = np.mean((truth - guess) ** 2, axis=1)
    return 10.0 * np.log10(np.maximum(mse, _MSE_FLOOR))


def _method_label(name: str, hidden_units: Sequence[int]) -> str:
    """``mlp 300-300-300`` for networks, the bare name otherwise."""
    if name in ("mlp", "deep_autoencoder"):
        return f"{name} {'-'.join(str(int(width)) for width in hidden_units)}"
    return name


def _fit(template: BaseEstimator, X: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
    """Fit a clone of ``template``, its batch size capped at the number of training rows.

    Fitting warnings go to the log, as in :func:`~room_geometry_inference.models.fit_estimator`.
    """
    model = clone(template)
    caps = {
        key: min(value, X.shape[0])
        for key, value in model.get_params().items()
        if key.endswith("batch_size") and isinstance(value, int) and not isinstance(value, bool)
    }
    if caps:
        model.set_params(**caps)
    return fit_estimator(model, X, y)


def compare_denoisers(
    noisy: ArrayLike,
    clean: ArrayLike,
    groups: ArrayLike | None = None,
    *,
    methods: Sequence[str] = ("identity", "linear", "pca", "mlp"),
    architectures: Sequence[Sequence[int]] = ((300, 300, 300),),
    n_splits: int = 5,
    random_state: int = 0,
    options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Out-of-fold MSE (dB) of every denoiser on every response.

    Parameters
    ----------
    noisy, clean : array_like, shape (n_items, n_taps)
        Identified (noisy) and true (clean) responses, peak-normalized.
    groups : array_like of str, shape (n_items,), optional
        Room of every response; when given, all responses of a room stay in
        the same fold (:class:`~sklearn.model_selection.GroupKFold`).
    methods : sequence of str, optional
        Denoisers from :data:`DENOISER_NAMES`.
    architectures : sequence of sequence of int, optional
        Hidden-layer widths; ``mlp`` and ``deep_autoencoder`` are evaluated
        once per architecture.
    n_splits : int, optional
        Number of folds.
    random_state : int, optional
        Seed of the split (without groups) and of the networks.
    options : dict, optional
        Further keyword arguments of :func:`build_denoiser`.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`SCORE_COLUMNS`: one row per response and method, the
        method labeled with its architecture (``mlp 300-300-300``).

    Raises
    ------
    DatasetError
        If the arrays disagree or there are too few items or groups for the folds.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> clean = np.zeros((12, 20))
    >>> clean[np.arange(12), rng.integers(0, 20, 12)] = 1.0
    >>> noisy = clean + 0.05 * rng.random((12, 20))
    >>> scores = compare_denoisers(noisy, clean, methods=("identity", "linear"), n_splits=3)
    >>> scores.groupby("method")["mse_db"].mean().round(1).to_dict()
    {'identity': -30.4, 'linear': -36.6}
    """
    inputs = np.asarray(noisy, dtype=np.float64)
    targets = np.asarray(clean, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape != targets.shape:
        raise DatasetError(
            f"noisy and clean responses must be equal 2-D arrays, got {inputs.shape} "
            f"and {targets.shape}"
        )
    unknown = sorted(set(methods) - set(DENOISER_NAMES))
    if unknown:
        raise ValueError(f"unknown denoiser(s) {unknown}; choose from {list(DENOISER_NAMES)}")
    n_items = inputs.shape[0]
    labels = (
        np.full(n_items, "", dtype=object)
        if groups is None
        else np.asarray(groups).astype(str).astype(object)
    )
    grouped = groups is not None and bool(np.any(labels != ""))
    if grouped:
        n_groups = np.unique(labels).size
        if n_groups < n_splits:
            raise DatasetError(f"only {n_groups} groups for {n_splits} folds; lower n_splits")
        splitter: Any = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(inputs, groups=labels))
    else:
        if n_items < n_splits:
            raise DatasetError(f"only {n_items} responses for {n_splits} folds")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(splitter.split(inputs))
    settings = dict(options or {})
    variants: list[tuple[str, BaseEstimator]] = []
    for name in methods:
        shapes = architectures if name in ("mlp", "deep_autoencoder") else [()]
        for hidden in shapes:
            extra = {"hidden_units": tuple(hidden)} if hidden else {}
            model = build_denoiser(name, random_state=random_state, **settings, **extra)
            variants.append((_method_label(name, hidden), model))
    rows = []
    for fold, (train, test) in enumerate(folds):
        for label, template in variants:
            model = _fit(template, inputs[train], targets[train])
            scores = per_sample_mse_db(targets[test], model.predict(inputs[test]))
            rows.extend(
                {
                    "item": int(item),
                    "group": str(labels[item]),
                    "fold": fold,
                    "method": label,
                    "mse_db": float(score),
                }
                for item, score in zip(test, scores, strict=True)
            )
        _LOGGER.info("denoising fold %d/%d done", fold + 1, len(folds))
    table = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))
    order = {label: position for position, (label, _) in enumerate(variants)}
    table["_order"] = table["method"].map(order)
    table = table.sort_values(["_order", "item"], kind="stable").drop(columns="_order")
    return table.reset_index(drop=True)


def denoising_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample standard deviation (``ddof=1``) and median MSE per method.

    The ``gain_db`` column is how much lower the mean is than that of the
    ``identity`` baseline (``nan`` when the baseline was not evaluated).

    Examples
    --------
    >>> table = pd.DataFrame({"method": ["identity", "identity", "pca", "pca"],
    ...                       "mse_db": [-10.0, -12.0, -15.0, -17.0]})
    >>> denoising_summary(table)[["method", "mean_db", "gain_db"]].values.tolist()
    [['identity', -11.0, 0.0], ['pca', -16.0, 5.0]]
    """
    summary = (
        scores.groupby("method", sort=False)["mse_db"]
        .agg(n_items="size", mean_db="mean", std_db="std", median_db="median")
        .reset_index()
    )
    baseline = summary.loc[summary["method"] == "identity", "mean_db"]
    reference = float(baseline.iloc[0]) if len(baseline) else float("nan")
    summary["gain_db"] = reference - summary["mean_db"]
    return summary
