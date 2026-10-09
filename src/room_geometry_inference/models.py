"""Room classifiers and model files.

Four classifiers share one interface (``fit``, ``predict``, ``predict_proba``,
``classes_``), so cross-validation and training never branch on the model:

* ``svm``: standardization and a support vector machine with an RBF
  kernel (Cortes & Vapnik, 1995), probabilities by Platt scaling;
* ``knn``: standardization and a k-nearest-neighbor classifier (Cover &
  Hart, 1967) whose votes are weighted by inverse distance;
* ``tree``: a CART decision tree (Breiman et al., 1984);
* ``template``: the echo-pattern matcher of
  :mod:`room_geometry_inference.template_matching` (needs
  ``features.representation = "magnitude"``).

Everything learned from data lives inside the returned
:class:`~sklearn.pipeline.Pipeline`, so cross-validation fits it on the
training folds only. Model files are a dictionary with the keys ``format``,
``package_version``, ``config``, ``classes`` and ``estimator``, stored with
joblib.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .errors import ConfigError, ModelFileError
from .template_matching import TemplateMatcher

if TYPE_CHECKING:
    from .config import Config

_LOGGER = logging.getLogger(__name__)

#: Named kernel widths accepted by ``model.svm_gamma``.
SVM_GAMMAS: tuple[str, ...] = ("scale", "auto")
#: Format tag stored in, and required of, every model file.
MODEL_FORMAT: str = "room_geometry_inference/model-v1"

_BUNDLE_KEYS = ("format", "package_version", "config", "classes", "estimator")


def _svm(config: Config) -> BaseEstimator:
    classifier = SVC(
        C=config.model.svm_c,
        gamma=config.model.svm_gamma,
        kernel="rbf",
        probability=True,
        random_state=config.seed,
    )
    return Pipeline([("standardize", StandardScaler()), ("classify", classifier)])


def _knn(config: Config) -> BaseEstimator:
    classifier = KNeighborsClassifier(n_neighbors=config.model.n_neighbors, weights="distance")
    return Pipeline([("standardize", StandardScaler()), ("classify", classifier)])


def _tree(config: Config) -> BaseEstimator:
    classifier = DecisionTreeClassifier(
        max_depth=config.model.tree_max_depth, random_state=config.seed
    )
    return Pipeline([("classify", classifier)])


def _template(config: Config) -> BaseEstimator:
    echoes = config.echoes
    classifier = TemplateMatcher(
        n_echoes=echoes.n_echoes,
        tolerance_taps=config.model.template_tolerance_taps,
        min_score=config.model.template_min_score,
        pick_method=echoes.method,
        min_separation_taps=echoes.min_separation_taps,
        relative_height=echoes.relative_height,
        interpolate=echoes.interpolate,
        compensate_spreading=echoes.compensate_spreading,
    )
    return Pipeline([("classify", classifier)])


_FACTORIES: dict[str, Callable[[Config], BaseEstimator]] = {
    "svm": _svm,
    "knn": _knn,
    "tree": _tree,
    "template": _template,
}
#: Model names accepted by :func:`build_model` and ``model.name``.
MODEL_NAMES: tuple[str, ...] = tuple(_FACTORIES)


def build_model(config: Config) -> BaseEstimator:
    """Create the untrained classifier named by ``config.model.name``.

    Parameters
    ----------
    config : Config
        Complete configuration; the ``model`` section and ``seed`` are used
        (and ``echoes`` by the template matcher).

    Returns
    -------
    sklearn.pipeline.Pipeline

    Raises
    ------
    ConfigError
        For an unknown model name.

    References
    ----------
    .. [1] C. Cortes and V. Vapnik, "Support-vector networks," Machine
       Learning, vol. 20, no. 3, pp. 273-297, 1995.
    .. [2] T. Cover and P. Hart, "Nearest neighbor pattern classification,"
       IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21-27, 1967.
    .. [3] L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone,
       Classification and Regression Trees. Belmont, CA, USA: Wadsworth, 1984.

    Examples
    --------
    >>> from room_geometry_inference.config import Config
    >>> [name for name, _ in build_model(Config()).steps]
    ['standardize', 'classify']
    >>> type(build_model(Config().override({"model.name": "tree"}))[-1]).__name__
    'DecisionTreeClassifier'
    """
    name = config.model.name
    if name not in _FACTORIES:
        raise ConfigError(f"unknown model {name!r}; choose one of {list(MODEL_NAMES)}")
    return _FACTORIES[name](config)


def fit_estimator(estimator: BaseEstimator, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
    """Fit an estimator, routing its warnings into the log instead of the console.

    Returns
    -------
    sklearn.base.BaseEstimator
        The fitted estimator.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        _LOGGER.warning("the optimizer stopped before converging")
    for message in sorted(
        {str(item.message) for item in caught if not issubclass(item.category, ConvergenceWarning)}
    ):
        _LOGGER.warning("%s", message)
    return estimator


def save_model(
    path: str | Path,
    estimator: Any,
    config: Config,
    classes: Sequence[str] | None = None,
) -> Path:
    """Write a trained classifier with its configuration and class names.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination, conventionally ``model.joblib``.
    estimator : Any
        Trained estimator.
    config : Config
        Configuration it was trained with.
    classes : sequence of str, optional
        Class names; read from ``estimator.classes_`` when omitted.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    from . import __version__  # deferred: the package __init__ imports this module

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = [str(name) for name in (estimator.classes_ if classes is None else classes)]
    bundle: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "package_version": __version__,
        "config": config.to_dict(),
        "classes": names,
        "estimator": estimator,
    }
    joblib.dump(bundle, destination)
    _LOGGER.info("model written to %s", destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    """Read a model file written by :func:`save_model`.

    Only load files from sources you trust: joblib files are pickles and can
    run code when they are read.

    Parameters
    ----------
    path : str or pathlib.Path
        ``model.joblib``.

    Returns
    -------
    dict
        The bundle: ``format``, ``package_version``, ``config``, ``classes``
        and ``estimator``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ModelFileError
        If the file is unreadable, incomplete or not written by this package.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model file not found: {source}. Run 'train' first")
    try:
        bundle = joblib.load(source)
    except Exception as exc:  # unpickling raises many types for a foreign file
        raise ModelFileError(f"{source} is not a readable model file: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
        found = bundle.get("format") if isinstance(bundle, dict) else type(bundle).__name__
        raise ModelFileError(
            f"{source} was not written by this package (format {found!r}, expected "
            f"{MODEL_FORMAT!r})"
        )
    missing = [key for key in _BUNDLE_KEYS if key not in bundle]
    if missing:
        raise ModelFileError(f"{source} is incomplete; missing {missing}")
    return bundle
