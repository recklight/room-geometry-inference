"""Run configuration: frozen dataclasses validated on construction, stored as YAML.

Every number the pipeline uses is a field here. Lengths, durations, angles and
levels carry their unit in the name (``_m``, ``_s``, ``_deg``, ``_db``, ``_taps``).
Each section validates itself in ``__post_init__`` and raises
:class:`~room_geometry_inference.errors.ConfigError` naming the dotted key, the value it got and
what is allowed. Unknown keys are rejected rather than ignored, so a typo
cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .autoencoder import ACTIVATIONS, DEVICES
from .denoising import DENOISER_NAMES
from .echoes import PICK_METHODS
from .errors import ConfigError
from .excitation import EXCITATION_KINDS
from .identification import IDENTIFICATION_METHODS
from .image_source import DELAY_MODES, ORDER_MODES, speed_of_sound
from .inversion import RANK_METRICS
from .models import MODEL_NAMES, SVM_GAMMAS
from .recording import NOISE_KINDS
from .rir_features import REPRESENTATIONS
from .sensitivity import SWEEP_NAMES

#: Image formats accepted by ``output.figure_format``.
FIGURE_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")


def _choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; choose one of {list(allowed)}")


def _positive(name: str, value: float) -> None:
    if not value > 0:
        raise ConfigError(f"{name} must be strictly positive, got {value!r}")


def _number(name: str, value: object, *, minimum: float | None = None) -> None:
    """Reject booleans, strings and other non-numbers, then check the lower bound."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value != value:
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value!r}")


def _integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer of at least {minimum}, got {value!r}")


def _flag(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")


def _numbers(name: str, value: object, length: int | None = None) -> tuple[float, ...]:
    """A non-empty tuple of numbers, optionally of a fixed length."""
    if not isinstance(value, tuple) or not value:
        raise ConfigError(f"{name} must be a non-empty list of numbers, got {value!r}")
    if length is not None and len(value) != length:
        raise ConfigError(f"{name} must have {length} values, got {list(value)}")
    for item in value:
        _number(name, item)
    return value


def _range(name: str, value: object, *, minimum: float) -> None:
    low, high = _numbers(name, value, 2)
    if not minimum <= low <= high:
        raise ConfigError(f"{name} must satisfy {minimum} <= low <= high, got {list(value)}")


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives.

    Attributes
    ----------
    dataset : str
        Dataset index (CSV, see :mod:`room_geometry_inference.dataset`). A placeholder until you
        point it at your own data.
    """

    dataset: str = "data/dataset.csv"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ConfigError(f"data.dataset must be a non-empty path, got {self.dataset!r}")


@dataclass(frozen=True)
class SyntheticConfig:
    """Size and generative parameters of the synthetic rooms and recordings.

    Every class is a nominal room size; each synthetic room draws its sides
    around it, and each recording of a room places the microphone and the
    source at random. ``excitation``, ``duration_s``, ``snr_db`` and
    ``noise`` also describe the recordings simulated by the ``simulate`` and
    ``sensitivity`` commands.

    Attributes
    ----------
    class_names : tuple of str
        Room classes, one per entry of ``room_sizes_m``.
    room_sizes_m : tuple of tuple of float
        Nominal ``[width, depth]`` of every class, in meters.
    room_jitter_m : float
        Each side of a synthetic room is drawn uniformly within this distance
        of the nominal one.
    n_rooms_per_class : int
        Distinct rooms per class; they are the cross-validation groups.
    n_recordings_per_room : int
        Recordings (placements) per room.
    source_distance_range_m : tuple of float
        Direct-path distance drawn uniformly from ``[low, high]``.
    source_bearing_deg : float
        Bearing of the source seen from the microphone.
    wall_margin_m : float
        Minimum distance of microphone and source from every wall.
    excitation : str
        ``white``, ``pink``, ``speech_like``, ``babble`` or ``sine``.
    duration_s : float
        Length of every recording.
    snr_db : float or None
        Signal-to-noise ratio at the microphone; ``None`` adds no noise.
    noise : str
        ``white``, ``pink``, ``babble`` or ``sine``.
    """

    class_names: tuple[str, ...] = ("large", "small")
    room_sizes_m: tuple[tuple[float, ...], ...] = ((10.7, 7.5), (3.5, 3.0))
    room_jitter_m: float = 0.3
    n_rooms_per_class: int = 8
    n_recordings_per_room: int = 3
    source_distance_range_m: tuple[float, ...] = (0.8, 1.6)
    source_bearing_deg: float = 45.0
    wall_margin_m: float = 0.3
    excitation: str = "white"
    duration_s: float = 2.0
    snr_db: float | None = 30.0
    noise: str = "white"

    def __post_init__(self) -> None:
        names = self.class_names
        if not isinstance(names, tuple) or len(names) < 2:
            raise ConfigError(
                f"synthetic.class_names must list at least two classes, got {names!r}"
            )
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
            names
        ):
            raise ConfigError(f"synthetic.class_names must be distinct names, got {list(names)}")
        sizes = self.room_sizes_m
        if not isinstance(sizes, tuple) or len(sizes) != len(names):
            raise ConfigError(
                "synthetic.room_sizes_m must give one [width, depth] per class in "
                f"synthetic.class_names ({len(names)}), got {sizes!r}"
            )
        for size in sizes:
            _numbers("synthetic.room_sizes_m", size, 2)
        _number("synthetic.room_jitter_m", self.room_jitter_m, minimum=0.0)
        _integer("synthetic.n_rooms_per_class", self.n_rooms_per_class, minimum=1)
        _integer("synthetic.n_recordings_per_room", self.n_recordings_per_room, minimum=1)
        _range("synthetic.source_distance_range_m", self.source_distance_range_m, minimum=0.05)
        _number("synthetic.source_bearing_deg", self.source_bearing_deg)
        _number("synthetic.wall_margin_m", self.wall_margin_m, minimum=0.0)
        _choice("synthetic.excitation", self.excitation, EXCITATION_KINDS)
        _number("synthetic.duration_s", self.duration_s)
        _positive("synthetic.duration_s", self.duration_s)
        if self.snr_db is not None:
            _number("synthetic.snr_db", self.snr_db)
        _choice("synthetic.noise", self.noise, NOISE_KINDS)
        reach = self.source_distance_range_m[1] + 2.0 * self.wall_margin_m
        for name, size in zip(names, sizes, strict=True):
            if min(size) - self.room_jitter_m <= reach:
                raise ConfigError(
                    f"synthetic.room_sizes_m: the {name!r} room ({list(size)}) is too small for "
                    "synthetic.source_distance_range_m plus two synthetic.wall_margin_m "
                    f"after synthetic.room_jitter_m ({reach:g} m needed)"
                )


@dataclass(frozen=True)
class AcousticsConfig:
    """Sampling and the image-source model of every simulated or resynthesized response.

    Attributes
    ----------
    sample_rate : int
        Rate in Hz of all processing; recordings at other rates are resampled.
    speed_of_sound_m_s : float
        Speed of sound, used unless ``temperature_c`` is set.
    temperature_c : float or None
        Air temperature; when set, the speed of sound is derived from it.
    max_order : int
        Image-source truncation order.
    order_mode : str
        ``per_axis`` (every axis up to ``max_order``) or ``total`` (at most
        ``max_order`` reflections in all).
    reflection : float
        Pressure reflection coefficient of every wall.
    delay : str
        ``integer`` (arrivals rounded to samples) or ``fractional``
        (Hann-windowed sinc).
    rir_taps : int
        Length of simulated responses.
    """

    sample_rate: int = 16_000
    speed_of_sound_m_s: float = 343.0
    temperature_c: float | None = None
    max_order: int = 1
    order_mode: str = "per_axis"
    reflection: float = 1.0
    delay: str = "integer"
    rir_taps: int = 1600

    def __post_init__(self) -> None:
        _integer("acoustics.sample_rate", self.sample_rate, minimum=1000)
        _number("acoustics.speed_of_sound_m_s", self.speed_of_sound_m_s)
        _positive("acoustics.speed_of_sound_m_s", self.speed_of_sound_m_s)
        if self.temperature_c is not None:
            _number("acoustics.temperature_c", self.temperature_c, minimum=-80.0)
        _integer("acoustics.max_order", self.max_order, minimum=0)
        _choice("acoustics.order_mode", self.order_mode, ORDER_MODES)
        _number("acoustics.reflection", self.reflection)
        if not -1.0 <= self.reflection <= 1.0:
            raise ConfigError(f"acoustics.reflection must lie in [-1, 1], got {self.reflection!r}")
        _choice("acoustics.delay", self.delay, DELAY_MODES)
        _integer("acoustics.rir_taps", self.rir_taps, minimum=16)

    @property
    def speed(self) -> float:
        """Speed of sound in m/s: from the temperature when it is set."""
        if self.temperature_c is not None:
            return speed_of_sound(self.temperature_c)
        return float(self.speed_of_sound_m_s)


@dataclass(frozen=True)
class RoomConfig:
    """The single planar scene of the ``simulate`` and ``sensitivity`` commands.

    Attributes
    ----------
    size_m : tuple of float
        ``[width, depth]`` of the room.
    microphone_m : tuple of float
        Microphone position.
    source_m : tuple of float
        Source position.
    """

    size_m: tuple[float, ...] = (10.7, 7.5)
    microphone_m: tuple[float, ...] = (5.35, 3.75)
    source_m: tuple[float, ...] = (6.35, 4.75)

    def __post_init__(self) -> None:
        size = _numbers("room.size_m", self.size_m, 2)
        for name in ("microphone_m", "source_m"):
            point = _numbers(f"room.{name}", getattr(self, name), 2)
            if not all(0 < coordinate < side for coordinate, side in zip(point, size, strict=True)):
                raise ConfigError(
                    f"room.{name}={list(point)} is not strictly inside room.size_m={list(size)}"
                )
        if tuple(self.microphone_m) == tuple(self.source_m):
            raise ConfigError("room.microphone_m and room.source_m coincide")


@dataclass(frozen=True)
class IdentificationConfig:
    """Impulse-response identification from excitation and microphone.

    Attributes
    ----------
    method : str
        ``nlms``, ``lms`` or ``least_squares``.
    n_taps : int
        Length of the identified response.
    step : float
        Step size: in ``(0, 2)`` for NLMS, below ``2 / (n_taps * power)``
        for LMS; unused by least squares.
    regularization : float
        NLMS normalization guard (least squares: relative diagonal loading).
    prune_threshold : float
        Taps below this fraction of the peak are set to zero before echo
        inversion, the ``magnitude`` and ``imf_energy`` features and the
        denoising comparison; ``decay_curve`` and ``echo_delays`` use the
        identified weights as they are.
    """

    method: str = "nlms"
    n_taps: int = 1600
    step: float = 0.8
    regularization: float = 1e-10
    prune_threshold: float = 1e-3

    def __post_init__(self) -> None:
        _choice("identification.method", self.method, IDENTIFICATION_METHODS)
        _integer("identification.n_taps", self.n_taps, minimum=16)
        _number("identification.step", self.step)
        _positive("identification.step", self.step)
        if self.method == "nlms" and not self.step < 2:
            raise ConfigError(f"identification.step must lie in (0, 2) for NLMS, got {self.step!r}")
        _number("identification.regularization", self.regularization, minimum=0.0)
        _number("identification.prune_threshold", self.prune_threshold, minimum=0.0)
        if not self.prune_threshold < 1:
            raise ConfigError(
                f"identification.prune_threshold must lie in [0, 1), got {self.prune_threshold!r}"
            )


@dataclass(frozen=True)
class EchoesConfig:
    """Picking echo arrivals in an identified response.

    Attributes
    ----------
    n_echoes : int
        Arrivals kept, the direct sound included (nine for a first-order
        planar room).
    method : str
        ``peaks`` (separated local maxima) or ``top_k`` (largest taps).
    min_separation_taps : int
        Minimum distance between two picked peaks.
    relative_height : float
        Peaks below this fraction of the largest one are ignored.
    interpolate : bool
        Refine every arrival to sub-sample precision with a parabola.
    compensate_spreading : bool
        Weight every tap by its delay before picking, so distant (weaker)
        reflections compete on equal terms with near ones.
    """

    n_echoes: int = 9
    method: str = "peaks"
    min_separation_taps: int = 3
    relative_height: float = 0.02
    interpolate: bool = False
    compensate_spreading: bool = True

    def __post_init__(self) -> None:
        _integer("echoes.n_echoes", self.n_echoes, minimum=5)
        _choice("echoes.method", self.method, PICK_METHODS)
        _integer("echoes.min_separation_taps", self.min_separation_taps, minimum=1)
        _number("echoes.relative_height", self.relative_height, minimum=0.0)
        if not self.relative_height < 1:
            raise ConfigError(
                f"echoes.relative_height must lie in [0, 1), got {self.relative_height!r}"
            )
        _flag("echoes.interpolate", self.interpolate)
        _flag("echoes.compensate_spreading", self.compensate_spreading)

    def options(self) -> dict[str, Any]:
        """Keyword arguments of :func:`~room_geometry_inference.echoes.pick_echoes`."""
        return {
            "n_echoes": self.n_echoes,
            "method": self.method,
            "min_separation_taps": self.min_separation_taps,
            "relative_height": self.relative_height,
            "interpolate": self.interpolate,
            "compensate_spreading": self.compensate_spreading,
        }


@dataclass(frozen=True)
class InversionConfig:
    """Echo inversion and candidate ranking.

    Attributes
    ----------
    source_bearing_deg : float
        Assumed bearing of the source seen from the microphone (first-order
        arrivals cannot reveal it).
    rank_metric : str
        ``tap_match``, ``l1`` or ``l2``.
    tolerance_taps : float
        Arrival tolerance of ``tap_match``.
    refine : bool
        Least-squares fit of every candidate to all nine arrivals.
    max_candidates : int
        Distinct candidates kept and resynthesized.
    """

    source_bearing_deg: float = 45.0
    rank_metric: str = "tap_match"
    tolerance_taps: float = 2.0
    refine: bool = True
    max_candidates: int = 20

    def __post_init__(self) -> None:
        _number("inversion.source_bearing_deg", self.source_bearing_deg)
        _choice("inversion.rank_metric", self.rank_metric, RANK_METRICS)
        _number("inversion.tolerance_taps", self.tolerance_taps, minimum=0.0)
        _flag("inversion.refine", self.refine)
        _integer("inversion.max_candidates", self.max_candidates, minimum=1)


@dataclass(frozen=True)
class SensitivityConfig:
    """Sweeps of the ``sensitivity`` command around the ``room`` scene.

    Attributes
    ----------
    sweeps : tuple of str
        Any of ``delay_offset``, ``source_bearing``, ``source_shift``, ``snr``.
    delay_offsets_samples : tuple of float
        Constant offsets added to every arrival (a latency or a timing bias).
    bearings_deg : tuple of float
        Source bearings at the ``room`` source distance, inverted under the
        assumed ``inversion.source_bearing_deg``.
    shifts_m : tuple of float
        Source displacements along x and along y.
    snr_db : tuple of float
        Signal-to-noise ratios of the full chain (simulate, identify, pick, invert).
    n_trials : int
        Noise realizations per signal-to-noise ratio.
    """

    sweeps: tuple[str, ...] = ("delay_offset", "source_bearing", "source_shift", "snr")
    delay_offsets_samples: tuple[float, ...] = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
    bearings_deg: tuple[float, ...] = (5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0)
    shifts_m: tuple[float, ...] = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)
    snr_db: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 20.0, 30.0)
    n_trials: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.sweeps, tuple) or not self.sweeps:
            raise ConfigError(
                f"sensitivity.sweeps must list at least one sweep, got {self.sweeps!r}"
            )
        for name in self.sweeps:
            _choice("sensitivity.sweeps", name, SWEEP_NAMES)
        for name in ("delay_offsets_samples", "bearings_deg", "shifts_m", "snr_db"):
            _numbers(f"sensitivity.{name}", getattr(self, name))
        _integer("sensitivity.n_trials", self.n_trials, minimum=1)


@dataclass(frozen=True)
class DenoisingConfig:
    """Denoisers compared by the ``denoise`` command.

    Attributes
    ----------
    methods : tuple of str
        Any of ``identity``, ``linear``, ``pca``, ``mlp``,
        ``deep_autoencoder`` (the last needs the ``dl`` extra).
    architectures : tuple of tuple of int
        Hidden-layer widths; every network method is evaluated once per entry.
    activation : str
        ``logistic``, ``relu`` or ``tanh``.
    n_epochs : int
        Training passes of the networks.
    learning_rate : float
        Adam step size of the networks.
    batch_size : int
        Mini-batch size of the networks.
    weight_decay : float
        L2 penalty of the networks.
    n_components : int
        Principal components of ``pca``.
    device : str
        ``cpu`` or ``cuda`` (``deep_autoencoder``).
    """

    methods: tuple[str, ...] = ("identity", "linear", "pca", "mlp")
    architectures: tuple[tuple[int, ...], ...] = ((300, 300, 300),)
    activation: str = "logistic"
    n_epochs: int = 300
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    n_components: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.methods, tuple) or not self.methods:
            raise ConfigError(
                f"denoising.methods must list at least one method, got {self.methods!r}"
            )
        for name in self.methods:
            _choice("denoising.methods", name, DENOISER_NAMES)
        if not isinstance(self.architectures, tuple) or not self.architectures:
            raise ConfigError(
                "denoising.architectures must list at least one network, "
                f"got {self.architectures!r}"
            )
        for hidden in self.architectures:
            if not isinstance(hidden, tuple) or not hidden:
                raise ConfigError(
                    f"denoising.architectures entries must list layer widths, got {hidden!r}"
                )
            for width in hidden:
                _integer("denoising.architectures", width, minimum=1)
        _choice("denoising.activation", self.activation, ACTIVATIONS)
        _integer("denoising.n_epochs", self.n_epochs, minimum=1)
        _number("denoising.learning_rate", self.learning_rate)
        _positive("denoising.learning_rate", self.learning_rate)
        _integer("denoising.batch_size", self.batch_size, minimum=1)
        _number("denoising.weight_decay", self.weight_decay, minimum=0.0)
        _integer("denoising.n_components", self.n_components, minimum=1)
        _choice("denoising.device", self.device, DEVICES)

    def options(self) -> dict[str, Any]:
        """Keyword arguments shared by every denoiser (``options`` of ``compare_denoisers``)."""
        return {
            "activation": self.activation,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "n_components": self.n_components,
            "device": self.device,
        }


@dataclass(frozen=True)
class FeaturesConfig:
    """Model input computed from every identified response.

    Attributes
    ----------
    representation : str
        ``magnitude``, ``decay_curve``, ``echo_delays`` or ``imf_energy``.
    floor_db : float
        Floor of the energy decay curve.
    max_imfs : int
        Intrinsic mode functions described by ``imf_energy``.
    emd_tolerance : float
        Sifting stops when successive candidates differ by less than this
        normalized squared difference.
    """

    representation: str = "decay_curve"
    floor_db: float = -60.0
    max_imfs: int = 6
    emd_tolerance: float = 0.3

    def __post_init__(self) -> None:
        _choice("features.representation", self.representation, REPRESENTATIONS)
        _number("features.floor_db", self.floor_db)
        if not self.floor_db < 0:
            raise ConfigError(f"features.floor_db must be negative, got {self.floor_db!r}")
        _integer("features.max_imfs", self.max_imfs, minimum=1)
        _number("features.emd_tolerance", self.emd_tolerance)
        _positive("features.emd_tolerance", self.emd_tolerance)


@dataclass(frozen=True)
class ModelConfig:
    """Room classifier and its hyperparameters.

    Attributes
    ----------
    name : str
        ``svm``, ``knn``, ``tree`` or ``template``.
    svm_c : float
        Penalty of the support vector machine.
    svm_gamma : str or float
        RBF kernel width: ``scale``, ``auto`` or a positive number.
    n_neighbors : int
        Neighbors of the distance-weighted k-nearest-neighbor classifier.
    tree_max_depth : int or None
        Depth limit of the decision tree; ``None`` grows it fully.
    template_tolerance_taps : float
        Arrival tolerance of the template matcher.
    template_min_score : float
        Below this match rate the template matcher answers ``unknown``
        (``predict`` and :meth:`TemplateMatcher.predict_with_unknown` only;
        cross-validation always picks the best class).
    """

    name: str = "svm"
    svm_c: float = 10.0
    svm_gamma: str | float = "scale"
    n_neighbors: int = 5
    tree_max_depth: int | None = None
    template_tolerance_taps: float = 2.0
    template_min_score: float = 0.7

    def __post_init__(self) -> None:
        _choice("model.name", self.name, MODEL_NAMES)
        _number("model.svm_c", self.svm_c)
        _positive("model.svm_c", self.svm_c)
        if isinstance(self.svm_gamma, str):
            _choice("model.svm_gamma", self.svm_gamma, SVM_GAMMAS)
        else:
            _number("model.svm_gamma", self.svm_gamma)
            _positive("model.svm_gamma", self.svm_gamma)
        _integer("model.n_neighbors", self.n_neighbors, minimum=1)
        if self.tree_max_depth is not None:
            _integer("model.tree_max_depth", self.tree_max_depth, minimum=1)
        _number("model.template_tolerance_taps", self.template_tolerance_taps, minimum=0.0)
        _number("model.template_min_score", self.template_min_score, minimum=0.0)
        if self.template_min_score > 1:
            raise ConfigError(
                f"model.template_min_score must lie in [0, 1], got {self.template_min_score!r}"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation.

    Attributes
    ----------
    n_splits : int
        Number of folds; stratified, and grouped when the index has a
        ``group`` column (rooms), so no room is on both sides of a split.
    positive_class : str
        Class whose recall is the sensitivity and whose score gives the ROC
        curve when there are two classes.
    """

    n_splits: int = 5
    positive_class: str = "large"

    def __post_init__(self) -> None:
        _integer("evaluation.n_splits", self.n_splits, minimum=2)
        if not isinstance(self.positive_class, str) or not self.positive_class:
            raise ConfigError(
                f"evaluation.positive_class must be a class name, got {self.positive_class!r}"
            )


@dataclass(frozen=True)
class OutputConfig:
    """Where and how results are written.

    Attributes
    ----------
    root : str
        Run folders are created here when ``--out`` is not given.
    figure_format : str
        ``png``, ``pdf`` or ``svg``.
    dpi : int
        Resolution of raster figures.
    """

    root: str = "outputs"
    figure_format: str = "png"
    dpi: int = 150

    def __post_init__(self) -> None:
        _choice("output.figure_format", self.figure_format, FIGURE_FORMATS)
        _number("output.dpi", self.dpi)
        _positive("output.dpi", self.dpi)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "synthetic": SyntheticConfig,
    "acoustics": AcousticsConfig,
    "room": RoomConfig,
    "identification": IdentificationConfig,
    "echoes": EchoesConfig,
    "inversion": InversionConfig,
    "sensitivity": SensitivityConfig,
    "denoising": DenoisingConfig,
    "features": FeaturesConfig,
    "model": ModelConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
}

#: Sections whose values change what ``prepare`` writes.
FEATURE_SECTIONS: tuple[str, ...] = ("acoustics", "identification", "echoes", "features")


def _tuplify(value: Any) -> Any:
    """YAML lists become tuples, so configurations stay hashable."""
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    return value


def _listify(value: Any) -> Any:
    """Tuples become lists again, for YAML and JSON."""
    if isinstance(value, dict):
        return {key: _listify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_listify(item) for item in value]
    return value


def _build_section(name: str, cls: type, raw: Any) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"section {name!r} must be a mapping, got {type(raw).__name__}")
    known = sorted(f.name for f in dataclasses.fields(cls))
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in section {name!r}; valid keys are {known}")
    try:
        return cls(**{key: _tuplify(value) for key, value in raw.items()})
    except TypeError as exc:
        raise ConfigError(f"section {name!r}: {exc}") from None


@dataclass(frozen=True)
class Config:
    """Complete configuration of a run.

    Attributes
    ----------
    seed : int
        Master seed: splits, model initialization and synthetic data derive from it.
    data, synthetic, acoustics, room, identification, echoes, inversion
        See the section classes.
    sensitivity, denoising, features, model, evaluation, output
        See the section classes.

    Examples
    --------
    >>> config = Config().override({"evaluation.n_splits": 3})
    >>> config.evaluation.n_splits, Config.from_dict(config.to_dict()) == config
    (3, True)
    """

    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    acoustics: AcousticsConfig = field(default_factory=AcousticsConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)
    echoes: EchoesConfig = field(default_factory=EchoesConfig)
    inversion: InversionConfig = field(default_factory=InversionConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"seed must be a non-negative integer, got {self.seed!r}")
        if self.identification.n_taps < self.echoes.n_echoes:
            raise ConfigError(
                f"identification.n_taps={self.identification.n_taps!r} is shorter than "
                f"echoes.n_echoes={self.echoes.n_echoes!r}"
            )
        if self.model.name == "template" and self.features.representation != "magnitude":
            raise ConfigError(
                "model.name='template' matches echo arrivals and needs "
                f"features.representation='magnitude', got {self.features.representation!r}"
            )
        recording_samples = self.synthetic.duration_s * self.acoustics.sample_rate
        if recording_samples <= self.identification.n_taps:
            raise ConfigError(
                f"synthetic.duration_s={self.synthetic.duration_s!r} gives "
                f"{int(recording_samples)} samples at acoustics.sample_rate="
                f"{self.acoustics.sample_rate!r}, not more than identification.n_taps="
                f"{self.identification.n_taps!r}"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> Config:
        """Build a configuration from a nested mapping, rejecting unknown keys."""
        data = dict(raw or {})
        known = sorted(f.name for f in dataclasses.fields(cls))
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ConfigError(f"unknown setting(s) {unknown}; valid top-level keys are {known}")
        kwargs: dict[str, Any] = {}
        for name, value in data.items():
            if name in _SECTIONS:
                kwargs[name] = _build_section(name, _SECTIONS[name], value)
            else:
                kwargs[name] = _tuplify(value)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML file; keys it leaves out keep their defaults."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"configuration file not found: {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{source} is not valid YAML: {exc}") from None
        if raw is not None and not isinstance(raw, Mapping):
            raise ConfigError(f"{source} must contain a mapping of settings")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested mapping (lists instead of tuples), ready for YAML or JSON."""
        return _listify(dataclasses.asdict(self))

    def to_yaml(self, path: str | Path) -> Path:
        """Write the configuration as YAML and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
        return destination

    def override(self, updates: Mapping[str, Any]) -> Config:
        """Return a validated copy with dotted keys replaced, e.g. ``{"output.dpi": 300}``."""
        tree = self.to_dict()
        for dotted, value in updates.items():
            *parents, leaf = dotted.split(".")
            node = tree
            for part in parents:
                child = node.get(part)
                if not isinstance(child, dict):
                    valid = sorted(key for key, item in node.items() if isinstance(item, dict))
                    raise ConfigError(
                        f"unknown configuration section in {dotted!r}; valid sections are {valid}"
                    )
                node = child
            if leaf not in node:
                raise ConfigError(
                    f"unknown configuration key {dotted!r}; valid keys here are {sorted(node)}"
                )
            node[leaf] = value
        return Config.from_dict(tree)
