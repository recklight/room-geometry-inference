"""Room geometry from a single microphone: simulation, identification, echo inversion.

A loudspeaker plays a known excitation and one microphone records it. The
impulse response between them is identified with an adaptive filter (NLMS,
LMS or least squares); its strongest echoes are converted to distances and
inverted, under the first-order image-source model of a rectangular room, to
the room size and the source distance. The same responses feed a
cross-validated room-size classifier, a comparison of response denoisers and
sensitivity sweeps of the inversion.

Main entry points:

* :class:`Config`: every setting, validated, stored as YAML.
* :func:`image_source_rir`, :func:`speed_of_sound`: image-source simulation
  of planar and three-dimensional rectangular rooms.
* :func:`make_excitation`, :func:`add_noise`, :func:`read_recording`,
  :func:`write_recording`: test signals, noise at an exact SNR and the
  two-channel measurement files.
* :func:`nlms`, :func:`lms`, :func:`least_squares_response`,
  :func:`normalized_misalignment_db`: impulse-response identification.
* :func:`pick_echoes`, :func:`taps_to_distances`, :func:`tap_shift_error`:
  echo arrivals.
* :func:`invert_echoes`, :func:`rank_candidates`, :func:`bearing_family`,
  :class:`RoomEstimate`: closed-form echo inversion, ranking by resynthesis
  and the bearing ambiguity.
* :func:`invert_response`, :func:`simulate_scene`: the estimation chain end
  to end.
* :func:`run_sweeps`, :func:`summarize_sweeps`: sensitivity to timing,
  bearing, source position and noise.
* :func:`decompose_emd`, :func:`represent_response`: empirical mode
  decomposition and the feature representations of a response.
* :func:`compare_denoisers`, :class:`DeepDenoisingAutoencoder`: response
  denoising (the deep autoencoder needs the ``dl`` extra).
* :class:`TemplateMatcher`: echo-pattern matching that can answer ``unknown``.
* :func:`write_synthetic_dataset`: synthetic measurements with known geometry.
* :func:`load_dataset_index`, :func:`save_features`, :func:`load_features`:
  the ``dataset.csv`` index and ``features.npz``.
* :func:`build_model`, :func:`save_model`, :func:`load_model`: the classifier
  registry (``svm``, ``knn``, ``tree``, ``template``) and model files.
* :func:`prepare_features`, :func:`train_model`, :func:`predict_inputs`,
  :func:`invert_measurements`, :func:`denoising_pairs`: the steps behind the
  command line.
* :func:`cross_validate`, :class:`CrossValidationResult`: stratified,
  room-grouped k-fold evaluation.
* :func:`plot_confusion_matrix` and the other ``plot_*`` functions: report
  figures, each with a table of the values it plots.
"""

from __future__ import annotations

from .autoencoder import DeepDenoisingAutoencoder
from .config import (
    FEATURE_SECTIONS,
    FIGURE_FORMATS,
    AcousticsConfig,
    Config,
    DataConfig,
    DenoisingConfig,
    EchoesConfig,
    EvaluationConfig,
    FeaturesConfig,
    IdentificationConfig,
    InversionConfig,
    ModelConfig,
    OutputConfig,
    RoomConfig,
    SensitivityConfig,
    SyntheticConfig,
)
from .dataset import (
    DATASET_INDEX_NAME,
    GEOMETRY_COLUMNS,
    FeatureSet,
    index_audio_folder,
    items_table,
    load_dataset_index,
    load_features,
    load_svmlight,
    save_features,
    save_svmlight,
)
from .denoising import (
    DENOISER_NAMES,
    LinearCalibration,
    PCADenoiser,
    build_denoiser,
    compare_denoisers,
    denoising_summary,
    per_sample_mse_db,
)
from .echoes import pick_echoes, tap_shift_error, taps_to_distances
from .emd import ModeDecomposition, decompose_emd
from .errors import (
    ConfigError,
    DatasetError,
    GeometryError,
    InversionError,
    MissingDependencyError,
    ModelFileError,
    require_module,
)
from .evaluation import CrossValidationResult, classification_metrics, cross_validate
from .excitation import EXCITATION_KINDS, make_excitation
from .figures import (
    SERIES_COLORS,
    confusion_matrix_table,
    denoising_scores_table,
    impulse_response_table,
    learning_curve_table,
    new_figure,
    plot_confusion_matrix,
    plot_denoising_scores,
    plot_impulse_response,
    plot_learning_curve,
    plot_roc_curve,
    plot_room_estimates,
    plot_sensitivity,
    roc_curve_table,
    room_estimates_table,
    save_figure,
    sensitivity_table,
)
from .geometry import canonical_layouts, equivalent_layouts, room_size_error
from .identification import (
    learning_curve_db,
    least_squares_response,
    lms,
    nlms,
    normalized_misalignment_db,
    prune_small_taps,
)
from .image_source import ImageSourceResult, image_source_rir, speed_of_sound
from .inversion import (
    RoomEstimate,
    bearing_family,
    best_estimate,
    closed_form_first_order,
    first_order_distances,
    invert_echoes,
    rank_candidates,
)
from .logging_utils import configure_logging
from .manifest import build_manifest, run_directory, write_json, write_manifest
from .models import MODEL_NAMES, build_model, load_model, save_model
from .pipeline import (
    denoising_pairs,
    invert_measurements,
    measurement_response,
    predict_inputs,
    prepare_features,
    response_features,
    train_model,
)
from .recording import (
    Recording,
    add_noise,
    measured_snr_db,
    read_recording,
    reverberate,
    write_recording,
)
from .rir_features import REPRESENTATIONS, decay_curve_db, represent_response
from .room_estimation import InversionResult, SceneResult, invert_response, simulate_scene
from .sensitivity import SWEEP_NAMES, run_sweeps, summarize_sweeps
from .synthetic import write_synthetic_dataset
from .template_matching import TemplateMatcher

__version__ = "0.1.0"

__all__ = [
    "DATASET_INDEX_NAME",
    "DENOISER_NAMES",
    "EXCITATION_KINDS",
    "FEATURE_SECTIONS",
    "FIGURE_FORMATS",
    "GEOMETRY_COLUMNS",
    "MODEL_NAMES",
    "REPRESENTATIONS",
    "SERIES_COLORS",
    "SWEEP_NAMES",
    "AcousticsConfig",
    "Config",
    "ConfigError",
    "CrossValidationResult",
    "DataConfig",
    "DatasetError",
    "DeepDenoisingAutoencoder",
    "DenoisingConfig",
    "EchoesConfig",
    "EvaluationConfig",
    "FeatureSet",
    "FeaturesConfig",
    "GeometryError",
    "IdentificationConfig",
    "ImageSourceResult",
    "InversionConfig",
    "InversionError",
    "InversionResult",
    "LinearCalibration",
    "MissingDependencyError",
    "ModeDecomposition",
    "ModelConfig",
    "ModelFileError",
    "OutputConfig",
    "PCADenoiser",
    "Recording",
    "RoomConfig",
    "RoomEstimate",
    "SceneResult",
    "SensitivityConfig",
    "SyntheticConfig",
    "TemplateMatcher",
    "__version__",
    "add_noise",
    "bearing_family",
    "best_estimate",
    "build_denoiser",
    "build_manifest",
    "build_model",
    "canonical_layouts",
    "classification_metrics",
    "closed_form_first_order",
    "compare_denoisers",
    "configure_logging",
    "confusion_matrix_table",
    "cross_validate",
    "decay_curve_db",
    "decompose_emd",
    "denoising_pairs",
    "denoising_scores_table",
    "denoising_summary",
    "equivalent_layouts",
    "first_order_distances",
    "image_source_rir",
    "impulse_response_table",
    "index_audio_folder",
    "invert_echoes",
    "invert_measurements",
    "invert_response",
    "items_table",
    "learning_curve_db",
    "learning_curve_table",
    "least_squares_response",
    "lms",
    "load_dataset_index",
    "load_features",
    "load_model",
    "load_svmlight",
    "make_excitation",
    "measured_snr_db",
    "measurement_response",
    "new_figure",
    "nlms",
    "normalized_misalignment_db",
    "per_sample_mse_db",
    "pick_echoes",
    "plot_confusion_matrix",
    "plot_denoising_scores",
    "plot_impulse_response",
    "plot_learning_curve",
    "plot_roc_curve",
    "plot_room_estimates",
    "plot_sensitivity",
    "predict_inputs",
    "prepare_features",
    "prune_small_taps",
    "rank_candidates",
    "read_recording",
    "represent_response",
    "require_module",
    "response_features",
    "reverberate",
    "roc_curve_table",
    "room_estimates_table",
    "room_size_error",
    "run_directory",
    "run_sweeps",
    "save_features",
    "save_figure",
    "save_model",
    "save_svmlight",
    "sensitivity_table",
    "simulate_scene",
    "speed_of_sound",
    "summarize_sweeps",
    "tap_shift_error",
    "taps_to_distances",
    "train_model",
    "write_json",
    "write_manifest",
    "write_recording",
    "write_synthetic_dataset",
]
