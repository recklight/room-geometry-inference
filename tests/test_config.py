"""Configuration defaults, YAML round trips and validation messages."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from room_geometry_inference import (
    FEATURE_SECTIONS,
    Config,
    ConfigError,
    build_denoiser,
    pick_echoes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))


def test_default_yaml_equals_the_dataclass_defaults():
    assert Config.from_yaml(PROJECT_ROOT / "configs" / "default.yaml") == Config()


def test_default_yaml_lists_every_key_in_dataclass_order():
    text = (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("# Default configuration.")
    assert lines[1].startswith("# Override single values on the command line with")
    keys = [line.split(":")[0].strip() for line in lines[2:] if not line.lstrip().startswith("#")]
    expected = []
    for name, value in Config().to_dict().items():
        expected.append(name)
        if isinstance(value, dict):
            expected.extend(value)
    assert keys == expected


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
def test_every_configuration_file_loads(path):
    assert isinstance(Config.from_yaml(path), Config)


def test_quick_demo_points_at_the_synthetic_dataset():
    config = Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")
    assert config.data.dataset == "data/synthetic/dataset.csv"
    first = (PROJECT_ROOT / "configs" / "quick_demo.yaml").read_text(encoding="utf-8")
    assert first.startswith(
        "# Small, fast settings for synthetic data (examples, smoke tests, CI)."
    )


def test_yaml_round_trip_keeps_every_value(tmp_path):
    config = Config().override(
        {
            "seed": 7,
            "synthetic.room_sizes_m": [[9.0, 6.0], [4.0, 3.5]],
            "denoising.architectures": [[64, 32], [16]],
            "acoustics.temperature_c": 25.0,
            "model.svm_gamma": 0.5,
        }
    )
    assert Config.from_yaml(config.to_yaml(tmp_path / "nested" / "config.yaml")) == config
    assert config.denoising.architectures == ((64, 32), (16,))


def test_the_speed_of_sound_follows_the_temperature_when_set():
    assert Config().acoustics.speed == 343.0
    warm = Config().override({"acoustics.temperature_c": 30.0})
    assert warm.acoustics.speed == pytest.approx(349.0, abs=0.2)


def test_feature_sections_are_the_ones_prepare_depends_on():
    assert FEATURE_SECTIONS == ("acoustics", "identification", "echoes", "features")


def test_unknown_key_is_rejected_and_the_valid_keys_listed():
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['n_echo'\].*valid keys are"):
        Config.from_dict({"echoes": {"n_echo": 9}})


def test_unknown_top_level_setting_is_rejected():
    with pytest.raises(ConfigError, match="valid top-level keys"):
        Config.from_dict({"echo": {}})


def test_override_rejects_unknown_sections_and_keys():
    with pytest.raises(ConfigError, match="unknown configuration section"):
        Config().override({"rooms.size_m": [3, 3]})
    with pytest.raises(ConfigError, match=r"unknown configuration key 'model\.kernel'"):
        Config().override({"model.kernel": "rbf"})


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        ("seed", -1, "seed must be a non-negative integer"),
        ("synthetic.class_names", ["large"], "synthetic.class_names"),
        ("synthetic.class_names", ["a", "a"], "distinct names"),
        ("synthetic.room_sizes_m", [[10.0, 7.0]], "one [width, depth] per class"),
        ("synthetic.room_sizes_m", [[10.0, 7.0], [2.0, 1.5]], "too small"),
        ("synthetic.source_distance_range_m", [1.5, 0.5], "low <= high"),
        ("synthetic.excitation", "chirp", "synthetic.excitation='chirp'"),
        ("synthetic.noise", "speech_like", "synthetic.noise"),
        ("synthetic.duration_s", 0.05, "not more than identification.n_taps"),
        ("acoustics.sample_rate", 100, "acoustics.sample_rate"),
        ("acoustics.order_mode", "manhattan", "acoustics.order_mode"),
        ("acoustics.reflection", 1.5, "acoustics.reflection must lie in"),
        ("acoustics.delay", "exact", "acoustics.delay"),
        ("room.microphone_m", [11.0, 3.0], "not strictly inside"),
        ("room.source_m", [5.35, 3.75], "coincide"),
        ("room.size_m", [10.0], "must have 2 values"),
        ("identification.method", "rls", "identification.method"),
        ("identification.step", 2.0, "(0, 2) for NLMS"),
        ("identification.prune_threshold", 1.0, "identification.prune_threshold"),
        ("echoes.n_echoes", 4, "echoes.n_echoes"),
        ("echoes.method", "energy", "echoes.method"),
        ("echoes.compensate_spreading", "yes", "echoes.compensate_spreading"),
        ("inversion.rank_metric", "cosine", "inversion.rank_metric"),
        ("inversion.refine", 1, "inversion.refine"),
        ("sensitivity.sweeps", ["room_size"], "sensitivity.sweeps"),
        ("sensitivity.snr_db", [], "sensitivity.snr_db"),
        ("denoising.methods", ["wiener"], "denoising.methods"),
        ("denoising.architectures", [[]], "layer widths"),
        ("denoising.device", "tpu", "denoising.device"),
        ("features.representation", "mfcc", "features.representation='mfcc'"),
        ("features.floor_db", 3.0, "features.floor_db must be negative"),
        ("model.name", "forest", "model.name='forest'"),
        ("model.svm_gamma", "wide", "model.svm_gamma"),
        ("model.svm_gamma", -1.0, "model.svm_gamma must be strictly positive"),
        ("model.name", "template", "needs features.representation='magnitude'"),
        ("model.n_neighbors", "five", "model.n_neighbors"),
        ("evaluation.n_splits", 1, "evaluation.n_splits"),
        ("evaluation.positive_class", "", "evaluation.positive_class"),
        ("output.figure_format", "gif", "output.figure_format"),
        ("identification.n_taps", 4, "identification.n_taps"),
    ],
)
def test_invalid_values_name_the_dotted_key(key, value, fragment):
    escaped = (
        fragment.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[").replace("]", r"\]")
    )
    with pytest.raises(ConfigError, match=escaped):
        Config().override({key: value})


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        Config.from_yaml(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("model: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not valid YAML"):
        Config.from_yaml(broken)
    listing = tmp_path / "listing.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a mapping"):
        Config.from_yaml(listing)


def test_a_section_must_be_a_mapping():
    with pytest.raises(ConfigError, match="section 'model' must be a mapping"):
        Config.from_dict({"model": 3})


def test_configurations_are_hashable_and_comparable():
    assert hash(Config()) == hash(Config())
    assert Config().override({"seed": 1}) != Config()
    assert Config().echoes.options()["n_echoes"] == 9


def test_section_options_carry_every_setting_their_functions_take():
    echo_parameters = set(inspect.signature(pick_echoes).parameters) - {"response"}
    assert set(Config().echoes.options()) == echo_parameters
    denoiser_parameters = set(inspect.signature(build_denoiser).parameters)
    assert set(Config().denoising.options()) == denoiser_parameters - {
        "name",
        "hidden_units",
        "random_state",
    }
