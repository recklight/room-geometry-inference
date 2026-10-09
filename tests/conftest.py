"""Shared fixtures. Every fixture is synthesized in code; no recorded data is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from room_geometry_inference import (
    Config,
    FeatureSet,
    image_source_rir,
    prepare_features,
    write_synthetic_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: The reference scenes: a large and a small room, the source 1 m off on both axes.
LARGE_ROOM = {"room_m": (10.7, 7.5), "microphone_m": (5.35, 3.75), "source_m": (6.35, 4.75)}
SMALL_ROOM = {"room_m": (3.5, 3.0), "microphone_m": (1.75, 1.5), "source_m": (2.75, 2.5)}


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def fast_config():
    return Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")


@pytest.fixture
def synthetic_dataset(tmp_path, fast_config):
    return write_synthetic_dataset(tmp_path / "data", fast_config.synthetic, rng=0)


@pytest.fixture(scope="session")
def quick_dataset(tmp_path_factory):
    """One synthetic dataset (quick settings) shared by the tests that only read it."""
    config = Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")
    return write_synthetic_dataset(
        tmp_path_factory.mktemp("shared") / "data", config.synthetic, rng=0
    )


@pytest.fixture(scope="session")
def quick_features(quick_dataset):
    """Features of :func:`quick_dataset`, prepared once."""
    config = Config.from_yaml(PROJECT_ROOT / "configs" / "quick_demo.yaml")
    return prepare_features(quick_dataset, config)


def simulate(scene: dict, sample_rate: int = 16_000, **options):
    """Image-source response of one of the reference scenes."""
    return image_source_rir(
        scene["room_m"],
        scene["source_m"],
        scene["microphone_m"],
        sample_rate=sample_rate,
        **options,
    )


def make_toy_features(
    rng: np.random.Generator,
    *,
    classes: tuple[str, ...] = ("large", "small"),
    rooms_per_class: int = 6,
    items_per_room: int = 3,
    separation: float = 3.0,
    with_groups: bool = True,
) -> FeatureSet:
    """Small random feature vectors whose class shifts the mean; for fast evaluation tests."""
    rows, labels, groups, paths = [], [], [], []
    for position, label in enumerate(classes):
        for room in range(rooms_per_class):
            for item in range(items_per_room):
                rows.append(separation * position + rng.normal(0.0, 1.0, size=6))
                labels.append(label)
                groups.append(f"{label}_room_{room}" if with_groups else "")
                paths.append(f"{label}/{label}_{room}_{item}.wav")
    return FeatureSet(
        X=np.stack(rows),
        y=np.asarray(labels),
        groups=np.asarray(groups),
        paths=np.asarray(paths),
        config=Config().to_dict(),
    )


@pytest.fixture
def make_toy():
    """The :func:`make_toy_features` factory, for tests that need a variant."""
    return make_toy_features
