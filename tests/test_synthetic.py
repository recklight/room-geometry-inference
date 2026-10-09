"""Synthetic measurements: reproducibility, file layout, and rooms and placements in range."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from room_geometry_inference import (
    GEOMETRY_COLUMNS,
    AcousticsConfig,
    SyntheticConfig,
    read_recording,
    write_synthetic_dataset,
)
from room_geometry_inference.synthetic import make_placement, make_room_size

SMALL = SyntheticConfig(n_rooms_per_class=2, n_recordings_per_room=2, duration_s=0.2)


def _files(folder):
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def test_the_same_seed_writes_identical_files(tmp_path):
    write_synthetic_dataset(tmp_path / "a", SMALL, rng=5)
    write_synthetic_dataset(tmp_path / "b", SMALL, rng=5)
    assert _files(tmp_path / "a") == _files(tmp_path / "b")


def test_another_seed_writes_different_files(tmp_path):
    write_synthetic_dataset(tmp_path / "a", SMALL, rng=5)
    write_synthetic_dataset(tmp_path / "b", SMALL, rng=6)
    assert _files(tmp_path / "a") != _files(tmp_path / "b")


def test_layout_index_and_groups(tmp_path):
    index = write_synthetic_dataset(tmp_path, SMALL, rng=0)
    table = pd.read_csv(index)
    assert table.columns.tolist() == ["path", "label", "group", *GEOMETRY_COLUMNS]
    assert table["path"].tolist() == [
        f"{label}/{label}_{k:03d}.wav" for label in ("large", "small") for k in range(4)
    ]
    assert table.groupby("group").size().tolist() == [2, 2, 2, 2]
    rooms = table.groupby("group")[["room_width_m", "room_depth_m"]].nunique()
    assert (rooms == 1).all().all()


def test_rooms_stay_near_their_nominal_size_and_placements_inside(tmp_path):
    table = pd.read_csv(write_synthetic_dataset(tmp_path, SMALL, rng=1))
    large = table[table["label"] == "large"]
    assert np.all(np.abs(large["room_width_m"] - 10.7) <= 0.3)
    for column, side in (("mic_x_m", "room_width_m"), ("source_y_m", "room_depth_m")):
        assert np.all((table[column] >= 0.3) & (table[column] <= table[side] - 0.3))
    distance = np.hypot(
        table["source_x_m"] - table["mic_x_m"], table["source_y_m"] - table["mic_y_m"]
    )
    assert np.all((distance > 0.79) & (distance < 1.61))


def test_measurements_are_two_channel_and_reverberant(tmp_path):
    index = write_synthetic_dataset(tmp_path, SMALL, rng=2, acoustics=AcousticsConfig())
    recording = read_recording(index.parent / "small" / "small_000.wav")
    assert recording.sample_rate == 16_000 and recording.excitation.size == 3200
    lag = np.argmax(np.correlate(recording.microphone[:2000], recording.excitation[:2000], "full"))
    assert lag - 1999 > 0  # the microphone hears the source after the direct-path delay


def test_placement_and_size_helpers():
    size = make_room_size((5.0, 4.0), 0.2, rng=0)
    assert np.all(np.abs(size - [5.0, 4.0]) <= 0.2)
    mic, source = make_placement(size, (1.0, 1.0), 0.0, 0.3, rng=0)
    assert source[1] == mic[1] and source[0] - mic[0] == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError, match="does not fit"):
        make_placement(np.array([1.0, 1.0]), (2.0, 2.0), 45.0, 0.3, rng=0)
