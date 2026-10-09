"""Dataset index reading, folder indexing, features.npz and LIBSVM export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from room_geometry_inference import (
    GEOMETRY_COLUMNS,
    Config,
    ConfigError,
    DatasetError,
    FeatureSet,
    index_audio_folder,
    items_table,
    load_dataset_index,
    load_features,
    load_svmlight,
    save_features,
    save_svmlight,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_paths_are_relative_to_the_index_folder_not_the_working_directory(tmp_path, monkeypatch):
    _touch(tmp_path / "data" / "large" / "one.wav")
    index = tmp_path / "data" / "dataset.csv"
    index.write_text("path,label,group\nlarge/one.wav,large,room_1\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    table = load_dataset_index(index)
    assert Path(table.loc[0, "path"]) == (tmp_path / "data" / "large" / "one.wav").resolve()
    assert table.loc[0, "source"] == "large/one.wav"
    assert list(table.columns[:4]) == ["path", "source", "label", "group"]


def test_windows_separators_are_accepted(tmp_path):
    _touch(tmp_path / "x" / "one.wav")
    (tmp_path / "dataset.csv").write_text("path,label\nx\\one.wav,x\n", encoding="utf-8")
    table = load_dataset_index(tmp_path / "dataset.csv")
    assert table.loc[0, "source"] == "x/one.wav" and table.loc[0, "group"] == ""


def test_geometry_columns_are_numbers_and_complete(tmp_path):
    _touch(tmp_path / "a.wav")
    header = ",".join(["path", "label", *GEOMETRY_COLUMNS])
    (tmp_path / "dataset.csv").write_text(f"{header}\na.wav,x,6,4,2,1,3,2\n", encoding="utf-8")
    table = load_dataset_index(tmp_path / "dataset.csv")
    assert table["room_width_m"].dtype == np.float64 and table.loc[0, "source_y_m"] == 2.0
    (tmp_path / "bad.csv").write_text(f"{header}\na.wav,x,6,4,2,one,3,2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="'mic_y_m' is not a number on line"):
        load_dataset_index(tmp_path / "bad.csv")
    (tmp_path / "part.csv").write_text("path,label,room_width_m\na.wav,x,6\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="lacks"):
        load_dataset_index(tmp_path / "part.csv")


def test_the_label_is_optional_for_inversion(tmp_path):
    _touch(tmp_path / "a.wav")
    (tmp_path / "dataset.csv").write_text("path\na.wav\n", encoding="utf-8")
    assert load_dataset_index(tmp_path / "dataset.csv", require_label=False)["label"].tolist() == [
        ""
    ]
    with pytest.raises(DatasetError, match=r"lacks the column\(s\) \['label'\]"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_missing_columns_are_named_with_the_columns_found(tmp_path):
    (tmp_path / "dataset.csv").write_text("file,class\na.wav,x\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"lacks the column\(s\) \['path', 'label'\]; found"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_missing_files_are_listed_five_at_most(tmp_path):
    rows = "\n".join(f"missing_{number}.wav,x" for number in range(8))
    (tmp_path / "dataset.csv").write_text(f"path,label\n{rows}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"8 file\(s\).*missing_4\.wav \.\.\. and 3 more"):
        load_dataset_index(tmp_path / "dataset.csv")


def test_empty_values_and_missing_index_are_explained(tmp_path):
    _touch(tmp_path / "a.wav")
    (tmp_path / "dataset.csv").write_text("path,label\na.wav,\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"empty 'label' on line\(s\) \[2\]"):
        load_dataset_index(tmp_path / "dataset.csv")
    with pytest.raises(FileNotFoundError, match="synthesize"):
        load_dataset_index(tmp_path / "absent.csv")
    (tmp_path / "empty.csv").write_text("path,label\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="lists no recordings"):
        load_dataset_index(tmp_path / "empty.csv")


def test_folder_layout_is_indexed_with_one_group_per_file(tmp_path):
    for name in ("small/b.wav", "small/a.WAV", "large/c.wav", "notes.txt", "small/x.txt"):
        _touch(tmp_path / name)
    table = pd.read_csv(index_audio_folder(tmp_path))
    assert table["path"].tolist() == ["large/c.wav", "small/a.WAV", "small/b.wav"]
    assert table["label"].tolist() == ["large", "small", "small"]
    assert table["group"].tolist() == ["large/c", "small/a", "small/b"]
    with pytest.raises(FileNotFoundError):
        index_audio_folder(tmp_path / "absent")


def test_an_edited_index_is_kept_unless_overwrite_is_requested(tmp_path):
    for name in ("small/a.wav", "large/b.wav"):
        _touch(tmp_path / name)
    index = index_audio_folder(tmp_path)
    edited = index.read_text(encoding="utf-8").replace("large/b\n", "hall_A\n")
    index.write_text(edited, encoding="utf-8")
    with pytest.raises(DatasetError, match="already exists"):
        index_audio_folder(tmp_path)
    assert index.read_text(encoding="utf-8") == edited
    index_audio_folder(tmp_path, overwrite=True)
    assert pd.read_csv(index)["group"].tolist() == ["large/b", "small/a"]


def _features() -> FeatureSet:
    return FeatureSet(
        X=np.arange(12, dtype=np.float64).reshape(3, 4),
        y=np.array(["large", "small", "large"]),
        groups=np.array(["r1", "r2", "r1"]),
        paths=np.array(["large/a.wav", "small/b.wav", "large/c.wav"]),
        config=Config().to_dict(),
    )


def test_features_round_trip_and_are_byte_identical(tmp_path):
    original = _features()
    first = save_features(tmp_path / "one" / "features.npz", original)
    second = save_features(tmp_path / "two" / "features.npz", original)
    assert first.read_bytes() == second.read_bytes()
    loaded = load_features(first)
    for name in ("X", "y", "groups", "paths"):
        assert np.array_equal(getattr(loaded, name), getattr(original, name)), name
    assert loaded.config == original.config and loaded.classes == ["large", "small"]
    with np.load(first, allow_pickle=False) as archive:
        assert sorted(archive.files) == sorted(["X", "y", "groups", "paths", "config_json"])
        assert archive["config_json"].ndim == 0


def test_items_table_has_one_row_per_measurement():
    assert list(items_table(_features()).columns) == ["path", "label", "group"]


def test_changed_feature_settings_are_named(tmp_path):
    path = save_features(tmp_path / "features.npz", _features())
    assert load_features(path, Config()).n_items == 3
    assert load_features(path, Config().override({"model.svm_c": 1.0})).n_items == 3
    changed = Config().override({"features.representation": "magnitude"})
    with pytest.raises(ConfigError, match=r"features\.representation;"):
        load_features(path, changed)


def test_foreign_or_missing_feature_files_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'prepare' first"):
        load_features(tmp_path / "absent.npz")
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip archive")
    with pytest.raises(DatasetError, match="not a readable features file"):
        load_features(garbage)
    partial = tmp_path / "partial.npz"
    np.savez(partial, X=np.zeros((1, 2)))
    with pytest.raises(DatasetError, match="missing"):
        load_features(partial)


def test_feature_sets_check_their_lengths():
    with pytest.raises(DatasetError, match="one entry per row"):
        FeatureSet(
            X=np.zeros((2, 3)),
            y=np.array(["a"]),
            groups=np.array(["", ""]),
            paths=np.array(["a", "b"]),
            config={},
        )


def test_libsvm_export_round_trips(tmp_path):
    target = save_svmlight(tmp_path / "features.svm", _features())
    X, y = load_svmlight(target, n_features=4)
    assert np.allclose(X, _features().X) and y.tolist() == [0, 1, 0]
    assert target.read_text().splitlines()[0].startswith("0 2:1")


def test_the_synthetic_index_loads_with_absolute_paths_and_geometry(synthetic_dataset):
    table = load_dataset_index(synthetic_dataset)
    assert len(table) == 16 and table["label"].nunique() == 2 and table["group"].nunique() == 8
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in table["path"])
    assert table["source"].str.match(r"^(large|small)/\1_\d{3}\.wav$").all()
    assert set(GEOMETRY_COLUMNS) <= set(table.columns)
