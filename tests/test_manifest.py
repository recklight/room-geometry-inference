"""Run folders, manifests and reproducible JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from room_geometry_inference import build_manifest, run_directory, write_json, write_manifest
from room_geometry_inference import manifest as manifest_module
from room_geometry_inference.manifest import array_sha256, file_sha256

MANIFEST_KEYS = [
    "command",
    "created_utc",
    "package_version",
    "git_commit",
    "seed",
    "versions",
    "inputs",
    "config",
    "details",
]


def test_run_directories_get_a_suffix_when_the_name_is_taken(tmp_path, monkeypatch):
    class FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(manifest_module, "datetime", FrozenClock)
    names = [run_directory(tmp_path / "runs", "evaluate").name for _ in range(3)]
    assert names == [
        "evaluate_20260102T030405Z",
        "evaluate_20260102T030405Z_1",
        "evaluate_20260102T030405Z_2",
    ]


def test_manifest_keys_and_input_fingerprints(tmp_path):
    present = tmp_path / "features.npz"
    present.write_bytes(b"abc")
    manifest = build_manifest(
        command="train",
        config={"seed": 3},
        inputs=[present, tmp_path / "absent.npz"],
        seed=3,
        extra={"n_items": np.int64(5)},
    )
    assert list(manifest) == MANIFEST_KEYS
    assert manifest["package_version"] == "0.1.0"
    assert list(manifest["versions"])[:2] == ["python", "platform"]
    assert "speechdsp" in manifest["versions"]
    assert manifest["inputs"][0] == {
        "path": present.as_posix(),
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "bytes": 3,
    }
    assert manifest["inputs"][1]["missing"] is True
    assert manifest["details"] == {"n_items": 5}
    target = write_manifest(tmp_path / "out" / "manifest.json", manifest)
    assert list(json.loads(target.read_text(encoding="utf-8"))) == MANIFEST_KEYS


def test_json_is_reproducible_with_nan_as_null(tmp_path):
    payload = {"uar": np.float64("nan"), "values": np.array([1.5, np.inf]), "path": Path("a/b")}
    first = write_json(tmp_path / "one.json", payload)
    second = write_json(tmp_path / "two.json", payload)
    assert first.read_bytes() == second.read_bytes()
    raw = first.read_bytes()
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    assert json.loads(raw) == {"uar": None, "values": [1.5, None], "path": "a/b"}


def test_digests_of_files_and_arrays(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"abc")
    assert file_sha256(sample).startswith("ba7816bf")
    assert array_sha256(np.zeros(3)) != array_sha256(np.zeros(3, dtype=np.float32))
    assert array_sha256(np.arange(4).reshape(2, 2)) != array_sha256(np.arange(4))
