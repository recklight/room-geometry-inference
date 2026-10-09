"""Sensitivity sweeps of the inversion."""

from __future__ import annotations

import numpy as np
import pytest

from room_geometry_inference import Config, run_sweeps, summarize_sweeps
from room_geometry_inference.sensitivity import (
    SWEEP_COLUMNS,
    sweep_delay_offset,
    sweep_snr,
    sweep_source_bearing,
    sweep_source_shift,
)


@pytest.fixture
def config():
    return Config().override(
        {
            "sensitivity.delay_offsets_samples": [0.0, 1.0, 10.0],
            "sensitivity.bearings_deg": [10.0, 45.0, 80.0, 225.0],
            "sensitivity.shifts_m": [-0.2, 0.0, 0.2, 5.0],
            "sensitivity.snr_db": [-10.0, 30.0],
            "sensitivity.n_trials": 2,
            "synthetic.duration_s": 1.0,
        }
    )


def test_the_error_grows_with_the_delay_offset(config):
    table = sweep_delay_offset(config)
    assert table.columns.tolist() == list(SWEEP_COLUMNS)
    errors = table["room_error_m"].to_numpy()
    assert errors[0] == pytest.approx(0.0, abs=1e-9) and np.all(np.diff(errors) > 0)
    assert table["tap_shift"].round(9).tolist() == [0.0, 1.0, 10.0]


def test_the_assumed_bearing_is_exact_and_others_cost_little(config):
    table = sweep_source_bearing(config)
    by_bearing = dict(zip(table["value"], table["room_error_m"], strict=True))
    assert by_bearing[45.0] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < by_bearing[10.0] < 0.2 and 0.0 < by_bearing[80.0] < 0.2
    assert table.loc[table["value"] == 225.0, "distance_error_m"].iloc[0] < 1e-9


def test_a_source_outside_the_room_is_reported(config):
    table = sweep_source_shift(config)
    assert table["parameter"].tolist() == ["x"] * 4 + ["y"] * 4
    assert (table["status"] == "outside_room").sum() == 2
    centered = table[(table["value"] == 0.0)]
    assert np.allclose(centered["room_error_m"], 0.0, atol=1e-9)


def test_low_snr_breaks_the_chain_and_high_snr_does_not(config):
    table = sweep_snr(config, rng=0)
    assert len(table) == 4
    errors = table.groupby("value")["room_error_m"].mean()
    assert errors[30.0] < 0.02 < errors[-10.0]


def test_every_configured_sweep_is_run_and_summarized(config):
    table = run_sweeps(config.override({"sensitivity.sweeps": ["delay_offset", "snr"]}), rng=0)
    assert table["sweep"].unique().tolist() == ["delay_offset", "snr"]
    summary = summarize_sweeps(table)
    assert summary.columns.tolist()[:3] == ["sweep", "parameter", "value"]
    snr_rows = summary[summary["sweep"] == "snr"]
    assert snr_rows["n_trials"].tolist() == [2, 2]


def test_the_snr_sweep_is_reproducible(config):
    first = sweep_snr(config.override({"sensitivity.snr_db": [20.0]}), rng=3)
    second = sweep_snr(config.override({"sensitivity.snr_db": [20.0]}), rng=3)
    assert first.equals(second)
