"""Age-distribution payload assembly, isolated from policyengine-us."""

from __future__ import annotations

import json

import numpy as np
import pytest

from populace.build.us.demographics import (
    AGE_BANDS,
    DEMOGRAPHICS_SCHEMA_VERSION,
    compute_age_distribution,
    demographics_payload,
    write_demographics,
)


def test_bands_partition_ages_with_weighted_population():
    ages = np.array([2, 10, 40, 70, 90])
    weights = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    rows = compute_age_distribution(ages, weights, benchmark=None)
    by_label = {r["label"]: r for r in rows}
    assert by_label["0–4"]["population"] == 100.0
    assert by_label["5–17"]["population"] == 200.0
    assert by_label["35–44"]["population"] == 300.0
    assert by_label["65–74"]["population"] == 400.0
    assert by_label["75+"]["population"] == 500.0  # open-ended upper band
    # shares sum to 1 over the full population.
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)


def test_benchmark_relative_error_is_signed():
    ages = np.array([10, 10, 10])  # all in 5–17
    weights = np.array([20_000_000.0, 20_000_000.0, 20_000_000.0])  # 60M in 5–17
    rows = compute_age_distribution(ages, weights, benchmark={"5–17": 50_000_000})
    row = next(r for r in rows if r["label"] == "5–17")
    assert row["population"] == 60_000_000
    assert row["benchmark"] == 50_000_000
    assert row["relative_error"] == pytest.approx(0.2)  # 60M vs 50M = +20% over


def test_payload_shape_and_totals():
    ages = np.array([3, 30, 80])
    weights = np.array([10.0, 20.0, 30.0])
    payload = demographics_payload(ages, weights, period=2024, release_id="rel-a", benchmark=None)
    assert payload["schema_version"] == DEMOGRAPHICS_SCHEMA_VERSION
    assert payload["period"] == 2024
    assert payload["total_population"] == 60.0
    assert payload["release_id"] == "rel-a"
    assert len(payload["age_bands"]) == len(AGE_BANDS)


def test_write_round_trips(tmp_path):
    payload = demographics_payload(np.array([5]), np.array([1.0]), period=2024)
    path = write_demographics(payload, tmp_path / "demographics.json")
    loaded = json.loads(path.read_text())
    assert loaded["age_bands"][1]["label"] == "5–17"
