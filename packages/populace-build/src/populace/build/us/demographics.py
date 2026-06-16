"""Demographic diagnostics: the dataset's weighted population by age band.

The fiscal-refresh release calibrates a tax/transfer target surface and does
*not* calibrate the age distribution, so the population by age is an emergent
property worth publishing and tracking: it tells a consumer how close the
weighted microdata is to the Census age structure, and whether that drifts
release over release. This is the demographic counterpart to
``calibration_diagnostics.json`` / ``reform_validation.json``.

The artifact, ``demographics.json``, carries the weighted population in each age
band, its population share, and — where a benchmark is published — the Census
national figure and the relative error. The microsimulation is isolated behind
small helpers so the payload assembly is unit-testable with plain arrays.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEMOGRAPHICS_SCHEMA_VERSION",
    "AGE_BANDS",
    "CENSUS_NATIONAL_AGE_BENCHMARK",
    "AgeBand",
    "compute_age_distribution",
    "demographics_payload",
    "population_by_age_from_sim",
    "write_demographics",
]

DEMOGRAPHICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgeBand:
    label: str
    min_age: int
    max_age: int | None  # None = open-ended (e.g. 75+)


#: Standard Census-style age bands.
AGE_BANDS: tuple[AgeBand, ...] = (
    AgeBand("0–4", 0, 4),
    AgeBand("5–17", 5, 17),
    AgeBand("18–24", 18, 24),
    AgeBand("25–34", 25, 34),
    AgeBand("35–44", 35, 44),
    AgeBand("45–54", 45, 54),
    AgeBand("55–64", 55, 64),
    AgeBand("65–74", 65, 74),
    AgeBand("75+", 75, None),
)

#: US resident population by age band — US Census Bureau, Annual Estimates of
#: the Resident Population (Vintage 2023), persons. Approximate round figures
#: for a directional benchmark; the band keys must match AGE_BANDS labels.
CENSUS_NATIONAL_AGE_BENCHMARK: dict[str, float] = {
    "0–4": 18_600_000,
    "5–17": 53_400_000,
    "18–24": 30_400_000,
    "25–34": 45_600_000,
    "35–44": 44_800_000,
    "45–54": 40_900_000,
    "55–64": 42_400_000,
    "65–74": 33_100_000,
    "75+": 25_000_000,
}
CENSUS_BENCHMARK_SOURCE = (
    "US Census Bureau, Annual Estimates of the Resident Population by age "
    "(Vintage 2023); approximate band totals"
)


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def compute_age_distribution(
    ages: np.ndarray,
    weights: np.ndarray,
    *,
    bands: tuple[AgeBand, ...] = AGE_BANDS,
    benchmark: dict[str, float] | None = CENSUS_NATIONAL_AGE_BENCHMARK,
) -> list[dict[str, Any]]:
    """Weighted population in each age band, with shares and benchmark error."""
    ages = np.asarray(ages, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    bench_total = float(sum(benchmark.values())) if benchmark else 0.0
    rows: list[dict[str, Any]] = []
    for band in bands:
        upper = band.max_age if band.max_age is not None else np.inf
        mask = (ages >= band.min_age) & (ages <= upper)
        population = float(weights[mask].sum())
        share = population / total if total else None
        bench = benchmark.get(band.label) if benchmark else None
        bench_share = (bench / bench_total) if (bench is not None and bench_total) else None
        rel_error = (
            (population - bench) / bench if (bench is not None and bench != 0) else None
        )
        rows.append(
            {
                "label": band.label,
                "min_age": band.min_age,
                "max_age": band.max_age,
                "population": _finite(population),
                "share": None if share is None else _finite(share),
                "benchmark": None if bench is None else _finite(bench),
                "benchmark_share": None if bench_share is None else _finite(bench_share),
                "relative_error": None if rel_error is None else _finite(rel_error),
            }
        )
    return rows


def population_by_age_from_sim(simulation: Any, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-person age and weight arrays from a policyengine-us Microsimulation."""
    age = simulation.calculate("age", period)
    return np.asarray(age.values), np.asarray(age.weights)


def demographics_payload(
    ages: np.ndarray,
    weights: np.ndarray,
    *,
    period: int,
    bands: tuple[AgeBand, ...] = AGE_BANDS,
    benchmark: dict[str, float] | None = CENSUS_NATIONAL_AGE_BENCHMARK,
    benchmark_source: str | None = CENSUS_BENCHMARK_SOURCE,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Render the weighted age distribution as a JSON-stable payload."""
    rows = compute_age_distribution(ages, weights, bands=bands, benchmark=benchmark)
    total = float(np.asarray(weights, dtype=np.float64).sum())
    bench_total = float(sum(benchmark.values())) if benchmark else None
    payload: dict[str, Any] = {
        "schema_version": DEMOGRAPHICS_SCHEMA_VERSION,
        "period": int(period),
        "measure": "person_weight",
        "total_population": _finite(total),
        "benchmark_total_population": None if bench_total is None else _finite(bench_total),
        "benchmark_source": benchmark_source if benchmark else None,
        "age_bands": rows,
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def write_demographics(payload: dict[str, Any], path: Path | str) -> Path:
    """Write the demographics payload as ``demographics.json``."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
