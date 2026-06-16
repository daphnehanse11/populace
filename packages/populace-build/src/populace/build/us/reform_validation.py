"""Reform validation: score policy reforms on the calibrated dataset and
compare the budget effect to the authority's (JCT's) official score.

Where ``calibration_diagnostics.json`` reports how well the calibrated weights
reproduce their *calibration targets*, ``reform_validation.json`` reports a
downstream property the calibration did not directly optimize: how closely the
dataset reproduces the *budget effects of scored policy reforms*. Two kinds of
reform are validated, and each row is labelled so a consumer can tell them
apart:

* **in-sample** — the JCT tax-expenditure reforms that are themselves
  calibration targets (``US_JCT_TAX_EXPENDITURE_REFORMS``). The dataset was
  tuned to hit these, so agreement is expected; the row is published for
  completeness and provenance, flagged ``in_sample=True``.
* **out-of-sample** — reforms the calibration never saw (e.g. provisions of
  the 2025 One Big Beautiful Bill Act), curated in ``obbba_reforms.json`` with
  their JCT scores. These are the genuine test of dataset fidelity.

The simulation is isolated behind an injected ``simulate`` callable so the
payload assembly is unit-testable without policyengine-us; the default factory
builds a ``Microsimulation`` over the freshly written release H5.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from populace.build.us.fiscal_targets import (
    US_JCT_TAX_EXPENDITURE_REFORMS,
    SimpleTaxExpenditureReform,
)

__all__ = [
    "REFORM_VALIDATION_SCHEMA_VERSION",
    "ReformValidationSpec",
    "in_sample_reform_specs",
    "out_of_sample_reform_specs",
    "load_default_reform_specs",
    "reform_validation_payload",
    "write_reform_validation",
]

#: Schema version of reform_validation.json. The calibration-diagnostics
#: dashboard keys its reader on it; bump with any shape change.
REFORM_VALIDATION_SCHEMA_VERSION = 1

#: The budget effect of a reform is the weighted-sum change of this variable
#: between the reform and baseline simulations, unless a spec overrides it. For
#: income-tax provisions this is the simulated federal income-tax revenue change
#: — the quantity JCT scores (− = revenue loss / cost; + = revenue raised).
DEFAULT_BUDGET_MEASURE = "income_tax"


@dataclass(frozen=True)
class ReformValidationSpec:
    """One reform to score on the dataset and compare to its JCT figure.

    Exactly one of ``neutralized_variable`` (an in-sample tax-expenditure
    neutralization) or ``parameter_changes`` (an out-of-sample
    ``Reform.from_dict`` payload) defines the reform.
    """

    id: str
    name: str
    category: str
    in_sample: bool
    period: int
    jct_score: float
    jct_window: str
    jct_source: str
    jct_source_url: str
    jct_score_type: str = "conventional"
    budget_measure: str = DEFAULT_BUDGET_MEASURE
    description: str = ""
    neutralized_variable: str | None = None
    parameter_changes: dict[str, Any] | None = None
    # How the budget effect is signed relative to the simulations. JCT scores a
    # *tax expenditure* as the revenue raised by repeal (reform − baseline, the
    # neutralize convention), but scores an *enacted provision* as the effect of
    # enacting it. OBBBA is already in the policyengine-us baseline, so its
    # provisions are validated by a counterfactual *revert* reform — there the
    # provision's effect is baseline − reform (JCT enactment sign).
    effect_direction: str = "reform_minus_baseline"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ReformValidationSpec.id is required.")
        has_neutralize = bool(self.neutralized_variable)
        has_params = bool(self.parameter_changes)
        if has_neutralize == has_params:
            raise ValueError(
                f"{self.id}: provide exactly one of neutralized_variable or "
                "parameter_changes."
            )
        if self.effect_direction not in {"reform_minus_baseline", "baseline_minus_reform"}:
            raise ValueError(
                f"{self.id}: effect_direction must be 'reform_minus_baseline' or "
                "'baseline_minus_reform'."
            )

    def build_reform(self) -> Any:
        """Construct the policyengine reform object for this spec.

        Imports are lazy so the module (and its unit tests) load without
        policyengine-us installed.
        """
        if self.neutralized_variable:
            from policyengine_core.reforms import Reform

            variable = self.neutralized_variable

            class _Neutralize(Reform):
                def apply(self) -> None:
                    self.neutralize_variable(variable)

            return _Neutralize
        from policyengine_core.reforms import Reform

        return Reform.from_dict(self.parameter_changes, country_id="us")


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def in_sample_reform_specs(
    reforms: Iterable[SimpleTaxExpenditureReform] = US_JCT_TAX_EXPENDITURE_REFORMS,
    *,
    period: int,
) -> tuple[ReformValidationSpec, ...]:
    """The JCT tax-expenditure calibration targets as validation specs."""
    specs: list[ReformValidationSpec] = []
    for reform in reforms:
        specs.append(
            ReformValidationSpec(
                id=reform.target_name,
                name=reform.target_name,
                category="JCT tax expenditure",
                in_sample=True,
                period=int(period),
                jct_score=float(reform.value),
                jct_window="annual",
                jct_source=reform.source,
                jct_source_url="",
                budget_measure=reform.output_variable or DEFAULT_BUDGET_MEASURE,
                neutralized_variable=reform.neutralized_variable,
            )
        )
    return tuple(specs)


def _obbba_config_path() -> Path:
    return Path(str(files(__package__).joinpath("obbba_reforms.json")))


def out_of_sample_reform_specs(
    path: Path | None = None,
    *,
    period: int,
) -> tuple[ReformValidationSpec, ...]:
    """Curated out-of-sample reforms (OBBBA provisions) from JSON config."""
    config_path = path or _obbba_config_path()
    if not config_path.exists():
        return ()
    payload = json.loads(config_path.read_text())
    specs: list[ReformValidationSpec] = []
    for raw in payload.get("reforms", ()):
        jct = raw.get("jct", {})
        specs.append(
            ReformValidationSpec(
                id=raw["id"],
                name=raw["name"],
                category=raw.get("category", "OBBBA"),
                in_sample=False,
                period=int(raw.get("period", period)),
                jct_score=float(jct["score"]),
                jct_window=str(jct.get("window", "")),
                jct_source=str(jct.get("source", "")),
                jct_source_url=str(jct.get("source_url", "")),
                jct_score_type=str(jct.get("score_type", "conventional")),
                budget_measure=str(raw.get("budget_measure", DEFAULT_BUDGET_MEASURE)),
                description=str(raw.get("description", "")),
                parameter_changes=raw["parameter_changes"],
                # OBBBA provisions are baked into the baseline, so the config
                # encodes a revert; the provision's effect is baseline − reform.
                effect_direction=str(raw.get("effect_direction", "baseline_minus_reform")),
            )
        )
    return tuple(specs)


def load_default_reform_specs(
    *,
    period: int,
    obbba_path: Path | None = None,
) -> tuple[ReformValidationSpec, ...]:
    """In-sample JCT tax expenditures + out-of-sample OBBBA provisions."""
    return (
        *in_sample_reform_specs(period=period),
        *out_of_sample_reform_specs(obbba_path, period=period),
    )


# A simulate(reform_or_None) -> object with .calculate(measure, period).sum().
SimulateFn = Callable[[Any], Any]


def _weighted_total(simulation: Any, measure: str, period: int) -> float:
    """Weighted population total of ``measure`` (MicroSeries .sum() is
    weight-aware in policyengine-us)."""
    return float(simulation.calculate(measure, period).sum())


def default_simulate_factory(dataset_path: Path) -> SimulateFn:
    """Build a simulate() that runs a Microsimulation over the release H5."""

    def simulate(reform: Any) -> Any:
        from policyengine_us import Microsimulation
        from policyengine_us.data import USSingleYearDataset

        dataset = USSingleYearDataset(file_path=str(dataset_path))
        if reform is None:
            return Microsimulation(dataset=dataset)
        return Microsimulation(dataset=dataset, reform=reform)

    return simulate


def reform_validation_payload(
    specs: Sequence[ReformValidationSpec],
    *,
    period: int,
    simulate: SimulateFn | None = None,
    in_sample_estimates: dict[str, float] | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Score each reform on the dataset and render the JSON-stable payload.

    In-sample reforms are JCT tax-expenditure *calibration targets*: their
    populace budget effect is the calibrated ``final_estimate`` the calibration
    already produced — passed in via ``in_sample_estimates`` (keyed by spec id),
    so no extra simulation is run for them. Out-of-sample reforms (OBBBA
    provisions) are simulated: a baseline is built once and each reform's budget
    effect is the weighted-sum change of its budget measure (reform − baseline).

    ``simulate`` is required only if any out-of-sample spec is present (or an
    in-sample estimate is missing); when absent, those rows publish a null
    budget effect rather than failing the build. The shape matches the
    calibration-diagnostics dashboard's reform_validation reader.
    """
    estimates = in_sample_estimates or {}
    baseline: Any = None
    baseline_totals: dict[tuple[int, str], float] = {}

    def baseline_total(measure: str, at_period: int) -> float:
        nonlocal baseline
        if baseline is None:
            baseline = simulate(None)  # type: ignore[misc]
        key = (at_period, measure)
        if key not in baseline_totals:
            baseline_totals[key] = _weighted_total(baseline, measure, at_period)
        return baseline_totals[key]

    def simulated_effect(spec: ReformValidationSpec) -> tuple[float | None, float | None, float | None]:
        if simulate is None:
            return None, None, None
        base = baseline_total(spec.budget_measure, spec.period)
        reform_total = _weighted_total(simulate(spec.build_reform()), spec.budget_measure, spec.period)
        raw = reform_total - base
        # A counterfactual revert measures the provision as baseline − reform.
        effect = raw if spec.effect_direction == "reform_minus_baseline" else -raw
        return effect, base, reform_total

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.in_sample and spec.id in estimates:
            effect: float | None = float(estimates[spec.id])
            base_total: float | None = None
            reform_total: float | None = None
        else:
            effect, base_total, reform_total = simulated_effect(spec)
        rows.append(
            {
                "id": spec.id,
                "name": spec.name,
                "category": spec.category,
                "in_sample": spec.in_sample,
                "period": spec.period,
                "description": spec.description or None,
                "jct": {
                    "score": _finite(spec.jct_score),
                    "score_type": spec.jct_score_type,
                    "window": spec.jct_window or None,
                    "source": spec.jct_source or None,
                    "source_url": spec.jct_source_url or None,
                },
                "populace": {
                    "budget_effect": None if effect is None else _finite(effect),
                    "period": spec.period,
                    "window": spec.jct_window or None,
                    "measure": spec.budget_measure,
                    "baseline_total": None if base_total is None else _finite(base_total),
                    "reform_total": None if reform_total is None else _finite(reform_total),
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": REFORM_VALIDATION_SCHEMA_VERSION,
        "baseline_period": int(period),
        "scoring_window": "see per-reform jct.window",
        "reforms": rows,
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def write_reform_validation(payload: dict[str, Any], path: Path | str) -> Path:
    """Write the reform-validation payload as ``reform_validation.json``."""
    path = Path(path)
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path
