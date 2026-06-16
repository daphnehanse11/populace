"""Reform-validation payload assembly, isolated from policyengine-us.

The simulation is injected, so these tests exercise the budget-effect math and
the in-sample/out-of-sample split without running a Microsimulation.
"""

from __future__ import annotations

import json

import pytest

from populace.build.us.reform_validation import (
    REFORM_VALIDATION_SCHEMA_VERSION,
    ReformValidationSpec,
    in_sample_reform_specs,
    out_of_sample_reform_specs,
    reform_validation_payload,
    write_reform_validation,
)


class _FakeSeries:
    def __init__(self, total: float) -> None:
        self._total = total

    def sum(self) -> float:
        return self._total


class _FakeSim:
    """A sim whose weighted total for a measure shifts by a per-reform delta."""

    def __init__(self, totals: dict[str, float]) -> None:
        self._totals = totals

    def calculate(self, measure: str, period):  # noqa: ARG002
        return _FakeSeries(self._totals[measure])


def _oos_spec(score: float) -> ReformValidationSpec:
    return ReformValidationSpec(
        id="obbba_salt",
        name="OBBBA — SALT cap to $40k",
        category="OBBBA",
        in_sample=False,
        period=2024,
        jct_score=score,
        jct_window="FY2025-2034",
        jct_source="JCX-00-25",
        jct_source_url="https://www.jct.gov/",
        parameter_changes={"gov.example.cap": {"2025-01-01.2034-12-31": 40000}},
    )


def test_spec_requires_exactly_one_reform_definition():
    with pytest.raises(ValueError):
        ReformValidationSpec(
            id="x", name="x", category="c", in_sample=False, period=2024,
            jct_score=1.0, jct_window="", jct_source="", jct_source_url="",
        )
    with pytest.raises(ValueError):
        ReformValidationSpec(
            id="x", name="x", category="c", in_sample=False, period=2024,
            jct_score=1.0, jct_window="", jct_source="", jct_source_url="",
            neutralized_variable="v", parameter_changes={"a": 1},
        )


def test_in_sample_uses_calibration_estimate_no_simulation():
    specs = (
        ReformValidationSpec(
            id="nation/jct/mortgage", name="Mortgage interest deduction",
            category="JCT tax expenditure", in_sample=True, period=2024,
            jct_score=30e9, jct_window="annual", jct_source="JCT", jct_source_url="",
            neutralized_variable="mortgage_interest_deduction",
        ),
    )

    def simulate(_reform):  # pragma: no cover - must not be called
        raise AssertionError("in-sample rows must not simulate")

    payload = reform_validation_payload(
        specs,
        period=2024,
        simulate=simulate,
        in_sample_estimates={"nation/jct/mortgage": 28e9},
    )
    row = payload["reforms"][0]
    assert row["in_sample"] is True
    assert row["populace"]["budget_effect"] == pytest.approx(28e9)
    assert row["jct"]["score"] == pytest.approx(30e9)


def test_out_of_sample_budget_effect_is_reform_minus_baseline(monkeypatch):
    # baseline income_tax total 2.0e12; under the reform it rises by 50e9.
    spec = _oos_spec(score=-60e9)
    monkeypatch.setattr(spec.__class__, "build_reform", lambda self: "REFORM")

    def simulate(reform):
        total = 2.0e12 + 50e9 if reform == "REFORM" else 2.0e12
        return _FakeSim({"income_tax": total})

    payload = reform_validation_payload([spec], period=2024, simulate=simulate)
    row = payload["reforms"][0]
    assert row["in_sample"] is False
    assert row["populace"]["budget_effect"] == pytest.approx(50e9)
    assert row["populace"]["baseline_total"] == pytest.approx(2.0e12)
    assert row["populace"]["reform_total"] == pytest.approx(2.05e12)
    assert row["jct"]["score"] == pytest.approx(-60e9)


def test_counterfactual_revert_flips_sign():
    # A revert reform: baseline (provision on) income tax 2.0e12; reverting the
    # provision (reform) raises it to 2.033e12. The provision's effect is
    # baseline − reform = −33e9 (a cost), matching the JCT enactment sign.
    spec = _oos_spec(score=-33e9)
    object.__setattr__(spec, "effect_direction", "baseline_minus_reform")

    def simulate(reform):
        total = 2.033e12 if reform is not None else 2.0e12
        return _FakeSim({"income_tax": total})

    payload = reform_validation_payload([spec], period=2024, simulate=simulate)
    assert payload["reforms"][0]["populace"]["budget_effect"] == pytest.approx(-33e9)


def test_shipped_obbba_config_is_out_of_sample_counterfactual():
    specs = out_of_sample_reform_specs(period=2026)
    assert {s.id for s in specs} >= {"obbba_no_tax_on_tips", "obbba_no_tax_on_overtime"}
    for spec in specs:
        assert spec.effect_direction == "baseline_minus_reform"
        assert spec.period == 2026
        assert spec.jct_score < 0  # OBBBA provisions are costs
        assert spec.jct_source.startswith("JCX-35-25")


def test_out_of_sample_null_when_no_simulate():
    payload = reform_validation_payload([_oos_spec(-1.0)], period=2024, simulate=None)
    assert payload["reforms"][0]["populace"]["budget_effect"] is None
    assert payload["schema_version"] == REFORM_VALIDATION_SCHEMA_VERSION


def test_in_sample_specs_built_from_jct_reforms():
    specs = in_sample_reform_specs(period=2024)
    assert specs, "expected at least one JCT tax-expenditure reform"
    assert all(s.in_sample for s in specs)
    assert all(s.neutralized_variable for s in specs)


def test_out_of_sample_specs_load_from_default_config():
    # The shipped OBBBA config (if present) must parse into valid specs.
    specs = out_of_sample_reform_specs(period=2024)
    for spec in specs:
        assert spec.in_sample is False
        assert spec.parameter_changes
        assert spec.jct_source


def test_write_round_trips(tmp_path):
    payload = reform_validation_payload([_oos_spec(-1.0)], period=2024, simulate=None)
    path = write_reform_validation(payload, tmp_path / "reform_validation.json")
    assert json.loads(path.read_text())["reforms"][0]["id"] == "obbba_salt"
