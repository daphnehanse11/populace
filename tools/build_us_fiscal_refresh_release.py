"""Build a contract-valid US fiscal refresh release from a Populace H5.

This is a narrow release builder for the Issue #40 fiscal target surface. It
starts from an existing Populace US H5, materializes the current structured
fiscal target rows, recalibrates only the household weights, writes a fresh
PolicyEngine-US H5, and emits the release contract files required by
``populace-publish-release``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.us import (
    US_FISCAL_TARGET_SPECS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    hard_target_package_aliases,
    us_source_coverage_diagnostics,
    write_us_source_coverage_diagnostics,
)
from populace.build.us.demographics import (
    demographics_payload,
    population_by_age_from_sim,
    write_demographics,
)
from populace.calibrate import TargetRegistry, calibrate
from populace.calibrate.diagnostics import (
    diagnostics_payload,
    write_calibration_diagnostics,
)
from populace.frame import Frame, MassChange, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine
from populace.frame.units import US_SCHEMA

PERIOD = 2024
REPO_ID = "policyengine/populace-us"
DATASET_FILENAME = "populace_us_2024.h5"
CALIBRATION_FILENAME = "populace_us_2024_calibration.npz"

DIRECT_ACTIVE_ALIASES = (
    "census-stc-individual-income-tax",
    "soi-table-1-1",
    "soi-table-1-2",
    "soi-table-1-4",
    "soi-table-2-1",
    "soi-table-2-5",
    "soi-table-2-5-eitc-agi-children-2022",
    "soi-table-4-3",
    "soi-state-2022",
    "soi-historic-table-2",
    "soi-historic-table-2-state-agi-2022",
    "soi-historic-table-2-state-broad-2022",
    "soi-historic-table-2-state-eitc-2022",
    "soi-w2-statistics-2020",
)

REVIEWED_EXCLUDED_ALIASES = (
    "bea-nipa-pension-contributions",
    "bea-nipa-personal-income-components",
    "bea-nipa-personal-income-disposition",
    "bea-nipa-total-wages-salaries",
    "census-acs-s0101-congressional-district-age-2024",
    "census-acs-s0101-national-age-2024",
    "census-acs-s0101-state-age-2024",
    "census-pep-2024-national-age-sex",
    "census-pep-2024-state-age-sex",
    "cms-aca-effectuated-enrollment-2022",
    "cms-aca-oep-state-level",
    "cms-aca-oep-state-level-2022",
    "cms-aca-oep-state-level-2025",
    "cms-medicaid-chip-monthly-enrollment-dataset",
    "cms-medicaid-chip-monthly-enrollment-december-2024",
    "cms-medicare-trustees-report-2025-part-b-premium-income",
    "cms-nhe-historical-service-source",
    "hhs-acf-liheap-fy2023-national-profile",
    "hhs-acf-liheap-fy2024-national-profile",
    "hhs-acf-tanf-caseload-2024",
    "hhs-acf-tanf-financial-2024",
    "ssa-annual-statistical-supplement-2025",
    "ssa-ssi-table-7b1-2024",
    "usda-snap-fy69-to-current",
)

FISCAL_TARGET_SOURCE_KEYS = {
    "cbo": "Congressional Budget Office revenue projections",
    "irs_soi": "IRS Statistics of Income public tables",
    "jct": "Joint Committee on Taxation tax expenditure estimates",
    "state_income_tax": "Census State Tax Collections individual income tax",
}

SOI_VARIABLE_MAP = {
    "adjusted_gross_income": "adjusted_gross_income",
    "business_net_profits": "self_employment_income",
    "business_net_losses": "self_employment_income",
    "capital_gains_distributions": "capital_gains",
    "capital_gains_gross": "capital_gains",
    "capital_gains_losses": "capital_losses",
    "employment_income": "employment_income",
    "estate_income": "estate_income",
    "estate_losses": "estate_income",
    "exempt_interest": "tax_exempt_interest_income",
    "ira_distributions": "taxable_ira_distributions",
    "ordinary_dividends": "dividend_income",
    "partnership_and_s_corp_income": "tax_unit_partnership_s_corp_income",
    "partnership_and_s_corp_losses": "tax_unit_partnership_s_corp_income",
    "qualified_dividends": "qualified_dividend_income",
    "rent_and_royalty_net_income": "rent_and_royalty_net_income",
    "rent_and_royalty_net_losses": "rent_and_royalty_net_income",
    "taxable_interest_income": "taxable_interest_income",
    "taxable_pension_income": "taxable_pension_income",
    "taxable_social_security": "tax_unit_taxable_social_security",
    "total_pension_income": "pension_income",
    "total_social_security": "tax_unit_social_security",
    "unemployment_compensation": "unemployment_compensation",
}

FILING_STATUS_MAP = {
    "All": None,
    "Head of Household": "HEAD_OF_HOUSEHOLD",
    "Married Filing Jointly/Surviving Spouse": {
        "JOINT",
        "SURVIVING_SPOUSE",
    },
    "Married Filing Separately": "SEPARATE",
    "Single": "SINGLE",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-h5",
        type=Path,
        help="Existing Populace US H5 to recalibrate. Defaults to HF latest.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-id")
    parser.add_argument("--epochs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--max-weight-ratio", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-demographics",
        action="store_true",
        help="Do not emit demographics.json (weighted population by age) for this release.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    packages = (
        "populace-build",
        "populace-calibrate",
        "populace-data",
        "populace-frame",
        "policyengine-core",
        "policyengine-us",
        "numpy",
        "pandas",
        "torch",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def _git_dirty() -> bool:
    return bool(_git_output("status", "--porcelain"))


def _download_base_h5() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=DATASET_FILENAME,
            repo_type="dataset",
        )
    )


def _load_frame(path: Path) -> Frame:
    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(path))
    tables = {
        "person": dataset.person.copy(),
        "household": dataset.household.copy(),
        "tax_unit": dataset.tax_unit.copy(),
        "spm_unit": dataset.spm_unit.copy(),
        "family": dataset.family.copy(),
        "marital_unit": dataset.marital_unit.copy(),
    }
    weights = tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


def _with_exportable_formula_inputs(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables.get("person")
    if person is not None and "partnership_s_corp_income" in person.columns:
        combined = person["partnership_s_corp_income"].to_numpy(dtype=np.float64)
        has_partnership = "partnership_income" in person.columns
        has_s_corp = "s_corp_income" in person.columns
        if not has_partnership and not has_s_corp:
            person["partnership_income"] = combined
            person["s_corp_income"] = np.zeros(len(person), dtype=np.float64)
        elif not has_partnership:
            person["partnership_income"] = combined - person["s_corp_income"].to_numpy(
                dtype=np.float64
            )
        elif not has_s_corp:
            person["s_corp_income"] = combined - person["partnership_income"].to_numpy(
                dtype=np.float64
            )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def _drop_formula_owned_columns(frame: Frame) -> Frame:
    adapter = PolicyEngineUSEngine()
    frame = _with_exportable_formula_inputs(frame)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    formula_owned = adapter._engine_computed_columns(tables, period=PERIOD)
    if not formula_owned:
        return frame
    stripped_tables = {
        entity: table.drop(
            columns=[column for column in formula_owned if column in table.columns]
        )
        for entity, table in tables.items()
    }
    return Frame(
        stripped_tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def _dataset_from_frame(
    frame: Frame,
    *,
    zero_variables: Iterable[str] = (),
    system=None,
):
    from policyengine_us.data import USSingleYearDataset

    frame = _drop_formula_owned_columns(frame)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for variable_name in zero_variables:
        if system is None:
            raise ValueError("system is required when zero_variables are provided.")
        entity = _variable_entity(system, variable_name)
        if entity is not None and variable_name in tables[entity]:
            tables[entity][variable_name] = 0
    tables["household"]["household_weight"] = frame.weights_for("household").values
    return USSingleYearDataset(
        person=tables["person"],
        household=tables["household"],
        tax_unit=tables["tax_unit"],
        spm_unit=tables["spm_unit"],
        family=tables["family"],
        marital_unit=tables["marital_unit"],
        time_period=PERIOD,
    )


def _calculate_array(
    simulation, variable: str, *, map_to: str | None = None
) -> np.ndarray:
    kwargs: dict[str, Any] = {"period": PERIOD}
    if map_to is not None:
        kwargs["map_to"] = map_to
    return np.asarray(simulation.calculate(variable, **kwargs))


def _tax_unit_to_household_positions(frame: Frame) -> np.ndarray:
    person = frame.table("person")
    household_ids = frame.table("household")["household_id"].to_numpy()
    tax_unit_ids = frame.table("tax_unit")["tax_unit_id"].to_numpy()

    membership = person[["person_tax_unit_id", "person_household_id"]].drop_duplicates()
    counts = membership.groupby("person_tax_unit_id")["person_household_id"].nunique()
    ambiguous = counts[counts != 1]
    if not ambiguous.empty:
        raise ValueError(
            "Tax units must be nested in households; ambiguous tax_unit_id "
            f"examples: {ambiguous.index[:5].tolist()}."
        )
    tax_to_household = (
        membership.drop_duplicates("person_tax_unit_id")
        .set_index("person_tax_unit_id")["person_household_id"]
        .reindex(tax_unit_ids)
    )
    if tax_to_household.isna().any():
        missing = tax_unit_ids[tax_to_household.isna().to_numpy()][:5].tolist()
        raise ValueError(f"Tax units with no person membership: {missing}.")
    household_positions = pd.Series(
        np.arange(len(household_ids), dtype=np.int64), index=household_ids
    )
    positions = household_positions.reindex(tax_to_household.to_numpy()).to_numpy()
    if np.isnan(positions).any():
        raise ValueError("Tax-unit household ids are not present in household table.")
    return positions.astype(np.int64)


def _collapse_tax_unit(
    values: np.ndarray, positions: np.ndarray, n_households: int
) -> np.ndarray:
    out = np.zeros(n_households, dtype=np.float64)
    np.add.at(out, positions, np.asarray(values, dtype=np.float64))
    return out


def _collapse_person(frame: Frame, values: np.ndarray) -> np.ndarray:
    household_ids = frame.table("household")["household_id"].to_numpy()
    person_households = frame.table("person")["person_household_id"].to_numpy()
    household_positions = pd.Series(
        np.arange(len(household_ids), dtype=np.int64), index=household_ids
    )
    positions = (
        household_positions.reindex(person_households).to_numpy().astype(np.int64)
    )
    out = np.zeros(len(household_ids), dtype=np.float64)
    np.add.at(out, positions, np.asarray(values, dtype=np.float64))
    return out


def _variable_entity(system, name: str) -> str | None:
    variable = system.variables.get(name)
    if variable is None:
        return None
    return variable.entity.key


def _household_values(
    *,
    frame: Frame,
    simulation,
    system,
    variable: str,
    tax_unit_positions: np.ndarray,
) -> np.ndarray:
    entity = _variable_entity(system, variable)
    if entity is None:
        raise KeyError(variable)
    if entity == "household":
        return _calculate_array(simulation, variable)
    if entity == "tax_unit":
        return _collapse_tax_unit(
            _calculate_array(simulation, variable),
            tax_unit_positions,
            frame.n("household"),
        )
    if entity == "person":
        return _collapse_person(frame, _calculate_array(simulation, variable))
    raise ValueError(f"Unsupported variable entity {entity!r} for {variable!r}.")


def _filing_status_names(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [getattr(value, "name", str(value)) for value in values], dtype=object
    )


def _as_bound(value: str) -> float:
    if value == "-inf":
        return -math.inf
    if value == "inf":
        return math.inf
    return float(value)


def _signed_component(values: np.ndarray, source_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if source_name == "adjusted_gross_income":
        return values
    if source_name == "capital_gains_losses":
        return np.maximum(values, 0.0)
    if source_name.endswith("_losses") or source_name in {
        "business_net_losses",
        "capital_gains_losses",
        "estate_losses",
        "partnership_and_s_corp_losses",
        "rent_and_royalty_net_losses",
    }:
        return np.minimum(values, 0.0) * -1.0
    return np.maximum(values, 0.0)


def _soi_component_row(
    values: np.ndarray, source_name: str, *, count: bool
) -> np.ndarray:
    component = _signed_component(values, source_name)
    if count:
        return (component > 0).astype(np.float64)
    return component


def _person_variable_to_tax_unit(*, frame: Frame, values: np.ndarray) -> np.ndarray:
    person = frame.table("person")
    tax_unit_ids = frame.table("tax_unit")["tax_unit_id"].to_numpy()
    tax_positions = (
        pd.Series(np.arange(len(tax_unit_ids), dtype=np.int64), index=tax_unit_ids)
        .reindex(person["person_tax_unit_id"].to_numpy())
        .to_numpy()
    )
    out = np.zeros(len(tax_unit_ids), dtype=np.float64)
    np.add.at(out, tax_positions.astype(np.int64), np.asarray(values, dtype=np.float64))
    return out


def _make_zero_variable_reform(system, variable_name: str):
    from policyengine_us.model_api import Reform, Variable

    original = system.variables[variable_name]

    class NeutralizedVariable(Variable):
        value_type = original.value_type
        entity = original.entity
        label = f"Neutralized {variable_name}"
        definition_period = original.definition_period
        unit = getattr(original, "unit", None)
        adds = None
        subtracts = None
        uprating = None

        def formula(self, period, parameters):  # pragma: no cover - exercised in PE
            return 0

    NeutralizedVariable.__name__ = variable_name

    class NeutralizeVariableReform(Reform):
        def apply(self):
            self.replace_variable(NeutralizedVariable)

    NeutralizeVariableReform.__name__ = f"neutralize_{variable_name}"
    return NeutralizeVariableReform


def _materialize_target_frame(
    base_frame: Frame,
) -> tuple[Frame, TargetRegistry, dict[str, object]]:
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    dataset = _dataset_from_frame(base_frame)
    simulation = Microsimulation(dataset=dataset)
    system = CountryTaxBenefitSystem()
    household = base_frame.table("household")
    tax_unit_positions = _tax_unit_to_household_positions(base_frame)
    n_households = base_frame.n("household")

    materialized = {
        entity: base_frame.table(entity).copy() for entity in base_frame.entities
    }
    hh = materialized["household"]

    income_tax_tax_unit = _calculate_array(simulation, "income_tax")
    taxable_income_tax_unit = _calculate_array(simulation, "taxable_income")
    agi_tax_unit = _calculate_array(simulation, "adjusted_gross_income")
    filing_status = _filing_status_names(_calculate_array(simulation, "filing_status"))

    hh["income_tax"] = _collapse_tax_unit(
        income_tax_tax_unit, tax_unit_positions, n_households
    )
    hh["state_income_tax"] = _household_values(
        frame=base_frame,
        simulation=simulation,
        system=system,
        variable="state_income_tax",
        tax_unit_positions=tax_unit_positions,
    )

    for spec in US_FISCAL_TARGET_SPECS:
        if spec.family == "state_income_tax":
            state_fips = int(spec.metadata["state_fips"])
            hh[spec.measure] = hh["state_income_tax"].where(
                household["state_fips"].to_numpy() == state_fips,
                0.0,
            )

    variable_cache: dict[str, np.ndarray] = {}
    for source_name, pe_name in SOI_VARIABLE_MAP.items():
        if pe_name == "rent_and_royalty_net_income":
            rental_income = _calculate_array(simulation, "rental_income")
            farm_rent_income = _calculate_array(simulation, "farm_rent_income")
            variable_cache[source_name] = _person_variable_to_tax_unit(
                frame=base_frame,
                values=rental_income + farm_rent_income,
            )
            continue
        if pe_name not in system.variables:
            continue
        entity = _variable_entity(system, pe_name)
        raw = _calculate_array(simulation, pe_name)
        if entity == "tax_unit":
            variable_cache[source_name] = raw.astype(np.float64)
        elif entity == "person":
            variable_cache[source_name] = _person_variable_to_tax_unit(
                frame=base_frame,
                values=raw,
            )
        else:
            raise ValueError(
                f"SOI variable {pe_name!r} has unsupported entity {entity!r}."
            )

    for spec in US_FISCAL_TARGET_SPECS:
        if spec.family != "irs_soi":
            continue
        source_name = spec.metadata["variable"]
        lower = _as_bound(spec.metadata["agi_lower_bound"])
        upper = _as_bound(spec.metadata["agi_upper_bound"])
        mask = (agi_tax_unit >= lower) & (agi_tax_unit < upper)
        if spec.metadata.get("taxable_only") == "true":
            mask &= (income_tax_tax_unit > 0) | (taxable_income_tax_unit > 0)
        status = FILING_STATUS_MAP[spec.metadata["filing_status"]]
        if isinstance(status, str):
            mask &= filing_status == status
        elif isinstance(status, set):
            mask &= np.isin(filing_status, sorted(status))
        is_count = spec.metadata.get("count") == "true"
        if source_name == "count":
            values = mask.astype(np.float64)
        else:
            if source_name not in variable_cache:
                continue
            values = (
                _soi_component_row(
                    variable_cache[source_name],
                    source_name,
                    count=is_count,
                )
                * mask
            )
        hh[spec.measure] = _collapse_tax_unit(values, tax_unit_positions, n_households)

    base_income_tax_household = hh["income_tax"].to_numpy(dtype=np.float64)
    for reform_spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        reform = _make_zero_variable_reform(system, reform_spec.neutralized_variable)
        reformed_dataset = _dataset_from_frame(
            base_frame,
            zero_variables=(reform_spec.neutralized_variable,),
            system=system,
        )
        reformed = Microsimulation(dataset=reformed_dataset, reform=reform)
        reform_income_tax = np.asarray(
            reformed.calculate("income_tax", period=PERIOD, map_to="household"),
            dtype=np.float64,
        )
        hh[reform_spec.measure] = reform_income_tax - base_income_tax_household

    compileable_specs = [
        spec
        for spec in US_FISCAL_TARGET_SPECS
        if spec.measure is None or spec.measure in hh.columns
    ]
    registry = TargetRegistry(compileable_specs, country="us")
    dropped = sorted(
        spec.name for spec in US_FISCAL_TARGET_SPECS if spec not in compileable_specs
    )
    target_frame = Frame(
        materialized,
        US_SCHEMA,
        {"household": base_frame.weights_for("household")},
        base_frame.strata,
    )
    return (
        target_frame,
        registry,
        {
            "declared_targets": len(US_FISCAL_TARGET_SPECS),
            "compiled_candidate_targets": len(compileable_specs),
            "dropped_target_names": dropped,
        },
    )


def _strip_calibration_columns(
    base_frame: Frame, calibrated_weights: np.ndarray
) -> Frame:
    base_frame = _drop_formula_owned_columns(base_frame)
    return base_frame.with_weights(
        "household",
        Weights(calibrated_weights, WeightKind.CALIBRATED),
        mass=MassChange(
            factor=calibrated_weights.sum() / base_frame.weights_for("household").total,
            reason="US fiscal target refresh calibration",
        ),
    )


def _write_npz(path: Path, *, result, registry: TargetRegistry) -> None:
    np.savez_compressed(
        path,
        household_weight=result.weights,
        initial_household_weight=result.initial_weights,
        target_names=np.asarray([d.name for d in result.diagnostics], dtype=object),
        target_values=np.asarray(
            [d.target for d in result.diagnostics], dtype=np.float64
        ),
        initial_estimates=np.asarray(
            [d.initial_estimate for d in result.diagnostics], dtype=np.float64
        ),
        final_estimates=np.asarray(
            [d.final_estimate for d in result.diagnostics], dtype=np.float64
        ),
        relative_errors=np.asarray(
            [d.relative_error for d in result.diagnostics], dtype=np.float64
        ),
        registry_version=np.asarray(registry.version),
    )


def _release_gate_failures(result, compilation: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    dropped = compilation.get("dropped_target_names") or []
    if dropped:
        failures.append(f"{len(dropped)} fiscal targets were not materialized.")
    if result.skipped:
        failures.append(
            f"{len(result.skipped)} fiscal targets were skipped by calibration."
        )
    if not result.diagnostics:
        failures.append("No fiscal targets were compiled.")
    zero_support = [
        diagnostic.name
        for diagnostic in result.diagnostics
        if float(getattr(diagnostic, "target", 0.0)) > 0.0
        and abs(float(getattr(diagnostic, "initial_estimate", 0.0))) <= 1e-9
        and abs(float(getattr(diagnostic, "final_estimate", 0.0))) <= 1e-9
    ]
    if zero_support:
        examples = ", ".join(zero_support[:5])
        suffix = "" if len(zero_support) <= 5 else ", ..."
        failures.append(
            f"{len(zero_support)} positive fiscal targets have zero "
            f"materialized support (examples: {examples}{suffix})."
        )
    if not math.isfinite(result.initial_loss) or not math.isfinite(result.final_loss):
        failures.append("Calibration loss is non-finite.")
    elif result.final_loss > result.initial_loss:
        failures.append(
            "Calibration final loss is worse than the initial loss "
            f"({result.final_loss} > {result.initial_loss})."
        )
    return failures


def _assert_release_gates(result, compilation: Mapping[str, object]) -> None:
    failures = _release_gate_failures(result, compilation)
    if failures:
        raise RuntimeError("Release gates failed: " + "; ".join(failures))


def _target_final_estimate(result, target_name: str) -> float:
    for diagnostic in result.diagnostics:
        if diagnostic.name == f"{target_name}@{PERIOD}":
            return float(diagnostic.final_estimate)
    raise KeyError(target_name)


def _state_income_tax_target_sum(result) -> float:
    return float(
        sum(
            diagnostic.final_estimate
            for diagnostic in result.diagnostics
            if diagnostic.name.startswith("state/")
            and diagnostic.name.endswith(f"/state_income_tax@{PERIOD}")
        )
    )


def _assert_export_matches_calibration(dataset_path: Path, result) -> None:
    target_frame, registry, compilation = _materialize_target_frame(
        _load_frame(dataset_path)
    )
    dropped = compilation.get("dropped_target_names") or []
    if dropped:
        raise RuntimeError(
            "Post-export sanity failed: "
            f"{len(dropped)} fiscal targets were not materialized after export."
        )
    diagnostics_by_name = {
        diagnostic.name: diagnostic for diagnostic in result.diagnostics
    }
    failures = []
    for target in registry.to_target_set():
        diagnostic = diagnostics_by_name.get(target.row_name)
        if diagnostic is None:
            failures.append(f"{target.row_name} missing from calibration diagnostics")
            continue
        expected = float(diagnostic.final_estimate)
        observed = target.achieved_value(
            target_frame, target_frame.weights_for(target.entity).values
        )
        tolerance = max(abs(expected) * 1e-6, 1_000_000.0)
        if abs(observed - expected) > tolerance:
            failures.append(
                f"{target.row_name} exported value {observed:.6g} differs from "
                f"calibration final estimate {expected:.6g} by more than "
                f"{tolerance:.6g}"
            )
    if failures:
        raise RuntimeError("Post-export sanity failed: " + "; ".join(failures))


def _artifact_entry(path: str, sha: str, *, kind: str, revision: str) -> dict[str, str]:
    return {
        "kind": kind,
        "path": path,
        "repo_id": REPO_ID,
        "revision": revision,
        "sha256": sha,
    }


def _write_demographics(
    *,
    release_dir: Path,
    dataset_path: Path,
    release_id: str,
) -> None:
    """Emit demographics.json: the dataset's weighted population by age band.

    The fiscal-refresh release does not calibrate the age distribution, so this
    is published as an emergent diagnostic (populace vs the Census age structure).
    """
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    sim = Microsimulation(dataset=USSingleYearDataset(file_path=str(dataset_path)))
    ages, weights = population_by_age_from_sim(sim, PERIOD)
    payload = demographics_payload(ages, weights, period=PERIOD, release_id=release_id)
    write_demographics(payload, release_dir / "demographics.json")


def _build_manifests(
    *,
    release_id: str,
    release_dir: Path,
    artifact_root: Path,
    result,
    registry: TargetRegistry,
    dropped: Mapping[str, object],
) -> None:
    dataset_path = artifact_root / DATASET_FILENAME
    calibration_path = artifact_root / CALIBRATION_FILENAME
    diagnostics_path = release_dir / "calibration_diagnostics.json"
    coverage_path = release_dir / "us_source_coverage.json"

    dataset_sha = _sha256(dataset_path)
    calibration_sha = _sha256(calibration_path)
    diagnostics_sha = _sha256(diagnostics_path)
    coverage_sha = _sha256(coverage_path)
    diag = diagnostics_payload(result, target_registry=registry)
    gate_failures = _release_gate_failures(result, dropped)

    commit = _git_output("rev-parse", "HEAD")
    built_at = datetime.now(UTC).isoformat()
    runtime = _runtime_versions()
    manifest = {
        "build_id": release_id,
        "build_sha": commit[:7],
        "created_at": built_at,
        "code": {
            "repository": "PolicyEngine/populace",
            "git_commit": commit,
            "git_dirty": False,
        },
        "runtime": runtime,
        "dataset": {
            "filename": DATASET_FILENAME,
            "sha256": dataset_sha,
        },
        "calibration": {
            "filename": CALIBRATION_FILENAME,
            "sha256": calibration_sha,
            "target_surface": {
                "sha256": diag["target_surface"]["sha256"],
                "n_targets": diag["target_surface"]["n_targets"],
            },
            "target_registry": {
                "version": registry.version,
                "n_specs": len(registry),
            },
        },
        "gates": {
            "calibration": {
                "passed": not gate_failures,
                "failures": gate_failures,
                "initial_loss": diag["initial_loss"],
                "final_loss": diag["final_loss"],
                "fraction_within_10pct": diag["fraction_within_10pct"],
            },
            "target_compilation": dropped,
        },
    }
    (release_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=1, allow_nan=False)
    )

    release_manifest = {
        "schema_version": 1,
        "data_package": {
            "name": "populace-data",
            "version": runtime["populace-data"],
        },
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release_id,
            "built_at": built_at,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": runtime["policyengine-core"],
            },
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": runtime["policyengine-us"],
            },
        },
        "artifacts": {
            "populace_us_2024": _artifact_entry(
                DATASET_FILENAME,
                dataset_sha,
                kind="microdata",
                revision=release_id,
            ),
            "populace_us_2024_calibration": _artifact_entry(
                CALIBRATION_FILENAME,
                calibration_sha,
                kind="calibration",
                revision=release_id,
            ),
            "calibration_diagnostics": _artifact_entry(
                "calibration_diagnostics.json",
                diagnostics_sha,
                kind="diagnostics",
                revision=release_id,
            ),
            "us_source_coverage": _artifact_entry(
                "us_source_coverage.json",
                coverage_sha,
                kind="diagnostics",
                revision=release_id,
            ),
            **(
                {
                    "demographics": _artifact_entry(
                        "demographics.json",
                        _sha256(release_dir / "demographics.json"),
                        kind="diagnostics",
                        revision=release_id,
                    )
                }
                if (release_dir / "demographics.json").exists()
                else {}
            ),
        },
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=1, allow_nan=False)
    )


def _reviewed_exclusions(active_aliases: Iterable[str]) -> dict[str, str]:
    active = set(active_aliases)
    hard = set(hard_target_package_aliases())
    unknown_active = sorted(active - hard)
    if unknown_active:
        raise RuntimeError(
            "Fiscal-refresh active aliases are not hard-target aliases: "
            + ", ".join(unknown_active)
        )
    excluded = hard - active
    reviewed = set(REVIEWED_EXCLUDED_ALIASES)
    if excluded != reviewed:
        missing = sorted(excluded - reviewed)
        extra = sorted(reviewed - excluded)
        raise RuntimeError(
            "Reviewed hard-target exclusion list is stale "
            f"(missing={missing}, extra={extra})."
        )
    return {
        alias: (
            "Reviewed fiscal-refresh exclusion: this release recalibrates the "
            "Issue #40 fiscal target surface only. This source family is not "
            "certified by this release and should be validated in a dedicated "
            "non-fiscal refresh before being treated as active calibration "
            "coverage."
        )
        for alias in REVIEWED_EXCLUDED_ALIASES
    }


def _fiscal_target_source_provenance() -> dict[str, dict[str, object]]:
    provenance: dict[str, dict[str, object]] = {}
    for spec in US_FISCAL_TARGET_SPECS:
        entry = provenance.setdefault(
            spec.family,
            {
                "label": FISCAL_TARGET_SOURCE_KEYS.get(spec.family, spec.family),
                "target_count": 0,
                "sources": [],
                "reference_urls": [],
            },
        )
        entry["target_count"] = int(entry["target_count"]) + 1
        sources = entry["sources"]
        if isinstance(sources, list) and spec.source not in sources:
            sources.append(spec.source)
        url = spec.metadata.get("reference_url")
        urls = entry["reference_urls"]
        if isinstance(urls, list) and url and url not in urls:
            urls.append(url)
    return {
        family: {
            "label": payload["label"],
            "target_count": payload["target_count"],
            "sources": sorted(payload["sources"]),
            "reference_urls": sorted(payload["reference_urls"]),
        }
        for family, payload in sorted(provenance.items())
    }


def _assert_us_release_id(release_id: str) -> None:
    if not release_id.startswith("populace-us-"):
        raise ValueError(
            "US fiscal refresh release ids must start with 'populace-us-' so "
            "the US release contract requires source coverage diagnostics."
        )


def main() -> None:
    args = _parse_args()
    if _git_dirty():
        raise SystemExit("Refusing to build a release from a dirty git worktree.")

    base_h5 = args.base_h5 or _download_base_h5()
    digest = _sha256(base_h5)[:7]
    build_timestamp = datetime.now(UTC)
    commit = _git_output("rev-parse", "--short=12", "HEAD")
    release_id = (
        args.release_id
        or f"populace-us-2024-{digest}-{commit}-{build_timestamp:%Y%m%dT%H%M%SZ}"
    )
    _assert_us_release_id(release_id)
    release_root = args.out.resolve()
    artifact_root = release_root / "artifacts"
    release_dir = release_root / "releases" / release_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    base_frame = _load_frame(base_h5)
    target_frame, registry, compilation = _materialize_target_frame(base_frame)
    result = calibrate(
        target_frame,
        registry.to_target_set(),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_weight_ratio=args.max_weight_ratio,
        seed=args.seed,
        mass="conserve",
    )
    _assert_release_gates(result, compilation)

    export_frame = _strip_calibration_columns(base_frame, result.weights)
    dataset_path = artifact_root / DATASET_FILENAME
    PolicyEngineUSEngine().write_dataset(export_frame, dataset_path, period=PERIOD)
    _assert_export_matches_calibration(dataset_path, result)

    calibration_path = artifact_root / CALIBRATION_FILENAME
    _write_npz(calibration_path, result=result, registry=registry)
    write_calibration_diagnostics(
        result,
        release_dir / "calibration_diagnostics.json",
        target_registry=registry,
        build={
            "base_dataset_sha256": _sha256(base_h5),
            "target_compilation": compilation,
        },
    )

    if not args.skip_demographics:
        _write_demographics(
            release_dir=release_dir,
            dataset_path=dataset_path,
            release_id=release_id,
        )

    active_aliases = DIRECT_ACTIVE_ALIASES
    coverage = us_source_coverage_diagnostics(
        active_target_aliases=active_aliases,
        reviewed_exclusions=_reviewed_exclusions(active_aliases),
    )
    coverage["fiscal_target_sources"] = _fiscal_target_source_provenance()
    write_us_source_coverage_diagnostics(
        coverage, release_dir / "us_source_coverage.json"
    )
    _build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=registry,
        dropped=compilation,
    )

    # Keep a copy of the exact base artifact beside diagnostics for local audit.
    shutil.copy2(base_h5, release_root / f"base_{base_h5.name}")
    print(
        json.dumps(
            {
                "release_id": release_id,
                "release_dir": str(release_dir),
                "artifact_root": str(artifact_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
