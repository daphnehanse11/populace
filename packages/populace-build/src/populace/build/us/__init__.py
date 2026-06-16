"""populace.build.us: the typed stage plan of the US dataset build.

This module is the declarative description of how the US population dataset
is assembled — the stage order, what each stage produces, and **which primary
survey each imputation draws from**, with citations. It replaces the
imperative driver's implicit structure with one reviewable object:
:func:`us_plan` returns the :class:`~populace.build.plan.StagePlan` whose
donor graph is the published sources diagram.

Every donor here is a primary source. Incumbent production datasets are not
build inputs; release comparisons against them live in the external benchmark
repo.

Implementations are injected: :func:`us_plan` requires one callable per
declared stage and refuses to assemble without all of them — there is no
stub, default, or fallback implementation (a plan you can run with a missing
stage would be the silent-fallback bug as a framework feature). Source-stage
content lives in the packaged JSON manifest loaded as
:data:`US_SOURCE_MANIFEST`; executable Python belongs only in shared Populace
runtimes that interpret those specs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files

from populace.build.plan import DonorSpec, Stage, StagePlan
from populace.build.source_manifest import (
    SourceManifest,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.us.demographics import (
    AGE_BANDS,
    DEMOGRAPHICS_SCHEMA_VERSION,
    AgeBand,
    compute_age_distribution,
    demographics_payload,
    write_demographics,
)
from populace.build.us.fiscal_targets import (
    US_FISCAL_LEDGER_PARITY_REGISTRY,
    US_FISCAL_LEDGER_PARITY_REPORT,
    US_FISCAL_MACRO_REALISM_BANDS,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_LEDGER_REFERENCES,
    US_FISCAL_TARGET_REGISTRY,
    US_FISCAL_TARGET_SPECS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_JCT_TAX_EXPENDITURE_TARGET_SPECS,
    US_SOI_FISCAL_TARGET_SPECS,
    US_STATE_INCOME_TAX_TARGET_SPECS,
    SimpleTaxExpenditureReform,
)
from populace.build.us.source_coverage import (
    ARCH_US_SOURCE_COVERAGE_CONTRACT_COMMIT,
    US_SOURCE_COVERAGE,
    hard_target_package_aliases,
    source_gap_family_ids,
    us_source_coverage_diagnostics,
    us_source_coverage_gate,
    validation_only_family_ids,
    write_us_source_coverage_diagnostics,
)
from populace.frame import Frame

__all__ = [
    "BuildConfig",
    "SimpleTaxExpenditureReform",
    "AgeBand",
    "AGE_BANDS",
    "DEMOGRAPHICS_SCHEMA_VERSION",
    "compute_age_distribution",
    "demographics_payload",
    "write_demographics",
    "ARCH_US_SOURCE_COVERAGE_CONTRACT_COMMIT",
    "US_DONORS",
    "US_FISCAL_MACRO_REALISM_BANDS",
    "US_FISCAL_LEDGER_PARITY_REGISTRY",
    "US_FISCAL_LEDGER_PARITY_REPORT",
    "US_FISCAL_TARGET_REGISTRY",
    "US_FISCAL_TARGET_SPECS",
    "US_FISCAL_TARGET_COVERAGE_REQUIREMENTS",
    "US_FISCAL_TARGET_LEDGER_REFERENCES",
    "US_JCT_TAX_EXPENDITURE_REFORMS",
    "US_JCT_TAX_EXPENDITURE_TARGET_SPECS",
    "US_NONNEGATIVE_SOURCE_OUTPUTS",
    "US_SOURCE_COVERAGE",
    "US_SOI_FISCAL_TARGET_SPECS",
    "US_SOURCE_MANIFEST",
    "US_SOURCE_STAGE_SPECS",
    "US_STAGE_NAMES",
    "US_STATE_INCOME_TAX_TARGET_SPECS",
    "hard_target_package_aliases",
    "source_gap_family_ids",
    "us_plan",
    "us_source_coverage_diagnostics",
    "us_source_coverage_gate",
    "write_us_source_coverage_diagnostics",
    "validation_only_family_ids",
]


@dataclass(frozen=True)
class BuildConfig:
    """The declared knobs of a US build — everything a manifest must record.

    Attributes:
        year: The dataset's time period.
        seed: The build-wide imputation seed.
        max_weight_ratio: The hard calibration bound (part of the dataset's
            provenance; recorded by the calibration's options too).
        calibration_epochs: Solver epochs.
        calibration_learning_rate: Solver learning rate.
        mass: Calibration mass policy (``"free"`` or ``"conserve"``).
        registry_path: Path to the versioned target-registry artifact the
            calibration compiles (see
            :mod:`populace.calibrate.registry`).
        extra: Free-form recorded settings (donor file paths, vintages).
    """

    year: int
    seed: int = 0
    max_weight_ratio: float = 50.0
    calibration_epochs: int = 3000
    calibration_learning_rate: float = 0.15
    mass: str = "free"
    registry_path: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.year < 1990:
            raise ValueError(f"year must be a survey year, got {self.year!r}.")
        if not (self.max_weight_ratio > 0):
            raise ValueError(
                f"max_weight_ratio must be positive, got {self.max_weight_ratio!r}."
            )
        if self.mass not in ("free", "conserve"):
            raise ValueError(f"mass must be 'free' or 'conserve', got {self.mass!r}.")

    def to_manifest(self) -> dict[str, object]:
        """A JSON-ready record for the release manifest."""
        return {
            "year": self.year,
            "seed": self.seed,
            "max_weight_ratio": self.max_weight_ratio,
            "calibration_epochs": self.calibration_epochs,
            "calibration_learning_rate": self.calibration_learning_rate,
            "mass": self.mass,
            "registry_path": self.registry_path,
            "extra": dict(self.extra),
        }


#: The US donor graph: every imputation stage's primary survey, with
#: citations. This is the single place the build's sources are declared —
#: the observatory's sources diagram and the dataset card derive from it.
US_DONORS: Mapping[str, DonorSpec] = {
    "scf_wealth": DonorSpec(
        survey="Fed SCF 2022",
        source="https://www.federalreserve.gov/econres/scfindex.htm",
        notes=(
            "Wealth components (accounts, stocks, bonds, debts) and net "
            "worth; household grain, head-carried person assets."
        ),
    ),
    "sipp_tips": DonorSpec(
        survey="Census SIPP",
        source="https://www.census.gov/programs-surveys/sipp.html",
        notes="Tip income for tipped occupations.",
    ),
    "org_wages": DonorSpec(
        survey="CPS ORG",
        source="https://www.bls.gov/cps/earnings.htm",
        notes=(
            "Hourly-wage labor-market inputs. Donor load failures abort the "
            "build — the silent zero-fallback this stage once had is "
            "structurally impossible under StagePlan."
        ),
    ),
    "meps_esi_premiums": DonorSpec(
        survey="MEPS-IC",
        source="https://meps.ahrq.gov/mepsweb/survey_comp/Insurance.jsp",
        notes="Employer-sponsored insurance premium parameters.",
    ),
    "prior_year_income": DonorSpec(
        survey="CPS ASEC (prior year)",
        source="https://www.census.gov/programs-surveys/cps.html",
        notes="PERIDNUM longitudinal join for prior-year earnings.",
    ),
    "puf_tax_detail": DonorSpec(
        survey="IRS PUF 2015 (uprated)",
        source="https://www.irs.gov/statistics/soi-tax-stats-individual-public-use-microdata-files",
        notes=(
            "Itemized-deduction detail, QBI components, partnership SE, "
            "mortgage-interest split; support clipped to the PUF's own "
            "realized ranges."
        ),
    ),
    "acs_rent": DonorSpec(
        survey="Census ACS 2022",
        source="https://www.census.gov/programs-surveys/acs",
        notes="Rent for renter households.",
    ),
    "vehicle_assets": DonorSpec(
        survey="Census SIPP",
        source="https://www.census.gov/programs-surveys/sipp.html",
        notes=(
            "Household vehicle count and value; vehicle value folds into "
            "household net worth."
        ),
    ),
}

#: Stage order of the US build. Derivation stages (no donor) interleave with
#: the donor imputations; the export/calibration stages close the plan.
US_STAGE_NAMES: tuple[str, ...] = (
    "asec_load",
    "unit_assignment",
    "derive_cps_carried",
    "puf_tax_detail",
    "scf_wealth",
    "sipp_tips",
    "org_wages",
    "meps_esi_premiums",
    "prior_year_income",
    "mortgage_conversion",
    "acs_rent",
    "vehicle_assets",
    "entity_placement",
    "export",
)


def _load_us_source_manifest() -> SourceManifest:
    return load_source_manifest(files(__package__).joinpath("source_stages.json"))


US_SOURCE_MANIFEST = _load_us_source_manifest()
US_SOURCE_STAGE_SPECS: tuple[SourceStageSpec, ...] = US_SOURCE_MANIFEST.stages
US_NONNEGATIVE_SOURCE_OUTPUTS: frozenset[str] = frozenset(
    output for stage in US_SOURCE_STAGE_SPECS for output in stage.nonnegative_outputs
)


def us_plan(
    implementations: Mapping[str, Callable[[Frame], Frame]],
) -> StagePlan:
    """Assemble the US build plan from stage implementations.

    Args:
        implementations: One ``transform(frame) -> Frame`` per stage in
            :data:`US_STAGE_NAMES`. ALL stages must be provided — a missing
            stage refuses to assemble (there are no default or stub
            implementations), and an unknown name is refused too (a typo
            must not silently drop a stage).

    Returns:
        The validated :class:`~populace.build.plan.StagePlan`, with each
        imputation stage carrying its :data:`US_DONORS` citation.

    Raises:
        ValueError: If any declared stage lacks an implementation, or an
            implementation is supplied for an undeclared stage.
    """
    missing = [name for name in US_STAGE_NAMES if name not in implementations]
    if missing:
        raise ValueError(
            f"us_plan needs an implementation for every declared stage; "
            f"missing {missing}. There are no stubs or fallbacks by design."
        )
    unknown = sorted(set(implementations) - set(US_STAGE_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown stage implementation(s) {unknown}; declared stages "
            f"are {list(US_STAGE_NAMES)}."
        )
    return StagePlan(
        Stage(
            name=name,
            transform=implementations[name],
            donor=US_DONORS.get(name),
        )
        for name in US_STAGE_NAMES
    )
