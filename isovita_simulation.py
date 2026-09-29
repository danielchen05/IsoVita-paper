"""Reusable simulation utilities for IsoVita.

The simulator intentionally generates transcript behavior *before* IsoVita's
feature-condensation step. Each transcript receives its own latent log-abundance
trajectory as a mixture of temporal components, after which the transcript
vector is converted to within-gene composition by softmax.

This design avoids planting an exact A/B balance as the data-generating model,
and can be reused for:

- feature-condensation intuition / stress-test figures;
- null calibration and empirical FDR simulations;
- power studies under smooth, discrete, localized, or mixed trajectories;
- sensitivity analyses over sample size, noise, isoform count, age design, and
  temporal rank.

The module does not depend on IsoVita itself. It only produces objects that are
compatible with ``isovita.build_one_feature_per_gene`` and related functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# SCENARIO SPECIFICATION
# =============================================================================


@dataclass(frozen=True)
class TemporalComponent:
    """One temporal basis function with transcript-specific loadings.

    Parameters
    ----------
    kind
        Temporal shape. Supported values are ``linear``, ``sigmoid``, ``step``,
        ``u_shape``, ``localized``, and ``sinusoid``.
    loadings
        One coefficient per isoform. These need not form a balance and are not
        normalized; they describe how strongly each transcript responds to this
        temporal component on the latent log-abundance scale.
    amplitude
        Global multiplier applied to this component.
    params
        Shape-specific parameters. Examples: ``center`` and ``scale`` for a
        sigmoid; ``change_age`` for a step; ``center`` and ``width`` for a
        localized Gaussian bump; ``cycles`` and ``phase`` for a sinusoid.
    """

    kind: str
    loadings: tuple[float, ...]
    amplitude: float = 1.0
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IsoformScenario:
    """Complete latent-trajectory specification for one simulated gene."""

    gene_id: str
    label: str
    baselines: tuple[float, ...]
    components: tuple[TemporalComponent, ...] = ()
    noise_sd: float = 0.20
    noise_corr: float = 0.10
    gene_total_log_mean: float = float(np.log(50.0))
    gene_total_log_sd: float = 0.15
    notes: str = ""

    @property
    def k(self) -> int:
        return len(self.baselines)

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("IsoVita simulation scenarios require at least 2 isoforms.")
        if self.noise_sd < 0:
            raise ValueError("noise_sd must be non-negative.")
        if self.gene_total_log_sd < 0:
            raise ValueError("gene_total_log_sd must be non-negative.")
        if self.noise_corr >= 1.0:
            raise ValueError("noise_corr must be < 1.")
        lower = -1.0 / max(self.k - 1, 1)
        if self.noise_corr <= lower:
            raise ValueError(
                f"For K={self.k}, noise_corr must be greater than {lower:.3f}."
            )
        for component in self.components:
            if len(component.loadings) != self.k:
                raise ValueError(
                    f"Component {component.kind!r} for {self.gene_id} has "
                    f"{len(component.loadings)} loadings but K={self.k}."
                )


@dataclass
class SimulatedIsoVitaData:
    """Tables emitted by :func:`simulate_isovita_dataset`."""

    metadata: pd.DataFrame
    expression: pd.DataFrame
    global_metrics: pd.DataFrame
    fractions: pd.DataFrame
    component_loadings: pd.DataFrame
    scenario_table: pd.DataFrame

    def pipeline_metadata(self) -> pd.DataFrame:
        """Return canonical metadata accepted by ``run_isovita_pipeline``."""
        out = self.metadata.reset_index()
        first = out.columns[0]
        if first != "sample_id":
            out = out.rename(columns={first: "sample_id"})
        return out

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Dictionary view convenient for notebooks."""
        return {
            "metadata": self.metadata,
            "expression": self.expression,
            "global_metrics": self.global_metrics,
            "fractions": self.fractions,
            "component_loadings": self.component_loadings,
            "scenario_table": self.scenario_table,
        }


# =============================================================================
# TEMPORAL BASIS FUNCTIONS
# =============================================================================


def _center_and_scale(values: np.ndarray) -> np.ndarray:
    """Center a temporal basis and scale it to max absolute value 1."""
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    max_abs = float(np.max(np.abs(x)))
    if max_abs <= np.finfo(float).eps:
        return np.zeros_like(x)
    return x / max_abs


def temporal_basis(
    age: np.ndarray,
    kind: str,
    params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Evaluate a standardized temporal basis on the supplied age vector."""
    age = np.asarray(age, dtype=float)
    params = dict(params or {})
    if age.ndim != 1 or len(age) == 0:
        raise ValueError("age must be a non-empty one-dimensional array.")

    age_min = float(np.min(age))
    age_max = float(np.max(age))
    age_mid = 0.5 * (age_min + age_max)
    age_span = max(age_max - age_min, np.finfo(float).eps)

    if kind == "linear":
        raw = (age - age_mid) / age_span

    elif kind == "sigmoid":
        center = float(params.get("center", age_mid))
        scale = float(params.get("scale", age_span / 5.0))
        if scale <= 0:
            raise ValueError("sigmoid scale must be positive.")
        raw = np.tanh((age - center) / scale)

    elif kind == "step":
        change_age = float(params.get("change_age", age_mid))
        raw = np.where(age < change_age, -1.0, 1.0)

    elif kind == "u_shape":
        center = float(params.get("center", age_mid))
        scale = float(params.get("scale", age_span / 2.0))
        if scale <= 0:
            raise ValueError("u_shape scale must be positive.")
        raw = ((age - center) / scale) ** 2

    elif kind == "localized":
        center = float(params.get("center", age_mid))
        width = float(params.get("width", age_span / 8.0))
        if width <= 0:
            raise ValueError("localized width must be positive.")
        raw = np.exp(-0.5 * ((age - center) / width) ** 2)

    elif kind == "sinusoid":
        cycles = float(params.get("cycles", 1.0))
        phase = float(params.get("phase", 0.0))
        raw = np.sin(2.0 * np.pi * cycles * (age - age_min) / age_span + phase)

    else:
        raise ValueError(
            f"Unknown temporal basis {kind!r}. Supported: linear, sigmoid, step, "
            "u_shape, localized, sinusoid."
        )

    return _center_and_scale(raw)


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _equicorrelated_noise(
    rng: np.random.Generator,
    n_samples: int,
    k: int,
    sd: float,
    corr: float,
) -> np.ndarray:
    if sd == 0:
        return np.zeros((n_samples, k), dtype=float)

    covariance = (sd**2) * (
        (1.0 - corr) * np.eye(k) + corr * np.ones((k, k))
    )
    return rng.multivariate_normal(
        mean=np.zeros(k, dtype=float),
        cov=covariance,
        size=n_samples,
    )


# =============================================================================
# AGE DESIGN
# =============================================================================


def simulate_ages(
    n_samples: int,
    *,
    age_min: float = 0.0,
    age_max: float = 85.0,
    design: str = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a reusable age design for simulation studies.

    ``uniform`` samples continuously over the age range. ``balanced`` places
    approximately equal numbers of samples across the range with small jitter.
    ``middle_heavy`` uses a beta distribution to create fewer boundary samples.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if age_max <= age_min:
        raise ValueError("age_max must exceed age_min.")

    rng = rng or np.random.default_rng()

    if design == "uniform":
        unit = rng.uniform(0.0, 1.0, size=n_samples)
    elif design == "balanced":
        unit = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
        unit += rng.normal(0.0, 0.15 / n_samples, size=n_samples)
        unit = np.clip(unit, 0.0, 1.0)
    elif design == "middle_heavy":
        unit = rng.beta(2.0, 2.0, size=n_samples)
    else:
        raise ValueError("design must be one of: uniform, balanced, middle_heavy.")

    ages = age_min + unit * (age_max - age_min)
    return np.sort(ages.astype(float))


# =============================================================================
# DATA GENERATOR
# =============================================================================


def simulate_isovita_dataset(
    scenarios: Sequence[IsoformScenario],
    *,
    n_samples: int = 180,
    ages: Optional[Sequence[float]] = None,
    age_min: float = 0.0,
    age_max: float = 85.0,
    age_design: str = "uniform",
    seed: int = 20260927,
    sample_prefix: str = "S",
) -> SimulatedIsoVitaData:
    """Simulate transcript-level TPM-like data for one or more genes.

    The noise-free latent log abundance for transcript ``j`` is

    ``baseline_j + sum_m amplitude_m * loading_jm * h_m(age)``.

    Correlated Gaussian noise is added on the same latent scale. Both the
    noise-free and observed latent vectors are then converted to compositional
    fractions with softmax. Observed fractions are multiplied by a positive
    log-normal gene total to form TPM-like expression values.
    """
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if len({s.gene_id for s in scenarios}) != len(scenarios):
        raise ValueError("Scenario gene_id values must be unique.")

    rng = np.random.default_rng(seed)

    if ages is None:
        age = simulate_ages(
            n_samples,
            age_min=age_min,
            age_max=age_max,
            design=age_design,
            rng=rng,
        )
    else:
        age = np.asarray(ages, dtype=float)
        if age.ndim != 1 or len(age) < 2:
            raise ValueError("ages must be a one-dimensional vector with >=2 values.")
        if np.any(~np.isfinite(age)):
            raise ValueError("ages must contain only finite values.")
        age = np.sort(age)
        n_samples = len(age)

    width = max(3, len(str(n_samples - 1)))
    samples = np.asarray(
        [f"{sample_prefix}{i:0{width}d}" for i in range(n_samples)],
        dtype=str,
    )
    metadata = pd.DataFrame({"age": age}, index=samples)
    metadata.index.name = "sample"

    expression_rows: list[dict[str, object]] = []
    global_metric_rows: list[dict[str, object]] = []
    fraction_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []

    for scenario_index, scenario in enumerate(scenarios):
        k = scenario.k
        tx_ids = [f"{scenario.gene_id}_TX{i + 1}" for i in range(k)]

        latent_true = np.repeat(
            np.asarray(scenario.baselines, dtype=float)[None, :],
            n_samples,
            axis=0,
        )

        component_values: dict[str, np.ndarray] = {}
        for component_index, component in enumerate(scenario.components, start=1):
            h = temporal_basis(age, component.kind, component.params)
            loadings = np.asarray(component.loadings, dtype=float)
            latent_true += (
                float(component.amplitude)
                * h[:, None]
                * loadings[None, :]
            )
            component_name = f"component_{component_index}"
            component_values[component_name] = h

            for tx, loading in zip(tx_ids, loadings):
                loading_rows.append({
                    "gene_id": scenario.gene_id,
                    "transcript_id": tx,
                    "component": component_name,
                    "kind": component.kind,
                    "amplitude": float(component.amplitude),
                    "loading": float(loading),
                    "scaled_loading": float(component.amplitude * loading),
                })

        noise = _equicorrelated_noise(
            rng,
            n_samples=n_samples,
            k=k,
            sd=float(scenario.noise_sd),
            corr=float(scenario.noise_corr),
        )
        latent_observed = latent_true + noise

        true_fraction = _softmax_rows(latent_true)
        observed_fraction = _softmax_rows(latent_observed)

        gene_total = rng.lognormal(
            mean=float(scenario.gene_total_log_mean),
            sigma=float(scenario.gene_total_log_sd),
            size=n_samples,
        )
        tpm = observed_fraction * gene_total[:, None]

        for j, tx in enumerate(tx_ids):
            expression_rows.append({
                "gene_id": scenario.gene_id,
                "transcript_id": tx,
                **dict(zip(samples, tpm[:, j])),
            })
            global_metric_rows.append({
                "gene_id": scenario.gene_id,
                "transcript_id": tx,
                "effective": True,
            })

            for i, sample in enumerate(samples):
                row: dict[str, object] = {
                    "gene_id": scenario.gene_id,
                    "scenario_label": scenario.label,
                    "sample": sample,
                    "age": float(age[i]),
                    "transcript_id": tx,
                    "true_latent_log_abundance": float(latent_true[i, j]),
                    "observed_latent_log_abundance": float(latent_observed[i, j]),
                    "true_fraction": float(true_fraction[i, j]),
                    "fraction": float(observed_fraction[i, j]),
                    "gene_total": float(gene_total[i]),
                    "tpm": float(tpm[i, j]),
                }
                for component_name, values in component_values.items():
                    row[component_name] = float(values[i])
                fraction_rows.append(row)

        scenario_rows.append({
            "scenario_index": int(scenario_index),
            "gene_id": scenario.gene_id,
            "label": scenario.label,
            "k": int(k),
            "n_components": int(len(scenario.components)),
            "noise_sd": float(scenario.noise_sd),
            "noise_corr": float(scenario.noise_corr),
            "notes": scenario.notes,
        })

    return SimulatedIsoVitaData(
        metadata=metadata,
        expression=pd.DataFrame(expression_rows),
        global_metrics=pd.DataFrame(global_metric_rows),
        fractions=pd.DataFrame(fraction_rows),
        component_loadings=pd.DataFrame(loading_rows),
        scenario_table=pd.DataFrame(scenario_rows),
    )


# =============================================================================
# REUSABLE SCENARIO FACTORIES
# =============================================================================


def make_feature_condensation_scenarios() -> tuple[IsoformScenario, ...]:
    """Four manuscript-oriented cases for the feature-condensation supplement.

    The cases progress from a two-isoform reference case to coherent multi-
    isoform movement, sparse condensation with passenger trajectories, and a
    genuinely multi-mode stress test in which one feature cannot represent all
    temporal variation.
    """
    common_noise = 0.20
    common_corr = 0.10

    return (
        IsoformScenario(
            gene_id="COND_A",
            label="Two-isoform nonlinear contrast",
            baselines=(0.20, -0.15),
            components=(
                TemporalComponent(
                    kind="sigmoid",
                    loadings=(1.00, -0.90),
                    amplitude=1.15,
                    params={"center": 38.0, "scale": 15.0},
                ),
                TemporalComponent(
                    kind="localized",
                    loadings=(0.18, -0.12),
                    amplitude=0.35,
                    params={"center": 18.0, "width": 7.0},
                ),
            ),
            noise_sd=common_noise,
            noise_corr=common_corr,
            notes=(
                "Reference case: with two isoforms, the compositional trajectory "
                "is intrinsically one-dimensional even though its age shape is nonlinear."
            ),
        ),
        IsoformScenario(
            gene_id="COND_B",
            label="Coherent multi-isoform groups",
            baselines=(0.25, 0.05, -0.05, -0.20),
            components=(
                TemporalComponent(
                    kind="sigmoid",
                    loadings=(1.00, 0.86, -0.96, -0.82),
                    amplitude=1.05,
                    params={"center": 42.0, "scale": 17.0},
                ),
                TemporalComponent(
                    kind="u_shape",
                    loadings=(0.10, -0.08, 0.06, -0.08),
                    amplitude=0.28,
                    params={"center": 43.0, "scale": 42.5},
                ),
            ),
            noise_sd=common_noise,
            noise_corr=common_corr,
            notes=(
                "Four isoforms share a dominant two-group contrast but have mildly "
                "different shapes and amplitudes."
            ),
        ),
        IsoformScenario(
            gene_id="COND_C",
            label="Dominant contrast with passenger isoforms",
            baselines=(0.20, 0.02, -0.08, -0.18, -0.28),
            components=(
                TemporalComponent(
                    kind="sigmoid",
                    loadings=(1.00, 0.92, -1.20, -0.12, -0.10),
                    amplitude=1.05,
                    params={"center": 36.0, "scale": 14.0},
                ),
                TemporalComponent(
                    kind="u_shape",
                    loadings=(0.05, -0.04, 0.03, -0.02, 0.62),
                    amplitude=0.45,
                    params={"center": 48.0, "scale": 38.0},
                ),
            ),
            noise_sd=common_noise,
            noise_corr=common_corr,
            notes=(
                "The main temporal axis is TX1/TX2 versus TX3. TX4 is weak and TX5 "
                "has a real but secondary U-shaped trajectory, so 'unused' need not "
                "mean biologically flat."
            ),
        ),
        IsoformScenario(
            gene_id="COND_D",
            label="Competing temporal modes",
            baselines=(0.24, 0.10, 0.00, -0.08, -0.18, -0.28),
            components=(
                TemporalComponent(
                    kind="sigmoid",
                    loadings=(1.00, 0.92, -0.98, -0.88, 0.00, 0.00),
                    amplitude=0.95,
                    params={"center": 40.0, "scale": 16.0},
                ),
                TemporalComponent(
                    kind="localized",
                    loadings=(0.00, 0.00, 0.00, 0.00, 1.00, -1.00),
                    amplitude=1.10,
                    params={"center": 42.5, "width": 10.0},
                ),
            ),
            noise_sd=common_noise,
            noise_corr=common_corr,
            notes=(
                "Stress test: a dominant four-isoform smooth contrast coexists with "
                "a strong orthogonal localized TX5-versus-TX6 mode. One condensed feature "
                "should capture the dominant PC1 rather than all temporal structure."
            ),
        ),
    )


def make_null_scenarios(
    n_genes: int,
    *,
    k: int = 4,
    noise_sd: float = 0.20,
    noise_corr: float = 0.10,
    seed: int = 1,
    gene_prefix: str = "NULL",
) -> tuple[IsoformScenario, ...]:
    """Convenience factory for age-null genes used in calibration studies."""
    if n_genes < 1:
        raise ValueError("n_genes must be >= 1.")
    if k < 2:
        raise ValueError("k must be >= 2.")

    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n_genes):
        baselines = tuple(rng.normal(0.0, 0.20, size=k).tolist())
        scenarios.append(
            IsoformScenario(
                gene_id=f"{gene_prefix}_{i + 1:05d}",
                label="Age-null",
                baselines=baselines,
                components=(),
                noise_sd=noise_sd,
                noise_corr=noise_corr,
                notes="No age-dependent latent component.",
            )
        )
    return tuple(scenarios)


__all__ = [
    "TemporalComponent",
    "IsoformScenario",
    "SimulatedIsoVitaData",
    "temporal_basis",
    "simulate_ages",
    "simulate_isovita_dataset",
    "make_feature_condensation_scenarios",
    "make_null_scenarios",
]
