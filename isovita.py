"""
IsoVita
=======

Core temporal isoform-trajectory analysis operating on standardized TPM input.

Input contract
--------------
``run_isovita_pipeline()`` expects inputs that have already been prepared at the
dataset level (for example by ``prepare_isovita_inputs.py``):

* ``tpm``: transcript-level TPM table with columns ``gene_id``,
  ``transcript_id``, followed by one column per retained sample;
* ``metadata``: one row per retained sample with canonical columns
  ``sample_id`` and numeric ``age``. Sample order must match the TPM columns.

The core module deliberately does not perform cohort filtering, postnatal
filtering, sample-ID parsing, transcript-to-gene mapping, or metadata column
auto-detection. Those belong in the preparation layer.

Production algorithm
--------------------
1. Apply the configured TPM eligibility filter used for IsoPlex input.
2. Run IsoPlex and identify effective isoforms.
3. For every gene with >=2 effective isoforms, build one age-informed feature:
   effective-isoform composition -> CLR -> smooth trajectories -> PC1 ->
   simplest near-optimal A/B balance -> log(GM(A) / GM(B)).
4. Run the two production trajectory detectors:
   ``discrete`` (unknown pure step) and ``continuous`` (smooth spline).
5. Return branch-specific FDR results and a final gene set equal to the union of
   FDR-significant genes from the selected production branches.

The discrete and continuous branches are parallel detectors, not mutually
exclusive trajectory classes. The balance is learned using age, so final
branch p-values use full-pipeline age permutation: every permutation repeats
the complete age-dependent feature-learning step before recomputing the branch
statistic. IsoPlex/effective-isoform membership is held fixed because it depends
only on expression, not age.

Optional windowed analysis
--------------------------
The source code retains ``localized`` as an explicitly requested analysis mode,
but it is not part of the production workflow and has no default windows.
Users must provide ``WindowConfig(edges=...)``. Within each supplied interval,
IsoVita runs the same two hypotheses used globally: the discrete change-point
test and the continuous spline test. There is no separate localized trajectory
model and no max-over-windows localized statistic.

Windowed p-values remain full-pipeline selection-aware permutation p-values.
BH FDR is applied separately to all gene x window discrete hypotheses and to all
gene x window continuous hypotheses. Windowed discoveries are returned as
separate outputs and never enter the production ``final_features`` union.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import f as f_dist, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


# ============================================================================
# CONFIGURATION
# ============================================================================


def _format_edge(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


@dataclass(frozen=True)
class WindowConfig:
    """Explicit age-window specification for optional windowed analysis.

    ``edges=(20, 40, 60, 85)`` defines the intervals 20-40, 40-60, and 60-85.
    Intervals are left-closed/right-open, except the final interval also includes
    its upper edge. No windows are supplied by IsoVita by default.
    """

    edges: Tuple[float, ...]

    def __post_init__(self) -> None:
        edges = tuple(float(x) for x in self.edges)
        if len(edges) < 2:
            raise ValueError("At least two window edges are required.")
        if any(not np.isfinite(x) for x in edges):
            raise ValueError("Window edges must be finite numeric values.")
        if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
            raise ValueError("Window edges must be strictly increasing.")
        object.__setattr__(self, "edges", edges)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(
            f"{_format_edge(a)}-{_format_edge(b)}"
            for a, b in zip(self.edges[:-1], self.edges[1:])
        )

    @property
    def intervals(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(zip(self.edges[:-1], self.edges[1:]))


@dataclass(frozen=True)
class IsoPlexConfig:
    """Preprocessing settings used before IsoPlex."""

    tpm_threshold: float = 0.5
    min_samples_above_tpm: int = 5
    require_positive_gene_total_every_sample: bool = True


@dataclass(frozen=True)
class BalanceConfig:
    """One-feature-per-gene CLR -> trajectory PC1 -> sparse balance settings."""

    spline_df: int = 4
    spline_degree: int = 3
    age_grid_size: int = 100
    clr_pseudocount: float = 1e-8
    min_samples: int = 20
    near_optimal_cosine_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.spline_df < self.spline_degree:
            raise ValueError("spline_df must be >= spline_degree.")
        if self.age_grid_size < 10:
            raise ValueError("age_grid_size must be at least 10.")
        if self.clr_pseudocount <= 0:
            raise ValueError("clr_pseudocount must be > 0.")
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3.")
        if not (0 <= self.near_optimal_cosine_tolerance < 1):
            raise ValueError(
                "near_optimal_cosine_tolerance must lie in [0, 1)."
            )


@dataclass(frozen=True)
class DiscreteConfig:
    """Pure-step change-point settings for the discrete branch."""

    min_samples_side: int = 10
    min_fraction_side: float = 0.15
    alpha_fdr: float = 0.05

    def __post_init__(self) -> None:
        if self.min_samples_side < 2:
            raise ValueError("min_samples_side must be at least 2.")
        if not (0.0 <= self.min_fraction_side < 0.5):
            raise ValueError("min_fraction_side must lie in [0, 0.5).")
        if not (0.0 < self.alpha_fdr < 1.0):
            raise ValueError("alpha_fdr must lie in (0, 1).")


@dataclass(frozen=True)
class ContinuousConfig:
    """Spline-test settings for the continuous branch."""

    spline_df: int = 3
    spline_degree: int = 3
    alpha_fdr: float = 0.05
    min_samples: int = 20
    robust_amp_q_low: float = 0.05
    robust_amp_q_high: float = 0.95

    def __post_init__(self) -> None:
        if self.spline_df < self.spline_degree:
            raise ValueError("spline_df must be >= spline_degree.")
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3.")
        if not (0.0 < self.alpha_fdr < 1.0):
            raise ValueError("alpha_fdr must lie in (0, 1).")
        if not (0.0 <= self.robust_amp_q_low < self.robust_amp_q_high <= 1.0):
            raise ValueError(
                "robust amplitude quantiles must satisfy 0 <= low < high <= 1."
            )


@dataclass(frozen=True)
class PermutationInferenceConfig:
    """Selection-aware full-pipeline age-permutation inference settings.

    ``batch_size`` controls how many age permutations are processed together
    during the vectorized pass. It changes memory/runtime only; it does not
    change the permutation set or inferential definition.
    """

    n_permutations: int = 1000
    random_seed: int = 0
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")


@dataclass(frozen=True)
class SelectionConfig:
    """Final union/ranking settings for production branches."""

    top_n_genes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.top_n_genes is not None and self.top_n_genes < 1:
            raise ValueError("top_n_genes must be None or >= 1.")


@dataclass(frozen=True)
class IsoVitaConfig:
    """Top-level configuration for IsoVita.

    ``windows`` is intentionally ``None`` by default. It is required only when
    the optional ``localized`` analysis mode is explicitly requested.
    """

    isoplex: IsoPlexConfig = field(default_factory=IsoPlexConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    discrete: DiscreteConfig = field(default_factory=DiscreteConfig)
    continuous: ContinuousConfig = field(default_factory=ContinuousConfig)
    inference: PermutationInferenceConfig = field(
        default_factory=PermutationInferenceConfig
    )
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    windows: Optional[WindowConfig] = None


@dataclass
class IsoVitaPipelineResult:
    """Container for shared intermediates and requested analysis outputs."""

    selected_branches: Tuple[str, ...]
    metadata: pd.DataFrame
    tpm: pd.DataFrame
    isoplex_expression_df: pd.DataFrame
    isoplex_filter_summary: pd.DataFrame
    sample_metrics: pd.DataFrame
    global_metrics: pd.DataFrame
    gene_routes: pd.DataFrame
    balance_summary: pd.DataFrame
    balance_loadings: pd.DataFrame
    balance_feature_matrix: pd.DataFrame
    discrete_trajectory_results: pd.DataFrame
    continuous_trajectory_results: pd.DataFrame
    localized_discrete_results: pd.DataFrame
    localized_continuous_results: pd.DataFrame
    gene_trajectory_summary: pd.DataFrame
    discrete_trajectory_candidates: pd.DataFrame
    continuous_trajectory_candidates: pd.DataFrame
    localized_discrete_candidates: pd.DataFrame
    localized_continuous_candidates: pd.DataFrame
    final_features: pd.DataFrame


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def robust_amplitude(
    values: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, q_high) - np.quantile(values, q_low))


def add_bh_fdr(
    df: pd.DataFrame,
    p_col: str,
    *,
    alpha: float = 0.05,
    valid_mask: Optional[pd.Series] = None,
    q_col: str = "q_value",
    reject_col: str = "fdr_significant",
) -> pd.DataFrame:
    """Return a copy with BH-adjusted p-values and rejection flags."""
    out = df.copy()

    if valid_mask is None:
        valid_mask = out[p_col].notna()
    else:
        valid_mask = pd.Series(valid_mask, index=out.index).fillna(False)
        valid_mask = valid_mask & out[p_col].notna()

    out[q_col] = np.nan
    out[reject_col] = False

    if int(valid_mask.sum()) > 0:
        reject, qvals, _, _ = multipletests(
            out.loc[valid_mask, p_col].to_numpy(dtype=float),
            alpha=alpha,
            method="fdr_bh",
        )
        out.loc[valid_mask, q_col] = qvals
        out.loc[valid_mask, reject_col] = reject

    return out


# ============================================================================
# CORE INPUT CONTRACT / ISOPLEX TPM FILTER
# ============================================================================


PRODUCTION_BRANCHES: Tuple[str, ...] = ("discrete", "continuous")
VALID_BRANCHES: Tuple[str, ...] = PRODUCTION_BRANCHES + ("localized",)


def normalize_branches(branches: Sequence[str]) -> Tuple[str, ...]:
    """Validate branch selection and return it in canonical algorithm order."""
    if isinstance(branches, str):
        requested = [branches]
    else:
        requested = [str(branch) for branch in branches]

    if not requested:
        raise ValueError(
            "At least one IsoVita branch must be selected from: "
            f"{', '.join(VALID_BRANCHES)}."
        )

    unknown = sorted(set(requested) - set(VALID_BRANCHES))
    if unknown:
        raise ValueError(
            f"Unknown IsoVita branch(es): {unknown}. "
            f"Choose from: {', '.join(VALID_BRANCHES)}."
        )

    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate branch names are not allowed.")

    return tuple(branch for branch in VALID_BRANCHES if branch in requested)


def validate_core_inputs(
    tpm: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Validate standardized TPM/metadata without performing dataset filtering."""
    required_tpm = {"gene_id", "transcript_id"}
    missing_tpm = required_tpm - set(tpm.columns)
    if missing_tpm:
        raise ValueError(
            "TPM input must already be gene-annotated. Missing columns: "
            f"{sorted(missing_tpm)}. Run prepare_isovita_inputs.py first."
        )

    required_meta = {"sample_id", "age"}
    missing_meta = required_meta - set(metadata.columns)
    if missing_meta:
        raise ValueError(
            "Metadata must already contain canonical columns sample_id and age. "
            f"Missing: {sorted(missing_meta)}."
        )

    expr = tpm.copy()
    expr["gene_id"] = expr["gene_id"].astype(str)
    expr["transcript_id"] = expr["transcript_id"].astype(str)

    if expr["transcript_id"].duplicated().any():
        duplicated = (
            expr.loc[expr["transcript_id"].duplicated(keep=False), "transcript_id"]
            .unique()
            .tolist()
        )
        raise ValueError(
            "TPM transcript_id values must be unique. "
            f"Examples: {duplicated[:10]}"
        )

    meta = metadata.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    if meta["sample_id"].duplicated().any():
        raise ValueError("Metadata sample_id values must be unique.")
    if meta["age"].isna().any():
        raise ValueError("Metadata age must be numeric and non-missing.")

    sample_cols = [
        str(column) for column in expr.columns
        if column not in {"gene_id", "transcript_id"}
    ]
    if not sample_cols:
        raise ValueError("TPM input contains no sample columns.")

    rename_map = {
        column: str(column)
        for column in expr.columns
        if column not in {"gene_id", "transcript_id"}
    }
    expr = expr.rename(columns=rename_map)

    if len(sample_cols) != len(set(sample_cols)):
        raise ValueError("TPM sample columns are not unique after string conversion.")

    metadata_samples = meta["sample_id"].tolist()
    if sample_cols != metadata_samples:
        raise ValueError(
            "TPM sample columns must exactly match metadata sample_id values in "
            "the same order. Harmonize/filter samples in prepare_isovita_inputs.py."
        )

    numeric = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("TPM sample values must be finite numeric values.")
    values = numeric.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("TPM sample values must be finite numeric values.")
    if np.any(values < 0):
        raise ValueError("TPM values must be non-negative.")
    expr[sample_cols] = numeric

    meta = meta.set_index("sample_id", drop=True)
    meta.index.name = "sample"
    return expr, meta, sample_cols





def filter_tpm_for_isoplex(
    tpm: pd.DataFrame,
    sample_cols: Sequence[str],
    *,
    config: IsoPlexConfig = IsoPlexConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the algorithm's TPM eligibility filters before running IsoPlex."""
    expr = tpm.copy()
    sample_cols = list(sample_cols)

    summary_rows = [{
        "step": "core_tpm_input",
        "n_transcripts": int(expr["transcript_id"].nunique()),
        "n_genes": int(expr["gene_id"].nunique()),
    }]

    keep_tx = (
        (expr[sample_cols] > config.tpm_threshold).sum(axis=1)
        >= config.min_samples_above_tpm
    )
    expr = expr.loc[keep_tx].copy()
    summary_rows.append({
        "step": "transcript_filter",
        "n_transcripts": int(expr["transcript_id"].nunique()),
        "n_genes": int(expr["gene_id"].nunique()),
    })

    if config.require_positive_gene_total_every_sample and not expr.empty:
        gene_total = expr.groupby("gene_id")[sample_cols].transform("sum")
        keep_gene = (gene_total > 0).all(axis=1)
        expr = expr.loc[keep_gene].copy()

    summary_rows.append({
        "step": "gene_total_filter",
        "n_transcripts": int(expr["transcript_id"].nunique()),
        "n_genes": int(expr["gene_id"].nunique()),
    })

    if expr.empty:
        raise ValueError(
            "No transcripts remain after the configured IsoPlex TPM filters."
        )

    return expr, pd.DataFrame(summary_rows)


# ============================================================================
# ISOPLEX / EFFECTIVE-ISOFORM ROUTING
# ============================================================================


def run_isoplex(expression_df: pd.DataFrame):
    """Run IsoPlex on a preprocessed expression table."""
    try:
        import isoplex
    except ImportError as exc:
        raise ImportError("IsoPlex is required for run_isoplex().") from exc

    return isoplex.compute_multi_sample_isoform_metrics(
        expression_df,
        gene_col="gene_id",
        feature_col="transcript_id",
        expression_type="tpm",
    )


def get_effective_isoforms(
    global_metrics: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Map each gene to its IsoPlex-effective transcript IDs."""
    required = {"gene_id", "transcript_id", "effective"}
    missing = required - set(global_metrics.columns)
    if missing:
        raise ValueError(f"IsoPlex global metrics missing: {sorted(missing)}")

    eff = global_metrics.loc[
        global_metrics["effective"].astype(bool),
        ["gene_id", "transcript_id"],
    ].copy()
    eff["gene_id"] = eff["gene_id"].astype(str)
    eff["transcript_id"] = eff["transcript_id"].astype(str)

    return (
        eff.groupby("gene_id")["transcript_id"]
        .apply(lambda x: list(dict.fromkeys(x)))
        .to_dict()
    )


def route_genes_by_effective_isoform_count(
    global_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Route genes under the unified condensation design.

    There is intentionally NO statistical distinction between K=2 and K>2.
    Both use the same CLR -> trajectory PC1 -> sparse-balance definition.
    """
    effective_map = get_effective_isoforms(global_metrics)
    all_genes = pd.unique(global_metrics["gene_id"].astype(str))

    rows = []
    for gene_id in all_genes:
        isoforms = effective_map.get(str(gene_id), [])
        n_eff = len(isoforms)
        route = "exclude_lt2" if n_eff < 2 else "balance_ge2"

        rows.append({
            "gene_id": str(gene_id),
            "n_effective": n_eff,
            "route": route,
            "effective_isoforms": tuple(isoforms),
        })

    return pd.DataFrame(rows).sort_values("gene_id").reset_index(drop=True)


# ============================================================================
# ONE FEATURE PER GENE: CLR -> SMOOTH -> SVD/PC1 -> SPARSE BALANCE
# ============================================================================


def _stabilize_composition(
    fractions: np.ndarray,
    pseudocount: float,
) -> np.ndarray:
    """Add a small positive floor and renormalize each sample composition."""
    x = np.asarray(fractions, dtype=float)
    if x.ndim != 2:
        raise ValueError("fractions must be a 2D sample x isoform matrix.")
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("fractions must be finite and non-negative.")

    x = x + float(pseudocount)
    row_sums = x.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Composition has a non-positive row sum after stabilization.")
    return x / row_sums


def clr_transform(composition: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform for a strictly positive composition."""
    x = np.asarray(composition, dtype=float)
    if np.any(x <= 0):
        raise ValueError("CLR requires strictly positive compositions.")
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def make_effective_isoform_composition(
    gene_id: str,
    expression_df: pd.DataFrame,
    effective_isoforms: Sequence[str],
    sample_cols: Sequence[str],
    metadata: pd.DataFrame,
    *,
    pseudocount: float,
    min_samples: int,
    gene_row_index: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, object]]:
    """
    Build the effective-isoform composition for one gene.

    Only IsoPlex-effective isoforms are retained, and they are renormalized to
    sum to one within that effective set for every usable sample.

    ``gene_row_index`` is an optional positional-row lookup created once for the
    expression table. Supplying it avoids a full-table boolean scan for every
    gene while leaving the resulting composition unchanged.
    """
    gene_id = str(gene_id)
    effective_isoforms = list(dict.fromkeys(map(str, effective_isoforms)))

    if len(effective_isoforms) < 2:
        return None

    if gene_row_index is not None:
        rows = gene_row_index.get(gene_id)
        if rows is None:
            return None
        gene = expression_df.iloc[np.asarray(rows, dtype=np.int64)].copy()
    else:
        # Backward-compatible fallback for direct external calls.
        gene = expression_df.loc[
            expression_df["gene_id"].astype(str).eq(gene_id)
        ].copy()

    if gene.empty:
        return None

    gene["transcript_id"] = gene["transcript_id"].astype(str)
    gene = gene.loc[gene["transcript_id"].isin(effective_isoforms)].copy()
    if gene["transcript_id"].nunique() < 2:
        return None

    # Preserve the effective-isoform order from IsoPlex.
    present = set(gene["transcript_id"])
    ordered_isoforms = [tx for tx in effective_isoforms if tx in present]
    if len(ordered_isoforms) < 2:
        return None

    mat = (
        gene.set_index("transcript_id")
        .loc[ordered_isoforms, list(sample_cols)]
        .astype(float)
        .T
    )
    mat.index = mat.index.astype(str)

    common_samples = [sample for sample in mat.index if sample in metadata.index]
    mat = mat.loc[common_samples]

    totals = mat.sum(axis=1)
    valid = np.isfinite(totals.to_numpy()) & (totals.to_numpy() > 0)
    mat = mat.loc[valid]

    if len(mat) < min_samples:
        return None

    raw_fractions = mat.div(mat.sum(axis=1), axis=0).to_numpy(dtype=float)
    composition = _stabilize_composition(raw_fractions, pseudocount)

    return {
        "gene_id": gene_id,
        "isoforms": ordered_isoforms,
        "samples": mat.index.to_numpy(dtype=str),
        "ages": metadata.loc[mat.index, "age"].to_numpy(dtype=float),
        "raw_fractions": raw_fractions,
        "composition": composition,
    }

def _build_gene_row_index(expression_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build a one-time gene -> positional-row lookup for a transcript table."""
    return {
        str(gene_id): np.asarray(rows, dtype=np.int64)
        for gene_id, rows in expression_df.groupby("gene_id", sort=False).indices.items()
    }


def _array_cache_token(values: np.ndarray) -> Tuple[int, bytes]:
    """Hashable exact token for a small numeric vector used in run-local caches."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return int(arr.size), arr.tobytes()


def _build_observed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: int,
) -> Dict[str, object]:
    """Build observed-pass spline objects once per age vector."""
    ages = np.asarray(ages, dtype=float)

    # Keep the original observed-fit ordering behavior for numerical continuity.
    order = np.argsort(ages)
    ages_sorted = ages[order]

    formula = (
        f"bs(age, df={int(spline_df)}, degree={int(spline_degree)}, "
        "include_intercept=False)"
    )
    X_df = dmatrix(
        formula,
        {"age": ages_sorted},
        return_type="dataframe",
    )
    X = np.asarray(X_df, dtype=float)

    age_grid = np.linspace(
        float(np.min(ages_sorted)),
        float(np.max(ages_sorted)),
        int(age_grid_size),
    )
    X_grid = np.asarray(
        build_design_matrices(
            [X_df.design_info],
            {"age": age_grid},
        )[0],
        dtype=float,
    )

    return {
        "order": order,
        "ages_sorted": ages_sorted,
        "X": X,
        "age_grid": age_grid,
        "X_grid": X_grid,
    }


def _get_cached_observed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: int,
    cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]],
) -> Dict[str, object]:
    """Reuse observed-feature Patsy bases across genes sharing the same age vector."""
    if cache is None:
        return _build_observed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )

    key = (
        "observed",
        int(spline_df),
        int(spline_degree),
        int(age_grid_size),
        *_array_cache_token(ages),
    )
    workspace = cache.get(key)
    if workspace is None:
        workspace = _build_observed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )
        cache[key] = workspace
    return workspace


def _get_cached_fixed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: Optional[int] = None,
    cache: Optional[Dict[Tuple[object, ...], Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Reuse permutation-inference spline bases across genes with the same age multiset."""
    ages = np.asarray(ages, dtype=float)
    ages_sorted = np.sort(ages, kind="mergesort")

    if cache is None:
        return _build_fixed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )

    key = (
        "fixed",
        int(spline_df),
        int(spline_degree),
        None if age_grid_size is None else int(age_grid_size),
        *_array_cache_token(ages_sorted),
    )
    workspace = cache.get(key)
    if workspace is None:
        workspace = _build_fixed_age_spline_workspace(
            ages,
            spline_df=spline_df,
            spline_degree=spline_degree,
            age_grid_size=age_grid_size,
        )
        cache[key] = workspace
    return workspace




def _build_fixed_age_spline_workspace(
    ages: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: Optional[int] = None,
) -> Dict[str, object]:
    """Precompute a spline basis for a fixed multiset of age values.

    Age permutations preserve the multiset of ages. Therefore the spline basis,
    its rank, and (for feature learning) the prediction grid are unchanged; only
    the expression/feature values assigned to those ages change.
    """
    ages = np.asarray(ages, dtype=float)
    order = np.argsort(ages, kind="mergesort")
    ages_sorted = ages[order]
    formula = (
        f"bs(age, df={int(spline_df)}, degree={int(spline_degree)}, "
        "include_intercept=False)"
    )
    X_df = dmatrix(
        formula,
        {"age": ages_sorted},
        return_type="dataframe",
    )
    X = np.asarray(X_df, dtype=float)
    workspace: Dict[str, object] = {
        "ages_sorted": ages_sorted,
        "X": X,
        "X_pinv": np.linalg.pinv(X),
        "rank": int(np.linalg.matrix_rank(X)),
    }

    if age_grid_size is not None:
        age_grid = np.linspace(
            float(np.min(ages_sorted)),
            float(np.max(ages_sorted)),
            int(age_grid_size),
        )
        X_grid = np.asarray(
            build_design_matrices(
                [X_df.design_info],
                {"age": age_grid},
            )[0],
            dtype=float,
        )
        workspace["age_grid"] = age_grid
        workspace["X_grid"] = X_grid
        workspace["X_grid_centered"] = (
            X_grid - X_grid.mean(axis=0, keepdims=True)
        )

    return workspace




def _spline_f_statistics_from_sorted_matrix(
    y_sorted_matrix: np.ndarray,
    workspace: Dict[str, object],
) -> np.ndarray:
    """Vectorized spline F statistics for rows already ordered by fixed ages."""
    Y = np.asarray(y_sorted_matrix, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    if Y.ndim != 2:
        raise ValueError("y_sorted_matrix must be 1D or 2D.")

    X = np.asarray(workspace["X"], dtype=float)
    X_pinv = np.asarray(workspace["X_pinv"], dtype=float)
    if Y.shape[1] != X.shape[0]:
        raise ValueError("Feature matrix does not match the spline workspace.")

    rank1 = int(workspace["rank"])
    df_diff = rank1 - 1
    df_resid = Y.shape[1] - rank1
    if df_diff <= 0 or df_resid <= 0:
        return np.full(Y.shape[0], np.nan, dtype=float)

    # beta: batch x p; fit: batch x n
    beta = Y @ X_pinv.T
    fit1 = beta @ X.T

    means = Y.mean(axis=1, keepdims=True)
    rss0 = np.sum((Y - means) ** 2, axis=1)
    rss1 = np.sum((Y - fit1) ** 2, axis=1)
    delta = np.maximum(rss0 - rss1, 0.0)

    eps = np.finfo(float).eps * np.maximum(1.0, rss0) * 32.0
    out = np.empty(Y.shape[0], dtype=float)

    constant = rss0 <= eps
    perfect = (~constant) & (rss1 <= eps)
    regular = (~constant) & (~perfect)

    out[constant] = 0.0
    out[perfect] = np.where(delta[perfect] > eps[perfect], np.inf, 0.0)

    if np.any(regular):
        numerator = delta[regular] / df_diff
        denominator = rss1[regular] / df_resid
        out[regular] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )

    return out


def _max_step_f_statistics_from_sorted_matrix(
    y_sorted_matrix: np.ndarray,
    cut_positions: np.ndarray,
) -> np.ndarray:
    """Vectorized maximized pure-step F statistic for sorted feature rows."""
    Y = np.asarray(y_sorted_matrix, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    cuts = np.asarray(cut_positions, dtype=int)
    if Y.ndim != 2:
        raise ValueError("y_sorted_matrix must be 1D or 2D.")
    if cuts.size == 0:
        return np.zeros(Y.shape[0], dtype=float)

    n = Y.shape[1]
    if n < 3:
        return np.full(Y.shape[0], np.nan, dtype=float)

    csum = np.cumsum(Y, axis=1)
    cssq = np.cumsum(Y * Y, axis=1)

    total_sum = csum[:, -1][:, None]
    total_ssq = cssq[:, -1][:, None]

    left_idx = cuts - 1
    sum_left = csum[:, left_idx]
    ssq_left = cssq[:, left_idx]
    sum_right = total_sum - sum_left
    ssq_right = total_ssq - ssq_left

    n_left = cuts[None, :].astype(float)
    n_right = (n - cuts)[None, :].astype(float)

    rss_left = np.maximum(
        ssq_left - (sum_left * sum_left) / n_left,
        0.0,
    )
    rss_right = np.maximum(
        ssq_right - (sum_right * sum_right) / n_right,
        0.0,
    )
    rss1 = np.maximum(rss_left + rss_right, 0.0)

    rss0 = np.maximum(
        total_ssq - (total_sum * total_sum) / n,
        0.0,
    )
    delta = np.maximum(rss0 - rss1, 0.0)
    denom = rss1 / (n - 2)

    eps = np.finfo(float).eps * np.maximum(1.0, rss0) * 32.0
    regular = denom > eps

    f_stats = np.zeros_like(rss1, dtype=float)
    f_stats[regular] = delta[regular] / denom[regular]
    f_stats[(~regular) & (delta > eps)] = np.inf

    return np.max(f_stats, axis=1)

def _orient_pc1(pc1: np.ndarray) -> np.ndarray:
    """Give the SVD loading vector a deterministic sign."""
    v = np.asarray(pc1, dtype=float).copy()
    idx = int(np.argmax(np.abs(v)))
    if v[idx] < 0:
        v *= -1.0
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("PC1 has zero norm.")
    return v / norm


def _assignment_to_balance_direction(assignment: np.ndarray) -> np.ndarray:
    """
    Convert {-1,0,+1} group labels to a zero-sum unit-norm balance direction.

    Group A receives equal positive weights, Group B equal negative weights,
    and unused isoforms receive zero weight. Dividing each group by its size
    enforces a zero sum; unit normalization gives a cosine-comparable vector.
    """
    z = np.asarray(assignment, dtype=np.int8)
    a = np.flatnonzero(z == 1)
    b = np.flatnonzero(z == -1)

    if len(a) == 0 or len(b) == 0:
        raise ValueError("A valid balance requires non-empty A and B groups.")

    direction = np.zeros(len(z), dtype=float)
    direction[a] = 1.0 / len(a)
    direction[b] = -1.0 / len(b)

    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Balance direction has zero norm.")

    return direction / norm


def _find_sparse_balance_exact(
    pc1: np.ndarray,
    *,
    near_optimal_cosine_tolerance: float,
) -> Dict[str, object]:
    """
    Exact polynomial-time sparse-balance search for K >= 3.

    For fixed group sizes (a, b), the cosine-maximizing balance places:
        - the a largest PC1 loadings in Group A
        - the b smallest PC1 loadings in Group B

    Therefore we only need to search group sizes rather than all
    {-1, 0, +1} assignments.

    This exactly preserves the balance-selection rule:
        1. find the globally best cosine;
        2. define the near-optimal set using the configured tolerance;
        3. choose the smallest total number of used isoforms;
        4. within that complexity, choose the highest cosine.

    Notes
    -----
    The selected balance, best cosine, chosen cosine, and n_used are exact.

    The exhaustive diagnostic ``n_near_optimal`` is not computed because
    counting every individual near-optimal assignment is itself combinatorial.
    It is diagnostic only and is not used in feature construction or inference.
    """
    v = np.asarray(pc1, dtype=float)

    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("PC1 has zero norm.")

    v = v / norm
    k = len(v)

    if k < 3:
        raise ValueError("Exact optimizer is defined here for K >= 3.")

    # Sort once. For fixed (a, b), A uses the a largest loadings and B uses
    # the b smallest. Stable sorting makes exact loading ties deterministic.
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]

    # prefix[j] = sum of the j smallest sorted loadings.
    prefix = np.empty(k + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(v_sorted, out=prefix[1:])
    total_sum = float(prefix[k])

    # For each total complexity m = a + b, record the best cosine attainable
    # by any balance using exactly m isoforms.
    best_by_used = np.full(k + 1, -np.inf, dtype=float)
    best_a_by_used = np.full(k + 1, -1, dtype=int)

    for m in range(2, k + 1):
        a = np.arange(1, m, dtype=int)
        b = m - a

        a_float = a.astype(float)
        b_float = b.astype(float)

        # Sum of the a largest and b smallest loadings.
        sum_a = total_sum - prefix[k - a]
        sum_b = prefix[b]

        mean_difference = sum_a / a_float - sum_b / b_float
        balance_norm = np.sqrt(1.0 / a_float + 1.0 / b_float)
        cosine = mean_difference / balance_norm

        # Deterministic first-tie behavior within a fixed total complexity.
        idx = int(np.argmax(cosine))
        best_by_used[m] = float(cosine[idx])
        best_a_by_used[m] = int(a[idx])

    # Original IsoVita rule: global optimum -> tolerance band -> simplest
    # complexity -> best cosine within that complexity.
    best = float(np.max(best_by_used[2:]))
    tolerance = float(near_optimal_cosine_tolerance)
    threshold = best - tolerance - 1e-12

    eligible_complexities = np.flatnonzero(best_by_used >= threshold)
    eligible_complexities = eligible_complexities[eligible_complexities >= 2]
    if eligible_complexities.size == 0:
        raise RuntimeError("No near-optimal balance complexity found.")

    n_used = int(eligible_complexities[0])
    a_size = int(best_a_by_used[n_used])
    b_size = int(n_used - a_size)

    assignment = np.zeros(k, dtype=np.int8)
    b_idx = order[:b_size]
    a_idx = order[k - a_size:]
    assignment[a_idx] = 1
    assignment[b_idx] = -1

    direction = _assignment_to_balance_direction(assignment)
    chosen_cosine = float(abs(np.dot(direction, v)))

    # Orient A/B toward the deterministically oriented PC1.
    if np.dot(direction, v) < 0:
        assignment = -assignment
        direction = -direction

    return {
        "assignment": assignment.astype(np.int8),
        "direction": direction.astype(float),
        "cosine_similarity": chosen_cosine,
        "best_cosine_similarity": best,
        "n_used": n_used,
        "n_near_optimal": np.nan,
    }



def _find_sparse_balance_assignments_batch(
    pc1_batch: np.ndarray,
    *,
    near_optimal_cosine_tolerance: float,
) -> np.ndarray:
    """Vectorized exact sparse-balance assignments for a batch of PC1 vectors.

    This applies the same optimization rule as ``find_sparse_balance`` but
    returns only the assignment matrix needed during permutation inference.
    """
    V = np.asarray(pc1_batch, dtype=float)
    if V.ndim == 1:
        V = V[None, :]
    if V.ndim != 2:
        raise ValueError("pc1_batch must be 1D or 2D.")

    batch, k = V.shape
    if k < 2:
        raise ValueError("At least two isoforms are required for a balance.")

    norms = np.linalg.norm(V, axis=1)
    if np.any(norms == 0):
        raise ValueError("PC1 has zero norm.")

    V = V / norms[:, None]

    # Match _orient_pc1(): orient by the loading with largest absolute magnitude.
    pivot = np.argmax(np.abs(V), axis=1)
    flip = V[np.arange(batch), pivot] < 0
    if np.any(flip):
        V[flip] *= -1.0

    if k == 2:
        assignments = np.tile(
            np.asarray([1, -1], dtype=np.int8),
            (batch, 1),
        )
        # The scalar implementation orients the balance toward PC1.
        # For K=2 this is equivalent to flipping when v0-v1 < 0.
        flip_balance = (V[:, 0] - V[:, 1]) < 0
        assignments[flip_balance] *= -1
        return assignments

    order = np.argsort(V, axis=1, kind="mergesort")
    V_sorted = np.take_along_axis(V, order, axis=1)

    prefix = np.zeros((batch, k + 1), dtype=float)
    prefix[:, 1:] = np.cumsum(V_sorted, axis=1)
    total_sum = prefix[:, -1]

    best_by_used = np.full((batch, k + 1), -np.inf, dtype=float)
    best_a_by_used = np.full((batch, k + 1), -1, dtype=np.int16)

    row_idx = np.arange(batch)
    for m in range(2, k + 1):
        a = np.arange(1, m, dtype=int)
        b = m - a

        sum_a = total_sum[:, None] - prefix[:, k - a]
        sum_b = prefix[:, b]

        mean_difference = (
            sum_a / a[None, :]
            - sum_b / b[None, :]
        )
        balance_norm = np.sqrt(
            1.0 / a[None, :]
            + 1.0 / b[None, :]
        )
        cosine = mean_difference / balance_norm

        best_idx = np.argmax(cosine, axis=1)
        best_by_used[:, m] = cosine[row_idx, best_idx]
        best_a_by_used[:, m] = a[best_idx]

    best = np.max(best_by_used[:, 2:], axis=1)
    threshold = (
        best
        - float(near_optimal_cosine_tolerance)
        - 1e-12
    )

    eligible = best_by_used[:, 2:] >= threshold[:, None]
    if np.any(~eligible.any(axis=1)):
        raise RuntimeError("No near-optimal balance complexity found.")

    n_used = np.argmax(eligible, axis=1) + 2
    a_size = best_a_by_used[row_idx, n_used].astype(int)
    b_size = n_used - a_size

    # ranks[b, isoform] gives that isoform's ascending loading rank.
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(k), order.shape),
        axis=1,
    )

    assignments = np.zeros((batch, k), dtype=np.int8)
    assignments[ranks < b_size[:, None]] = -1
    assignments[ranks >= (k - a_size)[:, None]] = 1

    return assignments


def _balance_features_from_assignments_batch(
    log_composition: np.ndarray,
    assignments: np.ndarray,
) -> np.ndarray:
    """Compute log-GM balance features for many assignments at once."""
    log_x = np.asarray(log_composition, dtype=float)
    A = np.asarray(assignments, dtype=np.int8)
    if log_x.ndim != 2 or A.ndim != 2:
        raise ValueError("log_composition and assignments must both be 2D.")
    if log_x.shape[1] != A.shape[1]:
        raise ValueError("Assignment width must equal the number of isoforms.")

    n_a = np.sum(A == 1, axis=1)
    n_b = np.sum(A == -1, axis=1)
    if np.any(n_a == 0) or np.any(n_b == 0):
        raise ValueError("Every balance requires non-empty A and B groups.")

    # Row-wise weights are +1/|A| for Group A and -1/|B| for Group B.
    weights = np.where(
        A == 1,
        1.0 / n_a[:, None],
        np.where(A == -1, -1.0 / n_b[:, None], 0.0),
    )

    return weights @ log_x.T


def _learn_permuted_balance_features_batch(
    clr_values: np.ndarray,
    permutation_orders: np.ndarray,
    *,
    config: BalanceConfig,
    spline_workspace: Dict[str, object],
    log_composition: np.ndarray,
) -> np.ndarray:
    """Relearn sparse balances for a batch of age permutations."""
    orders = np.asarray(permutation_orders, dtype=np.int64)
    if orders.ndim != 2:
        raise ValueError("permutation_orders must be a 2D matrix.")

    clr = np.asarray(clr_values, dtype=float)
    log_x = np.asarray(log_composition, dtype=float)
    if clr.ndim != 2 or clr.shape[0] != orders.shape[1]:
        raise ValueError("CLR values do not align with permutation orders.")
    if log_x.shape != clr.shape:
        raise ValueError("log_composition must have the same shape as CLR values.")

    X_pinv = np.asarray(spline_workspace["X_pinv"], dtype=float)
    X_grid_centered = np.asarray(spline_workspace["X_grid_centered"], dtype=float)

    # batch x sample x isoform
    clr_sorted = clr[orders]

    # batch x basis x isoform
    beta = np.matmul(X_pinv[None, :, :], clr_sorted)

    # batch x age_grid x isoform
    centered_grid = np.matmul(X_grid_centered[None, :, :], beta)

    # Batched SVD; only the first right singular vector is needed.
    _, _, vt = np.linalg.svd(centered_grid, full_matrices=False)
    assignments = _find_sparse_balance_assignments_batch(
        vt[:, 0, :],
        near_optimal_cosine_tolerance=config.near_optimal_cosine_tolerance,
    )

    return _balance_features_from_assignments_batch(log_x, assignments)


def find_sparse_balance(
    pc1: np.ndarray,
    *,
    near_optimal_cosine_tolerance: float = 0.01,
) -> Dict[str, object]:
    """Find the simplest near-optimal A/B/unused balance exactly.

    K = 2 has one balance up to sign and is handled analytically. For every
    K >= 3, IsoVita uses the exact polynomial-time optimizer over group sizes.

    """
    pc1 = np.asarray(pc1, dtype=float)
    norm = np.linalg.norm(pc1)
    if norm == 0:
        raise ValueError("PC1 has zero norm.")

    pc1 = pc1 / norm
    k = len(pc1)
    if k < 2:
        raise ValueError("At least two isoforms are required for a balance.")

    if k == 2:
        assignment = np.asarray([1, -1], dtype=np.int8)
        direction = _assignment_to_balance_direction(assignment)
        cosine = float(abs(np.dot(direction, pc1)))
        if np.dot(direction, pc1) < 0:
            assignment = -assignment
            direction = -direction
        return {
            "assignment": assignment,
            "direction": direction,
            "cosine_similarity": cosine,
            "best_cosine_similarity": cosine,
            "n_used": 2,
            "n_near_optimal": 1,
        }

    return _find_sparse_balance_exact(
        pc1,
        near_optimal_cosine_tolerance=float(near_optimal_cosine_tolerance),
    )


def compute_balance_feature(
    composition: np.ndarray,
    assignment: np.ndarray,
) -> np.ndarray:
    """Compute log(GM(Group A) / GM(Group B)) for every sample."""
    x = np.asarray(composition, dtype=float)
    z = np.asarray(assignment, dtype=np.int8)

    a = np.flatnonzero(z == 1)
    b = np.flatnonzero(z == -1)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("A valid balance requires non-empty A and B groups.")
    if np.any(x <= 0):
        raise ValueError("Balance feature requires strictly positive composition.")

    log_x = np.log(x)
    return log_x[:, a].mean(axis=1) - log_x[:, b].mean(axis=1)


def _learn_balance_from_prepared(
    prepared: Dict[str, object],
    ages: np.ndarray,
    *,
    config: BalanceConfig,
    spline_workspace: Dict[str, object],
    clr_values: np.ndarray,
    sort_order: np.ndarray,
) -> Dict[str, object]:
    """Learn the observed age-informed sparse balance from a prepared gene."""
    isoforms = list(prepared["isoforms"])
    samples = np.asarray(prepared["samples"], dtype=str)
    composition = np.asarray(prepared["composition"], dtype=float)
    ages = np.asarray(ages, dtype=float)
    clr = np.asarray(clr_values, dtype=float)
    order = np.asarray(sort_order, dtype=np.int64)

    if len(ages) != len(samples):
        raise ValueError("ages must align one-to-one with prepared samples.")
    if np.any(~np.isfinite(ages)):
        raise ValueError("ages must be finite.")
    if clr.shape[0] != len(ages):
        raise ValueError("CLR values must align one-to-one with prepared samples.")

    clr_sorted = clr[order]
    X = np.asarray(spline_workspace["X"], dtype=float)
    if X.shape[0] != len(ages):
        raise ValueError(
            "Age vector does not match the feature-learning spline workspace."
        )

    # Observed pass retains the original least-squares numerical path.
    beta = np.linalg.lstsq(X, clr_sorted, rcond=None)[0]
    age_grid = np.asarray(spline_workspace["age_grid"], dtype=float)
    X_grid = np.asarray(spline_workspace["X_grid"], dtype=float)
    fitted_grid = X_grid @ beta
    centered_grid = fitted_grid - fitted_grid.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(centered_grid, full_matrices=False)
    pc1 = _orient_pc1(vt[0])

    balance = find_sparse_balance(
        pc1,
        near_optimal_cosine_tolerance=config.near_optimal_cosine_tolerance,
    )

    assignment = np.asarray(balance["assignment"], dtype=np.int8)
    direction = np.asarray(balance["direction"], dtype=float)
    feature = compute_balance_feature(composition, assignment)

    return {
        "isoforms": isoforms,
        "samples": samples,
        "ages": ages,
        "composition": composition,
        "age_grid": age_grid,
        "fitted_grid": fitted_grid,
        "centered_grid": centered_grid,
        "singular_values": singular_values,
        "pc1": pc1,
        "balance": balance,
        "assignment": assignment,
        "direction": direction,
        "feature": feature,
    }

def fit_gene_balance(
    gene_id: str,
    expression_df: pd.DataFrame,
    effective_isoforms: Sequence[str],
    sample_cols: Sequence[str],
    metadata: pd.DataFrame,
    *,
    config: BalanceConfig = BalanceConfig(),
    gene_row_index: Optional[Dict[str, np.ndarray]] = None,
    prepared: Optional[Dict[str, object]] = None,
    spline_workspace_cache: Optional[
        Dict[Tuple[object, ...], Dict[str, object]]
    ] = None,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series]:
    """Fit the complete one-feature-per-gene IsoVita balance for one gene."""
    gene_id = str(gene_id)

    if prepared is None:
        prepared = make_effective_isoform_composition(
            gene_id,
            expression_df,
            effective_isoforms,
            sample_cols,
            metadata,
            pseudocount=config.clr_pseudocount,
            min_samples=config.min_samples,
            gene_row_index=gene_row_index,
        )

    if prepared is None:
        summary = {
            "gene_id": gene_id,
            "status": "failed_input",
            "n_effective": len(effective_isoforms),
        }
        return summary, pd.DataFrame(), pd.Series(dtype=float, name=gene_id)

    isoforms = list(prepared["isoforms"])
    samples = np.asarray(prepared["samples"], dtype=str)
    ages = np.asarray(prepared["ages"], dtype=float)

    try:
        observed_workspace = _get_cached_observed_age_spline_workspace(
            ages,
            spline_df=config.spline_df,
            spline_degree=config.spline_degree,
            age_grid_size=config.age_grid_size,
            cache=spline_workspace_cache,
        )
        learned = _learn_balance_from_prepared(
            prepared,
            ages,
            config=config,
            spline_workspace=observed_workspace,
            clr_values=clr_transform(np.asarray(prepared["composition"], dtype=float)),
            sort_order=np.asarray(observed_workspace["order"], dtype=np.int64),
        )
        singular_values = np.asarray(learned["singular_values"], dtype=float)
        pc1 = np.asarray(learned["pc1"], dtype=float)
        balance = learned["balance"]
        assignment = np.asarray(learned["assignment"], dtype=np.int8)
        direction = np.asarray(learned["direction"], dtype=float)
        feature = np.asarray(learned["feature"], dtype=float)

        a_idx = np.flatnonzero(assignment == 1)
        b_idx = np.flatnonzero(assignment == -1)
        u_idx = np.flatnonzero(assignment == 0)

        group_a = [isoforms[i] for i in a_idx]
        group_b = [isoforms[i] for i in b_idx]
        unused = [isoforms[i] for i in u_idx]

        total_s2 = float(np.sum(singular_values ** 2))
        pc1_explained = (
            float(singular_values[0] ** 2 / total_s2)
            if total_s2 > 0 else np.nan
        )

        summary = {
            "gene_id": gene_id,
            "status": "ok",
            "n_effective": len(isoforms),
            "n_samples": len(samples),
            "group_A": "|".join(group_a),
            "group_B": "|".join(group_b),
            "unused_isoforms": "|".join(unused),
            "n_group_A": len(group_a),
            "n_group_B": len(group_b),
            "n_unused": len(unused),
            "n_used": int(balance["n_used"]),
            "cosine_similarity": float(balance["cosine_similarity"]),
            "best_cosine_similarity": float(balance["best_cosine_similarity"]),
            "cosine_gap_from_best": float(
                balance["best_cosine_similarity"] - balance["cosine_similarity"]
            ),
            "n_near_optimal": balance["n_near_optimal"],
            "pc1_explained_variance_ratio": pc1_explained,
            "feature_robust_amplitude": robust_amplitude(feature),
            "feature_sd": float(np.std(feature, ddof=1)) if len(feature) > 1 else 0.0,
            "balance_method": "CLR_smooth_PC1_sparse_balance",
        }

        loadings = pd.DataFrame({
            "gene_id": gene_id,
            "transcript_id": isoforms,
            "pc1_loading": pc1,
            "balance_assignment": assignment.astype(int),
            "balance_direction": direction,
        })

        feature_series = pd.Series(feature, index=samples, name=gene_id, dtype=float)
        feature_series.index.name = "sample"

        return summary, loadings, feature_series

    except Exception as exc:
        summary = {
            "gene_id": gene_id,
            "status": f"failed: {exc}",
            "n_effective": len(isoforms),
            "n_samples": len(samples),
        }
        return summary, pd.DataFrame(), pd.Series(dtype=float, name=gene_id)

def build_one_feature_per_gene(
    expression_df: pd.DataFrame,
    global_metrics: pd.DataFrame,
    sample_cols: Sequence[str],
    metadata: pd.DataFrame,
    *,
    config: BalanceConfig = BalanceConfig(),
    progress_every: Optional[int] = None,
    gene_row_index: Optional[Dict[str, np.ndarray]] = None,
    prepared_cache: Optional[Dict[str, Optional[Dict[str, object]]]] = None,
    spline_workspace_cache: Optional[
        Dict[Tuple[object, ...], Dict[str, object]]
    ] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one sparse-balance feature for every gene with >=2 effective isoforms."""
    effective_map = get_effective_isoforms(global_metrics)
    genes = [gene for gene, txs in effective_map.items() if len(txs) >= 2]

    if gene_row_index is None:
        gene_row_index = _build_gene_row_index(expression_df)
    if prepared_cache is None:
        prepared_cache = {}

    summary_rows: List[Dict[str, object]] = []
    loading_tables: List[pd.DataFrame] = []
    feature_series: List[pd.Series] = []

    for i, gene_id in enumerate(genes):
        if progress_every and i % progress_every == 0:
            print(f"Balance: {i}/{len(genes)}", flush=True)

        if gene_id in prepared_cache:
            prepared = prepared_cache[gene_id]
        else:
            prepared = make_effective_isoform_composition(
                gene_id,
                expression_df,
                effective_map[gene_id],
                sample_cols,
                metadata,
                pseudocount=config.clr_pseudocount,
                min_samples=config.min_samples,
                gene_row_index=gene_row_index,
            )
            prepared_cache[gene_id] = prepared

        summary, loadings, feature = fit_gene_balance(
            gene_id,
            expression_df,
            effective_map[gene_id],
            sample_cols,
            metadata,
            config=config,
            gene_row_index=gene_row_index,
            prepared=prepared,
            spline_workspace_cache=spline_workspace_cache,
        )
        summary_rows.append(summary)

        if not loadings.empty:
            loading_tables.append(loadings)
        if not feature.empty:
            feature_series.append(feature)

    balance_summary = pd.DataFrame(summary_rows)
    if not balance_summary.empty:
        balance_summary = balance_summary.sort_values(
            ["status", "gene_id"],
            kind="stable",
        ).reset_index(drop=True)

    balance_loadings = (
        pd.concat(loading_tables, ignore_index=True)
        if loading_tables else
        pd.DataFrame(columns=[
            "gene_id", "transcript_id", "pc1_loading",
            "balance_assignment", "balance_direction",
        ])
    )

    if feature_series:
        balance_feature_matrix = pd.concat(feature_series, axis=1).T
        balance_feature_matrix.index.name = "gene_id"
        balance_feature_matrix.columns.name = "sample"
    else:
        balance_feature_matrix = pd.DataFrame()
        balance_feature_matrix.index.name = "gene_id"

    return balance_summary, balance_loadings, balance_feature_matrix

# ============================================================================
# PARALLEL TEMPORAL TESTING OF THE SINGLE GENE FEATURE
# ============================================================================


# ---------------------------------------------------------------------------
# DISCRETE BRANCH: ONE UNKNOWN PURE STEP
# ---------------------------------------------------------------------------


def _valid_change_point_positions(
    ages_sorted: np.ndarray,
    *,
    min_samples_side: int,
    min_fraction_side: float,
) -> Tuple[np.ndarray, int]:
    """Return valid split positions j (left has j samples) between distinct ages."""
    ages = np.asarray(ages_sorted, dtype=float)
    n = len(ages)
    min_side = max(
        int(min_samples_side),
        int(np.ceil(float(min_fraction_side) * n)),
    )

    if n < 2 * min_side:
        return np.asarray([], dtype=int), min_side

    # Position j means ages[:j] are left and ages[j:] are right.
    positions = np.arange(1, n, dtype=int)
    distinct_boundary = ages[:-1] < ages[1:]
    enough_left = positions >= min_side
    enough_right = (n - positions) >= min_side
    valid = distinct_boundary & enough_left & enough_right
    return positions[valid], min_side


def _step_stats_for_sorted_values(
    y_sorted: np.ndarray,
    cut_positions: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """Evaluate constant-vs-step statistics at all supplied split positions."""
    y = np.asarray(y_sorted, dtype=float)
    cuts = np.asarray(cut_positions, dtype=int)
    n = len(y)

    if n < 3 or cuts.size == 0:
        raise ValueError("At least one valid cut and three samples are required.")

    csum = np.cumsum(y)
    cssq = np.cumsum(y * y)
    total_sum = float(csum[-1])
    total_ssq = float(cssq[-1])

    rss0 = max(total_ssq - total_sum * total_sum / n, 0.0)

    j = cuts.astype(float)
    sum_left = csum[cuts - 1]
    ssq_left = cssq[cuts - 1]
    sum_right = total_sum - sum_left
    ssq_right = total_ssq - ssq_left

    rss_left = np.maximum(ssq_left - (sum_left * sum_left) / j, 0.0)
    n_right = n - cuts
    rss_right = np.maximum(
        ssq_right - (sum_right * sum_right) / n_right,
        0.0,
    )
    rss1 = np.maximum(rss_left + rss_right, 0.0)
    delta_rss = np.maximum(rss0 - rss1, 0.0)

    denom = rss1 / (n - 2)
    eps = np.finfo(float).eps * max(1.0, rss0) * 32.0
    f_stats = np.empty_like(rss1, dtype=float)
    regular = denom > eps
    f_stats[regular] = delta_rss[regular] / denom[regular]
    f_stats[~regular & (delta_rss > eps)] = np.inf
    f_stats[~regular & (delta_rss <= eps)] = 0.0

    left_mean = sum_left / j
    right_mean = sum_right / n_right
    step_effect = right_mean - left_mean
    step_r2 = (
        1.0 - rss1 / rss0
        if rss0 > eps
        else np.full_like(rss1, np.nan, dtype=float)
    )

    return {
        "rss0": float(rss0),
        "rss1": rss1,
        "delta_rss": delta_rss,
        "f_stat": f_stats,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "step_effect": step_effect,
        "step_r_squared": np.asarray(step_r2, dtype=float),
    }


@lru_cache(maxsize=32)
def _cached_permutation_indices(
    n: int,
    n_permutations: int,
    random_seed: int,
) -> np.ndarray:
    """Return reproducible synchronized permutations for a given sample size."""
    # Mixing n into the seed makes cached permutations deterministic even when
    # genes with different amounts of missingness are encountered in a
    # different order across runs.
    seed = np.random.SeedSequence([int(random_seed), int(n), int(n_permutations)])
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n) for _ in range(n_permutations)]).astype(
        np.int32,
        copy=False,
    )



def _get_cached_permutation_orders(
    ages: np.ndarray,
    *,
    n_permutations: int,
    random_seed: int,
    cache: Optional[Dict[Tuple[object, ...], np.ndarray]] = None,
) -> np.ndarray:
    """Return stable sort orders for all synchronized age permutations.

    Many genes share the same sample/age vector. Computing these B age sorts
    once per unique age vector avoids repeating one of the hottest pieces of
    Python/NumPy work for every gene.
    """
    ages = np.asarray(ages, dtype=float)
    key = (
        "perm_orders",
        int(n_permutations),
        int(random_seed),
        *_array_cache_token(ages),
    )

    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    perm_idx = _cached_permutation_indices(
        len(ages),
        int(n_permutations),
        int(random_seed),
    )

    permuted_ages = ages[perm_idx]
    orders = np.argsort(
        permuted_ages,
        axis=1,
        kind="mergesort",
    ).astype(np.int32, copy=False)

    if cache is not None:
        cache[key] = orders
    return orders




def _fit_discrete_step_statistic(
    age: np.ndarray,
    y: np.ndarray,
    *,
    config: DiscreteConfig,
    sort_order: Optional[np.ndarray] = None,
    precomputed_cut_positions: Optional[np.ndarray] = None,
    precomputed_min_side: Optional[int] = None,
) -> Dict[str, object]:
    """Fit one unknown pure step and return the maximized observed statistic."""
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(age) & np.isfinite(y)
    age = age[ok]
    y = y[ok]
    n = len(y)

    base = {
        "n": int(n),
        "tested": False,
        "n_candidate_cuts": 0,
        "best_change_age": np.nan,
        "n_left": np.nan,
        "n_right": np.nan,
        "left_mean": np.nan,
        "right_mean": np.nan,
        "step_effect": np.nan,
        "abs_step_effect": np.nan,
        "rss_null": np.nan,
        "rss_step": np.nan,
        "delta_rss": np.nan,
        "step_r_squared": np.nan,
        "max_f_stat": np.nan,
        "min_samples_side_used": np.nan,
    }

    if n < 3 or np.allclose(np.std(age), 0) or np.allclose(np.std(y), 0):
        return base

    # Precomputed orders are only valid when no values were dropped by the
    # finite mask. This is always true for the validated end-to-end pipeline.
    if sort_order is not None and bool(np.all(ok)):
        order = np.asarray(sort_order, dtype=np.int64)
    else:
        order = np.argsort(age, kind="mergesort")

    age_sorted = age[order]
    y_sorted = y[order]

    if precomputed_cut_positions is not None and bool(np.all(ok)):
        cut_positions = np.asarray(precomputed_cut_positions, dtype=int)
        min_side = (
            int(precomputed_min_side)
            if precomputed_min_side is not None
            else max(
                int(config.min_samples_side),
                int(np.ceil(float(config.min_fraction_side) * n)),
            )
        )
    else:
        cut_positions, min_side = _valid_change_point_positions(
            age_sorted,
            min_samples_side=config.min_samples_side,
            min_fraction_side=config.min_fraction_side,
        )

    base["min_samples_side_used"] = int(min_side)
    base["n_candidate_cuts"] = int(len(cut_positions))

    if cut_positions.size == 0:
        return base

    stats = _step_stats_for_sorted_values(y_sorted, cut_positions)
    f_stats = np.asarray(stats["f_stat"], dtype=float)
    if f_stats.size == 0 or np.all(np.isnan(f_stats)):
        return base

    # np.argmax is deterministic; exact ties resolve to the earliest age cut.
    best_idx = int(np.nanargmax(f_stats))
    best_pos = int(cut_positions[best_idx])
    best_f = float(f_stats[best_idx])
    change_age = float((age_sorted[best_pos - 1] + age_sorted[best_pos]) / 2.0)

    base.update({
        "tested": True,
        "best_change_age": change_age,
        "n_left": best_pos,
        "n_right": n - best_pos,
        "left_mean": float(np.asarray(stats["left_mean"])[best_idx]),
        "right_mean": float(np.asarray(stats["right_mean"])[best_idx]),
        "step_effect": float(np.asarray(stats["step_effect"])[best_idx]),
        "abs_step_effect": float(abs(np.asarray(stats["step_effect"])[best_idx])),
        "rss_null": float(stats["rss0"]),
        "rss_step": float(np.asarray(stats["rss1"])[best_idx]),
        "delta_rss": float(np.asarray(stats["delta_rss"])[best_idx]),
        "step_r_squared": float(np.asarray(stats["step_r_squared"])[best_idx]),
        "max_f_stat": best_f,
    })
    return base




# ---------------------------------------------------------------------------
# CONTINUOUS / LOCALIZED BRANCHES
# ---------------------------------------------------------------------------


def _fit_nested_spline_test(
    age: np.ndarray,
    y: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
) -> Dict[str, float]:
    """Compare intercept-only versus spline(age) with an OLS nested F-test."""
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(age) & np.isfinite(y)
    age = age[ok]
    y = y[ok]
    n = len(y)

    if n < 3 or np.allclose(np.std(age), 0) or np.allclose(np.std(y), 0):
        return {
            "n": n,
            "f_stat": np.nan,
            "p_value": np.nan,
            "df_diff": np.nan,
            "df_resid_full": np.nan,
            "r_squared": np.nan,
            "fitted_robust_amp": np.nan,
        }

    X0 = np.ones((n, 1), dtype=float)
    X1 = np.asarray(
        dmatrix(
            (
                f"bs(age, df={int(spline_df)}, degree={int(spline_degree)}, "
                "include_intercept=False)"
            ),
            {"age": age},
            return_type="dataframe",
        ),
        dtype=float,
    )

    beta0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    fit0 = X0 @ beta0
    fit1 = X1 @ beta1

    rss0 = float(np.sum((y - fit0) ** 2))
    rss1 = float(np.sum((y - fit1) ** 2))
    rank0 = int(np.linalg.matrix_rank(X0))
    rank1 = int(np.linalg.matrix_rank(X1))
    df_diff = rank1 - rank0
    df_resid = n - rank1

    if df_diff <= 0 or df_resid <= 0 or rss1 <= 0:
        f_stat = np.nan
        p_value = np.nan
    else:
        numerator = max(rss0 - rss1, 0.0) / df_diff
        denominator = rss1 / df_resid
        f_stat = numerator / denominator if denominator > 0 else np.nan
        p_value = (
            float(f_dist.sf(f_stat, df_diff, df_resid))
            if np.isfinite(f_stat) else np.nan
        )

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss1 / tss if tss > 0 else np.nan

    return {
        "n": n,
        "f_stat": float(f_stat) if np.isfinite(f_stat) else np.nan,
        "p_value": p_value,
        "df_diff": float(df_diff),
        "df_resid_full": float(df_resid),
        "r_squared": float(r2) if np.isfinite(r2) else np.nan,
        "fitted_robust_amp": robust_amplitude(fit1),
    }


def _feature_correlations(age: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(age) & np.isfinite(y)
    age = age[ok]
    y = y[ok]

    if len(y) < 3 or np.allclose(np.std(age), 0) or np.allclose(np.std(y), 0):
        return {
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "abs_pearson_r": np.nan,
        }

    pr = float(pearsonr(y, age)[0])
    sr = float(spearmanr(y, age)[0])
    return {
        "pearson_r": pr,
        "spearman_r": sr,
        "abs_pearson_r": abs(pr),
    }






# ---------------------------------------------------------------------------
# SELECTION-AWARE FULL-PIPELINE PERMUTATION INFERENCE
# ---------------------------------------------------------------------------


def _selection_aware_permutation_p_value(
    observed_stat: float,
    null_stats: np.ndarray,
) -> float:
    """Monte Carlo upper-tail p-value with the standard +1 correction."""
    null_stats = np.asarray(null_stats, dtype=float)
    if np.isnan(observed_stat):
        return np.nan
    if null_stats.size == 0 or np.any(np.isnan(null_stats)):
        raise ValueError(
            "Full-pipeline permutation statistics must not contain NaN."
        )
    return float(
        (1 + np.count_nonzero(null_stats >= float(observed_stat)))
        / (len(null_stats) + 1)
    )


def _window_mask(
    ages: np.ndarray,
    lower: float,
    upper: float,
    *,
    include_upper: bool,
) -> np.ndarray:
    """Return the deterministic sample mask for one configured age interval."""
    ages = np.asarray(ages, dtype=float)
    if include_upper:
        return (ages >= float(lower)) & (ages <= float(upper))
    return (ages >= float(lower)) & (ages < float(upper))


def _window_sorted_positions(
    ages_sorted: np.ndarray,
    lower: float,
    upper: float,
    *,
    include_upper: bool,
) -> np.ndarray:
    """Return positions for one interval in an already age-sorted vector."""
    ages_sorted = np.asarray(ages_sorted, dtype=float)
    left = int(np.searchsorted(ages_sorted, float(lower), side="left"))
    right = int(
        np.searchsorted(
            ages_sorted,
            float(upper),
            side="right" if include_upper else "left",
        )
    )
    return np.arange(left, right, dtype=np.int64)


def _annotate_discrete_no_inference(
    obs: Dict[str, object],
    *,
    n_permutations: int,
    inference_method: str,
) -> None:
    obs["selection_aware_p_value"] = np.nan
    obs["permutation_p_value"] = np.nan
    obs["p_value"] = np.nan
    obs["n_permutations"] = int(n_permutations)
    obs["inference_method"] = inference_method


def _annotate_continuous_no_inference(
    obs: Dict[str, object],
    *,
    n_permutations: int,
    inference_method: str,
) -> None:
    obs["selection_aware_p_value"] = np.nan
    obs["p_value"] = np.nan
    obs["n_permutations"] = int(n_permutations)
    obs["inference_method"] = inference_method


def _run_selection_aware_gene_tests(
    gene_id: str,
    prepared: Dict[str, object],
    observed_feature: np.ndarray,
    *,
    branches: Sequence[str],
    balance_config: BalanceConfig,
    discrete_config: DiscreteConfig,
    continuous_config: ContinuousConfig,
    windows: Optional[WindowConfig],
    inference_config: PermutationInferenceConfig,
    spline_workspace_cache: Optional[
        Dict[Tuple[object, ...], Dict[str, object]]
    ],
    permutation_orders: np.ndarray,
) -> Dict[str, object]:
    """Run requested global and optional windowed tests for one gene.

    ``localized`` is only an API flag for windowed analysis. Within every
    configured interval the exact same discrete and continuous hypotheses are
    evaluated as in the global analysis.
    """
    branches = normalize_branches(branches)
    gene_id = str(gene_id)

    if "localized" in branches and windows is None:
        raise ValueError(
            "Localized/windowed analysis requires explicit config.windows. "
            "Example: WindowConfig(edges=(20, 40, 60, 85))."
        )

    ages = np.asarray(prepared["ages"], dtype=float)
    observed_feature = np.asarray(observed_feature, dtype=float)
    if len(observed_feature) != len(ages):
        raise ValueError(
            f"Observed feature for {gene_id} does not align with prepared samples."
        )

    result: Dict[str, object] = {
        "localized_discrete": [],
        "localized_continuous": [],
    }

    # ------------------------------------------------------------------
    # Observed global statistics
    # ------------------------------------------------------------------
    if "discrete" in branches:
        discrete_obs = _fit_discrete_step_statistic(
            ages,
            observed_feature,
            config=discrete_config,
        )
        discrete_obs["gene_id"] = gene_id
        result["discrete"] = discrete_obs

    if "continuous" in branches:
        continuous_obs = _fit_nested_spline_test(
            ages,
            observed_feature,
            spline_df=continuous_config.spline_df,
            spline_degree=continuous_config.spline_degree,
        )
        continuous_obs.update(_feature_correlations(ages, observed_feature))
        continuous_obs["gene_id"] = gene_id
        continuous_obs["tested"] = bool(
            continuous_obs["n"] >= continuous_config.min_samples
            and not np.isnan(continuous_obs["f_stat"])
        )
        continuous_obs["analytic_p_value"] = continuous_obs["p_value"]
        if not continuous_obs["tested"]:
            continuous_obs["p_value"] = np.nan
        result["continuous"] = continuous_obs

    # ------------------------------------------------------------------
    # Observed windowed statistics: same two hypotheses within each interval
    # ------------------------------------------------------------------
    window_specs: List[Dict[str, object]] = []
    if "localized" in branches:
        assert windows is not None
        n_windows = len(windows.intervals)

        for window_index, ((lower, upper), label) in enumerate(
            zip(windows.intervals, windows.labels)
        ):
            include_upper = window_index == n_windows - 1
            mask = _window_mask(
                ages,
                lower,
                upper,
                include_upper=include_upper,
            )
            ages_w = ages[mask]
            feature_w = observed_feature[mask]

            discrete_obs = _fit_discrete_step_statistic(
                ages_w,
                feature_w,
                config=discrete_config,
            )
            discrete_obs.update({
                "gene_id": gene_id,
                "window": label,
                "window_index": int(window_index),
                "window_start": float(lower),
                "window_end": float(upper),
            })
            result["localized_discrete"].append(discrete_obs)

            continuous_obs = _fit_nested_spline_test(
                ages_w,
                feature_w,
                spline_df=continuous_config.spline_df,
                spline_degree=continuous_config.spline_degree,
            )
            continuous_obs.update(_feature_correlations(ages_w, feature_w))
            continuous_obs.update({
                "gene_id": gene_id,
                "window": label,
                "window_index": int(window_index),
                "window_start": float(lower),
                "window_end": float(upper),
            })
            continuous_obs["tested"] = bool(
                continuous_obs["n"] >= continuous_config.min_samples
                and not np.isnan(continuous_obs["f_stat"])
            )
            continuous_obs["analytic_p_value"] = continuous_obs["p_value"]
            if not continuous_obs["tested"]:
                continuous_obs["p_value"] = np.nan
            result["localized_continuous"].append(continuous_obs)

            window_specs.append({
                "window_index": int(window_index),
                "lower": float(lower),
                "upper": float(upper),
                "include_upper": bool(include_upper),
            })

    B = int(inference_config.n_permutations)
    global_inference_label = "full_pipeline_age_permutation"
    window_discrete_label = "full_pipeline_age_permutation_windowed_discrete"
    window_continuous_label = "full_pipeline_age_permutation_windowed_continuous"

    any_tested = False
    if "discrete" in branches:
        any_tested |= bool(result["discrete"]["tested"])
    if "continuous" in branches:
        any_tested |= bool(result["continuous"]["tested"])
    if "localized" in branches:
        any_tested |= any(
            bool(row["tested"]) for row in result["localized_discrete"]
        )
        any_tested |= any(
            bool(row["tested"]) for row in result["localized_continuous"]
        )

    if not any_tested:
        if "discrete" in branches:
            _annotate_discrete_no_inference(
                result["discrete"],
                n_permutations=B,
                inference_method=global_inference_label,
            )
        if "continuous" in branches:
            _annotate_continuous_no_inference(
                result["continuous"],
                n_permutations=B,
                inference_method=global_inference_label,
            )
        if "localized" in branches:
            for row in result["localized_discrete"]:
                _annotate_discrete_no_inference(
                    row,
                    n_permutations=B,
                    inference_method=window_discrete_label,
                )
            for row in result["localized_continuous"]:
                _annotate_continuous_no_inference(
                    row,
                    n_permutations=B,
                    inference_method=window_continuous_label,
                )
        return result

    # ------------------------------------------------------------------
    # Age-only reusable objects
    # ------------------------------------------------------------------
    reference_order = np.argsort(ages, kind="mergesort")
    reference_ages_sorted = ages[reference_order]

    continuous_workspace = None
    if "continuous" in branches and bool(result["continuous"]["tested"]):
        continuous_workspace = _get_cached_fixed_age_spline_workspace(
            ages,
            spline_df=continuous_config.spline_df,
            spline_degree=continuous_config.spline_degree,
            cache=spline_workspace_cache,
        )

    discrete_cut_positions = None
    if "discrete" in branches and bool(result["discrete"]["tested"]):
        discrete_cut_positions, _ = _valid_change_point_positions(
            reference_ages_sorted,
            min_samples_side=discrete_config.min_samples_side,
            min_fraction_side=discrete_config.min_fraction_side,
        )

    localized_positions: Dict[int, np.ndarray] = {}
    localized_discrete_cuts: Dict[int, np.ndarray] = {}
    localized_continuous_workspaces: Dict[int, Dict[str, object]] = {}

    if "localized" in branches:
        for spec in window_specs:
            i = int(spec["window_index"])
            positions = _window_sorted_positions(
                reference_ages_sorted,
                float(spec["lower"]),
                float(spec["upper"]),
                include_upper=bool(spec["include_upper"]),
            )
            localized_positions[i] = positions
            ages_w_sorted = reference_ages_sorted[positions]

            if bool(result["localized_discrete"][i]["tested"]):
                cuts, _ = _valid_change_point_positions(
                    ages_w_sorted,
                    min_samples_side=discrete_config.min_samples_side,
                    min_fraction_side=discrete_config.min_fraction_side,
                )
                localized_discrete_cuts[i] = cuts

            if bool(result["localized_continuous"][i]["tested"]):
                localized_continuous_workspaces[i] = (
                    _get_cached_fixed_age_spline_workspace(
                        ages_w_sorted,
                        spline_df=continuous_config.spline_df,
                        spline_degree=continuous_config.spline_degree,
                        cache=spline_workspace_cache,
                    )
                )

    # ------------------------------------------------------------------
    # Balance relearning setup
    # ------------------------------------------------------------------
    k = len(prepared["isoforms"])
    clr_values = None
    log_composition = None
    balance_workspace = None

    if k > 2:
        composition = np.asarray(prepared["composition"], dtype=float)
        log_composition = np.log(composition)
        clr_values = log_composition - log_composition.mean(
            axis=1,
            keepdims=True,
        )
        balance_workspace = _get_cached_fixed_age_spline_workspace(
            ages,
            spline_df=balance_config.spline_df,
            spline_degree=balance_config.spline_degree,
            age_grid_size=balance_config.age_grid_size,
            cache=spline_workspace_cache,
        )

    batch_size = min(int(inference_config.batch_size), B)
    permutation_orders = np.asarray(permutation_orders, dtype=np.int32)
    if permutation_orders.shape != (B, len(ages)):
        raise ValueError(
            "Precomputed permutation_orders have the wrong shape for "
            f"{gene_id}: expected {(B, len(ages))}, "
            f"got {permutation_orders.shape}."
        )

    discrete_null = (
        np.zeros(B, dtype=float)
        if "discrete" in branches and bool(result["discrete"]["tested"])
        else None
    )
    continuous_null = (
        np.zeros(B, dtype=float)
        if "continuous" in branches and bool(result["continuous"]["tested"])
        else None
    )
    localized_discrete_null = {
        i: np.zeros(B, dtype=float)
        for i in localized_discrete_cuts
    }
    localized_continuous_null = {
        i: np.zeros(B, dtype=float)
        for i in localized_continuous_workspaces
    }

    # ------------------------------------------------------------------
    # Batched full-pipeline permutations
    # ------------------------------------------------------------------
    for start in range(0, B, batch_size):
        stop = min(start + batch_size, B)
        orders = permutation_orders[start:stop]

        # K=2 has one balance up to sign. All test statistics are sign-invariant,
        # so explicit balance relearning is exactly unnecessary in this case.
        if k == 2:
            sorted_features = observed_feature[orders]
        else:
            permuted_features = _learn_permuted_balance_features_batch(
                clr_values,
                orders,
                config=balance_config,
                spline_workspace=balance_workspace,
                log_composition=log_composition,
            )
            sorted_features = np.take_along_axis(
                permuted_features,
                orders,
                axis=1,
            )

        if discrete_null is not None:
            stats = _max_step_f_statistics_from_sorted_matrix(
                sorted_features,
                discrete_cut_positions,
            )
            if np.any(np.isnan(stats)):
                raise RuntimeError(
                    f"Discrete permutation became untestable for {gene_id}."
                )
            discrete_null[start:stop] = stats

        if continuous_null is not None:
            stats = _spline_f_statistics_from_sorted_matrix(
                sorted_features,
                continuous_workspace,
            )
            if np.any(np.isnan(stats)):
                raise RuntimeError(
                    f"Continuous permutation became untestable for {gene_id}."
                )
            continuous_null[start:stop] = stats

        for i, null in localized_discrete_null.items():
            stats = _max_step_f_statistics_from_sorted_matrix(
                sorted_features[:, localized_positions[i]],
                localized_discrete_cuts[i],
            )
            if np.any(np.isnan(stats)):
                label = result["localized_discrete"][i]["window"]
                raise RuntimeError(
                    f"Windowed discrete permutation became untestable for "
                    f"{gene_id}, window {label}."
                )
            null[start:stop] = stats

        for i, null in localized_continuous_null.items():
            stats = _spline_f_statistics_from_sorted_matrix(
                sorted_features[:, localized_positions[i]],
                localized_continuous_workspaces[i],
            )
            if np.any(np.isnan(stats)):
                label = result["localized_continuous"][i]["window"]
                raise RuntimeError(
                    f"Windowed continuous permutation became untestable for "
                    f"{gene_id}, window {label}."
                )
            null[start:stop] = stats

    # ------------------------------------------------------------------
    # Monte Carlo p-values
    # ------------------------------------------------------------------
    if "discrete" in branches:
        obs = result["discrete"]
        p = (
            _selection_aware_permutation_p_value(
                float(obs["max_f_stat"]),
                discrete_null,
            )
            if discrete_null is not None
            else np.nan
        )
        obs["selection_aware_p_value"] = p
        obs["permutation_p_value"] = p
        obs["p_value"] = p
        obs["n_permutations"] = B
        obs["inference_method"] = global_inference_label

    if "continuous" in branches:
        obs = result["continuous"]
        p = (
            _selection_aware_permutation_p_value(
                float(obs["f_stat"]),
                continuous_null,
            )
            if continuous_null is not None
            else np.nan
        )
        obs["selection_aware_p_value"] = p
        obs["p_value"] = p
        obs["n_permutations"] = B
        obs["inference_method"] = global_inference_label

    if "localized" in branches:
        for i, obs in enumerate(result["localized_discrete"]):
            null = localized_discrete_null.get(i)
            p = (
                _selection_aware_permutation_p_value(
                    float(obs["max_f_stat"]),
                    null,
                )
                if null is not None
                else np.nan
            )
            obs["selection_aware_p_value"] = p
            obs["permutation_p_value"] = p
            obs["p_value"] = p
            obs["n_permutations"] = B
            obs["inference_method"] = window_discrete_label

        for i, obs in enumerate(result["localized_continuous"]):
            null = localized_continuous_null.get(i)
            p = (
                _selection_aware_permutation_p_value(
                    float(obs["f_stat"]),
                    null,
                )
                if null is not None
                else np.nan
            )
            obs["selection_aware_p_value"] = p
            obs["p_value"] = p
            obs["n_permutations"] = B
            obs["inference_method"] = window_continuous_label

    return result


def _warn_permutation_resolution(
    analysis_name: str,
    family_size: int,
    *,
    alpha: float,
    n_permutations: int,
) -> None:
    """Warn when Monte Carlo resolution is coarse relative to the first BH cutoff."""
    if family_size <= 0:
        return
    min_possible_p = 1.0 / (int(n_permutations) + 1.0)
    first_bh_threshold = float(alpha) / int(family_size)
    if min_possible_p > first_bh_threshold:
        warnings.warn(
            f"{analysis_name} permutation resolution may be coarse for BH: "
            f"minimum p={min_possible_p:.3g}, first BH threshold="
            f"{first_bh_threshold:.3g}. Increase inference.n_permutations "
            "when precise tail resolution is required.",
            RuntimeWarning,
            stacklevel=3,
        )


def _finalize_discrete_table(
    table: pd.DataFrame,
    *,
    config: DiscreteConfig,
    inference_config: PermutationInferenceConfig,
    windowed: bool,
) -> pd.DataFrame:
    if table.empty:
        return table

    tested = table["tested"].fillna(False).astype(bool)
    _warn_permutation_resolution(
        "Windowed discrete" if windowed else "Discrete",
        int(tested.sum()),
        alpha=config.alpha_fdr,
        n_permutations=inference_config.n_permutations,
    )
    table = add_bh_fdr(
        table,
        "p_value",
        alpha=config.alpha_fdr,
        valid_mask=tested,
        q_col="q_value",
        reject_col="fdr_significant",
    )

    if windowed:
        sort_cols = [
            "q_value", "p_value", "abs_step_effect",
            "window_index", "gene_id",
        ]
        ascending = [True, True, False, True, True]
    else:
        sort_cols = ["q_value", "p_value", "abs_step_effect", "gene_id"]
        ascending = [True, True, False, True]

    return table.sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)


def _finalize_continuous_table(
    table: pd.DataFrame,
    *,
    config: ContinuousConfig,
    inference_config: PermutationInferenceConfig,
    windowed: bool,
) -> pd.DataFrame:
    if table.empty:
        return table

    tested = table["tested"].fillna(False).astype(bool)
    _warn_permutation_resolution(
        "Windowed continuous" if windowed else "Continuous",
        int(tested.sum()),
        alpha=config.alpha_fdr,
        n_permutations=inference_config.n_permutations,
    )
    table = add_bh_fdr(
        table,
        "p_value",
        alpha=config.alpha_fdr,
        valid_mask=tested,
        q_col="q_value",
        reject_col="fdr_significant",
    )
    table["score_predictive_hybrid"] = (
        table["abs_pearson_r"].fillna(0.0)
        * table["fitted_robust_amp"].fillna(0.0)
    )

    if windowed:
        sort_cols = [
            "q_value", "p_value", "score_predictive_hybrid",
            "window_index", "gene_id",
        ]
        ascending = [True, True, False, True, True]
    else:
        sort_cols = [
            "q_value", "p_value", "score_predictive_hybrid", "gene_id",
        ]
        ascending = [True, True, False, True]

    return table.sort_values(
        sort_cols,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)


def run_selection_aware_temporal_tests(
    observed_feature_matrix: pd.DataFrame,
    prepared_cache: Dict[str, Optional[Dict[str, object]]],
    *,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
    balance_config: BalanceConfig = BalanceConfig(),
    discrete_config: DiscreteConfig = DiscreteConfig(),
    continuous_config: ContinuousConfig = ContinuousConfig(),
    windows: Optional[WindowConfig] = None,
    inference_config: PermutationInferenceConfig = PermutationInferenceConfig(),
    progress_every: Optional[int] = None,
    n_jobs: int = 1,
    spline_workspace_cache: Optional[
        Dict[Tuple[object, ...], Dict[str, object]]
    ] = None,
    permutation_order_cache: Optional[
        Dict[Tuple[object, ...], np.ndarray]
    ] = None,
) -> Dict[str, pd.DataFrame]:
    """Run production and optional windowed selection-aware inference."""
    branches = normalize_branches(branches)
    if "localized" in branches and windows is None:
        raise ValueError(
            "Localized/windowed analysis requires explicit windows. Set "
            "config.windows=WindowConfig(edges=(...))."
        )

    if spline_workspace_cache is None:
        spline_workspace_cache = {}
    if permutation_order_cache is None:
        permutation_order_cache = {}

    genes = [
        str(gene)
        for gene in observed_feature_matrix.index
        if str(gene) in prepared_cache and prepared_cache[str(gene)] is not None
    ]

    rows: Dict[str, List[Dict[str, object]]] = {
        "discrete": [],
        "continuous": [],
        "localized_discrete": [],
        "localized_continuous": [],
    }

    payloads: List[
        Tuple[str, Dict[str, object], np.ndarray, np.ndarray]
    ] = []
    for gene_id in genes:
        prepared = prepared_cache[gene_id]
        if prepared is None:
            continue

        samples = np.asarray(prepared["samples"], dtype=str)
        feature_row = observed_feature_matrix.loc[gene_id]
        missing = [sample for sample in samples if sample not in feature_row.index]
        if missing:
            raise ValueError(
                f"Observed feature matrix is missing samples for {gene_id}: "
                f"{missing[:5]}"
            )
        observed_feature = feature_row.loc[samples].to_numpy(dtype=float)

        permutation_orders = _get_cached_permutation_orders(
            np.asarray(prepared["ages"], dtype=float),
            n_permutations=int(inference_config.n_permutations),
            random_seed=int(inference_config.random_seed),
            cache=permutation_order_cache,
        )
        payloads.append(
            (gene_id, prepared, observed_feature, permutation_orders)
        )

    def _worker(payload):
        gene_id, prepared, observed_feature, permutation_orders = payload
        return _run_selection_aware_gene_tests(
            gene_id,
            prepared,
            observed_feature,
            branches=branches,
            balance_config=balance_config,
            discrete_config=discrete_config,
            continuous_config=continuous_config,
            windows=windows,
            inference_config=inference_config,
            spline_workspace_cache=spline_workspace_cache,
            permutation_orders=permutation_orders,
        )

    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    resolved_n_jobs = (
        max(1, int(os.cpu_count() or 1))
        if int(n_jobs) == -1
        else int(n_jobs)
    )

    if resolved_n_jobs == 1:
        result_iter = map(_worker, payloads)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=resolved_n_jobs)
        result_iter = executor.map(_worker, payloads)

    try:
        for i, gene_result in enumerate(result_iter):
            if progress_every and i % progress_every == 0:
                print(
                    f"Selection-aware inference: {i}/{len(payloads)}",
                    flush=True,
                )

            if "discrete" in branches:
                rows["discrete"].append(dict(gene_result["discrete"]))
            if "continuous" in branches:
                rows["continuous"].append(dict(gene_result["continuous"]))
            if "localized" in branches:
                rows["localized_discrete"].extend(
                    dict(row) for row in gene_result["localized_discrete"]
                )
                rows["localized_continuous"].extend(
                    dict(row) for row in gene_result["localized_continuous"]
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    out = {name: pd.DataFrame(values) for name, values in rows.items()}

    if "discrete" in branches:
        out["discrete"] = _finalize_discrete_table(
            out["discrete"],
            config=discrete_config,
            inference_config=inference_config,
            windowed=False,
        )
    if "continuous" in branches:
        out["continuous"] = _finalize_continuous_table(
            out["continuous"],
            config=continuous_config,
            inference_config=inference_config,
            windowed=False,
        )
    if "localized" in branches:
        # FDR families are detector-specific across every tested gene x window.
        out["localized_discrete"] = _finalize_discrete_table(
            out["localized_discrete"],
            config=discrete_config,
            inference_config=inference_config,
            windowed=True,
        )
        out["localized_continuous"] = _finalize_continuous_table(
            out["localized_continuous"],
            config=continuous_config,
            inference_config=inference_config,
            windowed=True,
        )

    return out


# ---------------------------------------------------------------------------
# MERGE PRODUCTION BRANCHES
# ---------------------------------------------------------------------------


def _production_branches(branches: Sequence[str]) -> Tuple[str, ...]:
    requested = normalize_branches(branches)
    return tuple(branch for branch in PRODUCTION_BRANCHES if branch in requested)


def _trajectory_signature(row: pd.Series, branches: Sequence[str]) -> str:
    labels = [
        branch
        for branch in _production_branches(branches)
        if bool(row.get(f"{branch}_fdr_significant", False))
    ]
    return "+".join(labels) if labels else "not_selected"


def summarize_gene_trajectories(
    balance_summary: pd.DataFrame,
    discrete_results: pd.DataFrame,
    continuous_results: pd.DataFrame,
    *,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
) -> pd.DataFrame:
    """Merge global evidence from selected production branches only."""
    production_branches = _production_branches(branches)
    ok_balance = balance_summary.loc[balance_summary["status"].eq("ok")].copy()
    if ok_balance.empty:
        return ok_balance

    summary = ok_balance.copy()

    if "discrete" in production_branches and not discrete_results.empty:
        discrete_keep = discrete_results.rename(columns={
            "p_value": "discrete_p_value",
            "q_value": "discrete_q_value",
            "fdr_significant": "discrete_fdr_significant",
            "max_f_stat": "discrete_max_f_stat",
            "step_r_squared": "discrete_step_r_squared",
            "step_effect": "discrete_step_effect",
            "abs_step_effect": "discrete_abs_step_effect",
            "best_change_age": "discrete_change_age",
            "left_mean": "discrete_left_mean",
            "right_mean": "discrete_right_mean",
            "n_left": "discrete_n_left",
            "n_right": "discrete_n_right",
            "n_candidate_cuts": "discrete_n_candidate_cuts",
        })
        keep_cols = [
            "gene_id", "discrete_p_value", "discrete_q_value",
            "discrete_fdr_significant", "discrete_max_f_stat",
            "discrete_step_r_squared", "discrete_step_effect",
            "discrete_abs_step_effect", "discrete_change_age",
            "discrete_left_mean", "discrete_right_mean",
            "discrete_n_left", "discrete_n_right",
            "discrete_n_candidate_cuts",
        ]
        summary = summary.merge(
            discrete_keep[[c for c in keep_cols if c in discrete_keep.columns]],
            on="gene_id",
            how="left",
        )

    if "continuous" in production_branches and not continuous_results.empty:
        continuous_keep = continuous_results.rename(columns={
            "p_value": "continuous_p_value",
            "q_value": "continuous_q_value",
            "fdr_significant": "continuous_fdr_significant",
            "f_stat": "continuous_f_stat",
            "r_squared": "continuous_r_squared",
            "fitted_robust_amp": "continuous_fitted_robust_amp",
            "pearson_r": "continuous_pearson_r",
            "spearman_r": "continuous_spearman_r",
            "abs_pearson_r": "continuous_abs_pearson_r",
        })
        keep_cols = [
            "gene_id", "continuous_p_value", "continuous_q_value",
            "continuous_fdr_significant", "continuous_f_stat",
            "continuous_r_squared", "continuous_fitted_robust_amp",
            "continuous_pearson_r", "continuous_spearman_r",
            "continuous_abs_pearson_r", "score_predictive_hybrid",
        ]
        summary = summary.merge(
            continuous_keep[[c for c in keep_cols if c in continuous_keep.columns]],
            on="gene_id",
            how="left",
        )

    for branch in PRODUCTION_BRANCHES:
        col = f"{branch}_fdr_significant"
        if col not in summary.columns:
            summary[col] = False
        summary[col] = summary[col].fillna(False).astype(bool)

    if production_branches:
        sig_cols = [f"{branch}_fdr_significant" for branch in production_branches]
        summary["n_detection_branches"] = summary[sig_cols].astype(int).sum(axis=1)
    else:
        summary["n_detection_branches"] = 0

    summary["trajectory_signature"] = summary.apply(
        lambda row: _trajectory_signature(row, production_branches),
        axis=1,
    )

    q_col_map = {
        "discrete": "discrete_q_value",
        "continuous": "continuous_q_value",
    }
    q_cols = [
        q_col_map[b]
        for b in production_branches
        if q_col_map[b] in summary.columns
    ]
    summary["min_branch_q_value"] = (
        summary[q_cols].min(axis=1, skipna=True) if q_cols else np.nan
    )

    return summary.sort_values(
        ["n_detection_branches", "min_branch_q_value", "gene_id"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def select_final_gene_features(
    gene_summary: pd.DataFrame,
    *,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
    config: SelectionConfig = SelectionConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return production-branch candidates and their unique gene-level union."""
    production_branches = _production_branches(branches)
    if gene_summary.empty:
        empty = gene_summary.copy()
        return empty, empty, empty

    candidates: Dict[str, pd.DataFrame] = {}
    for branch in PRODUCTION_BRANCHES:
        sig_col = f"{branch}_fdr_significant"
        if branch in production_branches and sig_col in gene_summary.columns:
            candidates[branch] = gene_summary.loc[
                gene_summary[sig_col].fillna(False)
            ].copy()
        else:
            candidates[branch] = gene_summary.iloc[0:0].copy()

    if production_branches:
        sig_cols = [f"{branch}_fdr_significant" for branch in production_branches]
        selected_mask = gene_summary[sig_cols].fillna(False).any(axis=1)
    else:
        selected_mask = pd.Series(False, index=gene_summary.index)

    selected = gene_summary.loc[selected_mask].copy()
    selected = selected.sort_values(
        ["n_detection_branches", "min_branch_q_value", "gene_id"],
        ascending=[False, True, True],
        na_position="last",
    )

    if config.top_n_genes is not None:
        selected = selected.head(int(config.top_n_genes)).copy()

    selected.insert(0, "rank_gene", np.arange(1, len(selected) + 1))

    return (
        candidates["discrete"].reset_index(drop=True),
        candidates["continuous"].reset_index(drop=True),
        selected.reset_index(drop=True),
    )


def _significant_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return FDR-significant rows from an already finalized result table."""
    if table.empty or "fdr_significant" not in table.columns:
        return table.iloc[0:0].copy()
    return table.loc[table["fdr_significant"].fillna(False)].reset_index(drop=True)


# ============================================================================
# END-TO-END ORCHESTRATOR
# ============================================================================


def run_isovita_pipeline(
    *,
    tpm: pd.DataFrame,
    metadata: pd.DataFrame,
    branches: Sequence[str] = PRODUCTION_BRANCHES,
    config: IsoVitaConfig = IsoVitaConfig(),
    progress_every: Optional[int] = None,
    n_jobs: int = 1,
) -> IsoVitaPipelineResult:
    """Run IsoVita using the requested analysis modes.

    Parameters
    ----------
    tpm
        Standardized transcript-level TPM table with ``gene_id``,
        ``transcript_id``, then one column per retained sample.
    metadata
        Standardized sample metadata with columns ``sample_id`` and ``age``.
        It must already reflect all dataset/cohort filtering.
    branches
        Defaults to the production workflow ``("discrete", "continuous")``.
        ``"localized"`` may be explicitly added for optional windowed analyses,
        but it never contributes to ``final_features``.
    config
        Top-level IsoVita configuration. ``config.windows`` must be explicitly
        supplied whenever ``"localized"`` is requested.
    n_jobs
        Number of shared-memory worker threads for selection-aware gene-level
        permutation inference. Default 1 preserves serial behavior. Use -1 for
        all available logical CPUs.
    """
    selected_branches = normalize_branches(branches)
    if "localized" in selected_branches and config.windows is None:
        raise ValueError(
            "Localized/windowed analysis requires explicit user-defined windows. "
            "Set config.windows=WindowConfig(edges=(...))."
        )

    full_tpm, meta, sample_cols = validate_core_inputs(tpm, metadata)

    isoplex_expression_df, isoplex_filter_summary = filter_tpm_for_isoplex(
        full_tpm,
        sample_cols,
        config=config.isoplex,
    )
    sample_metrics, global_metrics = run_isoplex(isoplex_expression_df)
    gene_routes = route_genes_by_effective_isoform_count(global_metrics)

    # Run-local acceleration structures. They do not alter statistical results.
    gene_row_index = _build_gene_row_index(full_tpm)
    prepared_cache: Dict[str, Optional[Dict[str, object]]] = {}
    spline_workspace_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}

    # IsoPlex determines effective-isoform membership. Feature values are then
    # computed from the original prepared TPM values for those effective isoforms.
    balance_summary, balance_loadings, balance_feature_matrix = (
        build_one_feature_per_gene(
            expression_df=full_tpm,
            global_metrics=global_metrics,
            sample_cols=sample_cols,
            metadata=meta,
            config=config.balance,
            progress_every=progress_every,
            gene_row_index=gene_row_index,
            prepared_cache=prepared_cache,
            spline_workspace_cache=spline_workspace_cache,
        )
    )

    branch_results = run_selection_aware_temporal_tests(
        observed_feature_matrix=balance_feature_matrix,
        prepared_cache=prepared_cache,
        branches=selected_branches,
        balance_config=config.balance,
        discrete_config=config.discrete,
        continuous_config=config.continuous,
        windows=config.windows,
        inference_config=config.inference,
        progress_every=progress_every,
        n_jobs=n_jobs,
        spline_workspace_cache=spline_workspace_cache,
    )

    discrete_results = branch_results["discrete"]
    continuous_results = branch_results["continuous"]
    localized_discrete_results = branch_results["localized_discrete"]
    localized_continuous_results = branch_results["localized_continuous"]

    # Production summary/final selection intentionally ignores windowed results.
    gene_summary = summarize_gene_trajectories(
        balance_summary,
        discrete_results,
        continuous_results,
        branches=selected_branches,
    )

    (
        discrete_candidates,
        continuous_candidates,
        final_features,
    ) = select_final_gene_features(
        gene_summary,
        branches=selected_branches,
        config=config.selection,
    )

    localized_discrete_candidates = _significant_rows(localized_discrete_results)
    localized_continuous_candidates = _significant_rows(localized_continuous_results)

    return IsoVitaPipelineResult(
        selected_branches=selected_branches,
        metadata=meta,
        tpm=full_tpm,
        isoplex_expression_df=isoplex_expression_df,
        isoplex_filter_summary=isoplex_filter_summary,
        sample_metrics=sample_metrics,
        global_metrics=global_metrics,
        gene_routes=gene_routes,
        balance_summary=balance_summary,
        balance_loadings=balance_loadings,
        balance_feature_matrix=balance_feature_matrix,
        discrete_trajectory_results=discrete_results,
        continuous_trajectory_results=continuous_results,
        localized_discrete_results=localized_discrete_results,
        localized_continuous_results=localized_continuous_results,
        gene_trajectory_summary=gene_summary,
        discrete_trajectory_candidates=discrete_candidates,
        continuous_trajectory_candidates=continuous_candidates,
        localized_discrete_candidates=localized_discrete_candidates,
        localized_continuous_candidates=localized_continuous_candidates,
        final_features=final_features,
    )


# ============================================================================
# OUTPUT HELPERS
# ============================================================================


def save_isovita_pipeline_outputs(
    result: IsoVitaPipelineResult,
    output_dir: str | Path,
) -> None:
    """Write shared outputs plus only the analyses that were requested."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "isoplex_filter_summary.csv": result.isoplex_filter_summary,
        "isoplex_sample_metrics.csv": result.sample_metrics,
        "isoplex_global_metrics.csv": result.global_metrics,
        "gene_routes.csv": result.gene_routes,
        "balance_summary.csv": result.balance_summary,
        "balance_loadings.csv": result.balance_loadings,
        "balance_feature_matrix.csv": result.balance_feature_matrix,
        "gene_trajectory_summary.csv": result.gene_trajectory_summary,
        "final_features.csv": result.final_features,
    }

    if "discrete" in result.selected_branches:
        tables["discrete_trajectory_results.csv"] = result.discrete_trajectory_results
        tables["discrete_trajectory_candidates.csv"] = (
            result.discrete_trajectory_candidates
        )

    if "continuous" in result.selected_branches:
        tables["continuous_trajectory_results.csv"] = (
            result.continuous_trajectory_results
        )
        tables["continuous_trajectory_candidates.csv"] = (
            result.continuous_trajectory_candidates
        )

    if "localized" in result.selected_branches:
        tables["localized_discrete_results.csv"] = result.localized_discrete_results
        tables["localized_continuous_results.csv"] = (
            result.localized_continuous_results
        )
        tables["localized_discrete_candidates.csv"] = (
            result.localized_discrete_candidates
        )
        tables["localized_continuous_candidates.csv"] = (
            result.localized_continuous_candidates
        )

    for filename, table in tables.items():
        table.to_csv(
            output_dir / filename,
            index=(filename == "balance_feature_matrix.csv"),
        )

    (output_dir / "selected_branches.txt").write_text(
        "\n".join(result.selected_branches) + "\n"
    )
