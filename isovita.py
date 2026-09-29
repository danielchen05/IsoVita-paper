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

Core algorithm
--------------
1. Apply the configured TPM eligibility filter used for IsoPlex input.
2. Run IsoPlex and identify effective isoforms.
3. For every gene with >=2 effective isoforms, build one age-informed feature:
   effective-isoform composition -> CLR -> smooth trajectories -> PC1 ->
   simplest near-optimal A/B balance -> log(GM(A) / GM(B)).
4. Run only the temporal detection branches requested by the user:
   ``discrete``, ``continuous``, and/or ``localized``.
5. Return branch-specific results and a final gene set equal to the union of
   FDR-significant genes from the selected branches. With one selected branch,
   the final set is exactly that branch's selected genes. When multiple branches
   are selected, this union is a downstream combination of branch-wise calls; it
   is not itself a separately FDR-calibrated omnibus family.

The branches are parallel detectors, not mutually exclusive trajectory classes.
IsoVita has no SPIT dependency.

Inferential note
----------------
The balance is learned using age, so final branch p-values are obtained by
permuting age and repeating the complete age-dependent feature-learning step
(CLR smooth trajectories -> PC1 -> sparse balance) before recomputing each
branch statistic. This full-pipeline permutation accounts for adaptive balance
selection. IsoPlex/effective-isoform membership is held fixed across
permutations because it depends only on expression, not age.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path
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


def _default_window_edges() -> Tuple[float, ...]:
    return (0.0, 10.0, 20.0, 40.0, 85.0)


def _format_edge(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


@dataclass(frozen=True)
class WindowConfig:
    """Age-window specification used by the localized branch."""

    edges: Tuple[float, ...] = field(default_factory=_default_window_edges)

    def __post_init__(self) -> None:
        if len(self.edges) < 3:
            raise ValueError("At least two age windows are required.")
        if any(b <= a for a, b in zip(self.edges[:-1], self.edges[1:])):
            raise ValueError("Window edges must be strictly increasing.")

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(
            f"{_format_edge(a)}-{_format_edge(b)}"
            for a, b in zip(self.edges[:-1], self.edges[1:])
        )


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
    cache_candidate_directions_through_k: int = 10

    def __post_init__(self) -> None:
        if self.spline_df < self.spline_degree:
            raise ValueError("spline_df must be >= spline_degree.")
        if self.age_grid_size < 10:
            raise ValueError("age_grid_size must be at least 10.")
        if self.clr_pseudocount <= 0:
            raise ValueError("clr_pseudocount must be > 0.")
        if not (0 <= self.near_optimal_cosine_tolerance < 1):
            raise ValueError(
                "near_optimal_cosine_tolerance must lie in [0, 1)."
            )


@dataclass(frozen=True)
class DiscreteConfig:
    """Pure-step change-point settings for the discrete branch.

    ``n_permutations`` and ``random_seed`` are retained for the legacy
    fixed-feature diagnostic test. Normal end-to-end IsoVita inference uses
    ``IsoVitaConfig.inference`` for all selected branches.
    """

    min_samples_side: int = 10
    min_fraction_side: float = 0.15
    n_permutations: int = 1000
    alpha_fdr: float = 0.05
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.min_samples_side < 2:
            raise ValueError("min_samples_side must be at least 2.")
        if not (0.0 <= self.min_fraction_side < 0.5):
            raise ValueError("min_fraction_side must lie in [0, 0.5).")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1.")
        if not (0.0 < self.alpha_fdr < 1.0):
            raise ValueError("alpha_fdr must lie in (0, 1).")


@dataclass(frozen=True)
class FeatureTestConfig:
    """Spline-test settings for the continuous and localized branches."""

    spline_df: int = 3
    spline_degree: int = 3
    alpha_fdr: float = 0.05
    min_samples_global: int = 20
    min_samples_window: int = 8
    robust_amp_q_low: float = 0.05
    robust_amp_q_high: float = 0.95


@dataclass(frozen=True)
class PermutationInferenceConfig:
    """Selection-aware full-pipeline age-permutation inference settings."""

    n_permutations: int = 1000
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1.")


@dataclass(frozen=True)
class SelectionConfig:
    """Final union/ranking settings after the selected branches are merged."""

    # None retains every gene selected by at least one branch. Set an integer
    # only when a downstream use case explicitly requires a top-N subset.
    top_n_genes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.top_n_genes is not None and self.top_n_genes < 1:
            raise ValueError("top_n_genes must be None or >= 1.")


@dataclass(frozen=True)
class IsoVitaConfig:
    """Top-level configuration for the unified IsoVita workflow."""

    windows: WindowConfig = field(default_factory=WindowConfig)
    isoplex: IsoPlexConfig = field(default_factory=IsoPlexConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    discrete: DiscreteConfig = field(default_factory=DiscreteConfig)
    feature_test: FeatureTestConfig = field(default_factory=FeatureTestConfig)
    inference: PermutationInferenceConfig = field(
        default_factory=PermutationInferenceConfig
    )
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass
class IsoVitaPipelineResult:
    """Container for shared intermediates and the user-selected branches."""

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
    localized_trajectory_results: pd.DataFrame
    gene_trajectory_summary: pd.DataFrame
    discrete_trajectory_candidates: pd.DataFrame
    continuous_trajectory_candidates: pd.DataFrame
    localized_trajectory_candidates: pd.DataFrame
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


VALID_BRANCHES: Tuple[str, ...] = ("discrete", "continuous", "localized")


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
) -> Optional[Dict[str, object]]:
    """
    Build the effective-isoform composition for one gene.

    Only IsoPlex-effective isoforms are retained, and they are renormalized to
    sum to one within that effective set for every usable sample.
    """
    gene_id = str(gene_id)
    effective_isoforms = list(dict.fromkeys(map(str, effective_isoforms)))

    if len(effective_isoforms) < 2:
        return None

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

    common_samples = [
        sample for sample in mat.index
        if sample in metadata.index
    ]
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


def _fit_clr_trajectories(
    ages: np.ndarray,
    clr_values: np.ndarray,
    *,
    spline_df: int,
    spline_degree: int,
    age_grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit one smooth CLR trajectory per isoform and evaluate on a common grid.

    Returns
    -------
    age_grid : (G,)
    fitted_grid : (G, K)
    centered_grid : (G, K)
    """
    ages = np.asarray(ages, dtype=float)
    y = np.asarray(clr_values, dtype=float)

    order = np.argsort(ages)
    ages_sorted = ages[order]
    y_sorted = y[order]

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

    beta = np.linalg.lstsq(X, y_sorted, rcond=None)[0]

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

    fitted_grid = X_grid @ beta
    centered_grid = fitted_grid - fitted_grid.mean(axis=0, keepdims=True)

    return age_grid, fitted_grid, centered_grid


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

    return workspace


def _fit_spline_f_from_fixed_age_workspace(
    ages: np.ndarray,
    y: np.ndarray,
    workspace: Dict[str, object],
) -> float:
    """Compute the nested spline F statistic using a precomputed age basis."""
    ages = np.asarray(ages, dtype=float)
    y = np.asarray(y, dtype=float)
    if ages.shape != y.shape:
        raise ValueError("ages and y must have the same shape.")
    if len(y) < 3 or np.allclose(np.std(ages), 0) or np.allclose(np.std(y), 0):
        return np.nan

    order = np.argsort(ages, kind="mergesort")
    y_sorted = y[order]
    X = np.asarray(workspace["X"], dtype=float)
    if len(y_sorted) != X.shape[0]:
        raise ValueError("Age multiset does not match the precomputed spline workspace.")

    beta = np.asarray(workspace["X_pinv"], dtype=float) @ y_sorted
    fit1 = X @ beta
    rss0 = float(np.sum((y_sorted - np.mean(y_sorted)) ** 2))
    rss1 = float(np.sum((y_sorted - fit1) ** 2))
    rank1 = int(workspace["rank"])
    df_diff = rank1 - 1
    df_resid = len(y_sorted) - rank1
    if df_diff <= 0 or df_resid <= 0 or rss1 <= 0:
        return np.nan
    numerator = max(rss0 - rss1, 0.0) / df_diff
    denominator = rss1 / df_resid
    return float(numerator / denominator) if denominator > 0 else np.nan


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


def _iter_canonical_assignments(k: int):
    """
    Yield one representative of every A/B/unused assignment up to sign.

    The first non-zero label is forced to +1, eliminating duplicate A/B swaps.
    """
    for labels in product((-1, 0, 1), repeat=k):
        z = np.asarray(labels, dtype=np.int8)
        if not np.any(z == 1) or not np.any(z == -1):
            continue
        first_nonzero = z[np.flatnonzero(z)[0]]
        if first_nonzero != 1:
            continue
        yield z


@lru_cache(maxsize=None)
def _cached_balance_candidates(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache assignments/directions for a given K; content depends only on K."""
    assignments = []
    directions = []
    n_used = []

    for z in _iter_canonical_assignments(k):
        assignments.append(z)
        directions.append(_assignment_to_balance_direction(z))
        n_used.append(int(np.count_nonzero(z)))

    if not assignments:
        raise ValueError(f"No valid balance assignments exist for K={k}.")

    return (
        np.vstack(assignments).astype(np.int8),
        np.vstack(directions).astype(float),
        np.asarray(n_used, dtype=np.int16),
    )


def find_sparse_balance(
    pc1: np.ndarray,
    *,
    near_optimal_cosine_tolerance: float = 0.01,
    cache_candidate_directions_through_k: int = 10,
) -> Dict[str, object]:
    """
    Exhaustively search A/B/unused balances and choose the simplest near-optimum.

    Selection rule
    --------------
    1. Compute absolute cosine similarity to PC1 for every valid balance.
    2. Let c* be the best cosine.
    3. Near-optimal set: cosine >= c* - tolerance.
    4. Choose the candidate using the fewest isoforms.
    5. Within that complexity, choose the highest cosine.
    6. Break exact numerical ties deterministically by candidate order.

    The chosen balance is finally oriented to have non-negative dot product
    with the deterministically oriented PC1.
    """
    pc1 = np.asarray(pc1, dtype=float)
    pc1 = pc1 / np.linalg.norm(pc1)
    k = len(pc1)

    if k < 2:
        raise ValueError("At least two isoforms are required for a balance.")

    # K=2 has exactly one balance up to sign; bypass exhaustive enumeration.
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

    tolerance = float(near_optimal_cosine_tolerance)

    if k <= int(cache_candidate_directions_through_k):
        assignments, directions, n_used = _cached_balance_candidates(k)
        cosine = np.abs(directions @ pc1)
        best = float(np.max(cosine))
        near = cosine >= (best - tolerance - 1e-12)
        min_used = int(np.min(n_used[near]))
        eligible = near & (n_used == min_used)
        eligible_idx = np.flatnonzero(eligible)
        chosen_idx = int(eligible_idx[np.argmax(cosine[eligible_idx])])

        assignment = assignments[chosen_idx].copy()
        direction = directions[chosen_idx].copy()
        chosen_cosine = float(cosine[chosen_idx])
        n_near = int(np.sum(near))

    else:
        # Memory-safe exact fallback for unusually large K.
        records: List[Tuple[float, int, np.ndarray, np.ndarray]] = []
        best = -np.inf
        for z in _iter_canonical_assignments(k):
            direction = _assignment_to_balance_direction(z)
            c = float(abs(np.dot(direction, pc1)))
            best = max(best, c)
            records.append((c, int(np.count_nonzero(z)), z.copy(), direction))

        threshold = best - tolerance - 1e-12
        near_records = [record for record in records if record[0] >= threshold]
        min_used = min(record[1] for record in near_records)
        near_simple = [record for record in near_records if record[1] == min_used]
        chosen_cosine, _, assignment, direction = max(
            near_simple,
            key=lambda record: record[0],
        )
        n_near = len(near_records)

    # PC1 sign is deterministic; orient A/B to point in that PC1 direction.
    if np.dot(direction, pc1) < 0:
        assignment = -assignment
        direction = -direction

    return {
        "assignment": assignment.astype(np.int8),
        "direction": direction.astype(float),
        "cosine_similarity": float(chosen_cosine),
        "best_cosine_similarity": float(best),
        "n_used": int(np.count_nonzero(assignment)),
        "n_near_optimal": int(n_near),
    }


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
    spline_workspace: Optional[Dict[str, object]] = None,
    clr_values: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Learn the age-informed sparse balance from an already prepared gene.

    The effective-isoform composition is fixed. ``ages`` may be the observed age
    vector or a permutation of it. When ``spline_workspace`` is supplied, the
    function reuses the spline basis for the unchanged multiset of age values,
    which makes full-pipeline permutation inference substantially faster.
    """
    isoforms = list(prepared["isoforms"])
    samples = np.asarray(prepared["samples"], dtype=str)
    composition = np.asarray(prepared["composition"], dtype=float)
    ages = np.asarray(ages, dtype=float)

    if len(ages) != len(samples):
        raise ValueError("ages must align one-to-one with prepared samples.")
    if np.any(~np.isfinite(ages)):
        raise ValueError("ages must be finite.")

    clr = (
        np.asarray(clr_values, dtype=float)
        if clr_values is not None
        else clr_transform(composition)
    )

    if spline_workspace is None:
        age_grid, fitted_grid, centered_grid = _fit_clr_trajectories(
            ages,
            clr,
            spline_df=config.spline_df,
            spline_degree=config.spline_degree,
            age_grid_size=config.age_grid_size,
        )
    else:
        order = np.argsort(ages, kind="mergesort")
        clr_sorted = clr[order]
        X = np.asarray(spline_workspace["X"], dtype=float)
        if X.shape[0] != len(ages):
            raise ValueError(
                "Age multiset does not match the feature-learning spline workspace."
            )
        beta = np.asarray(spline_workspace["X_pinv"], dtype=float) @ clr_sorted
        age_grid = np.asarray(spline_workspace["age_grid"], dtype=float)
        X_grid = np.asarray(spline_workspace["X_grid"], dtype=float)
        fitted_grid = X_grid @ beta
        centered_grid = fitted_grid - fitted_grid.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(centered_grid, full_matrices=False)
    pc1 = _orient_pc1(vt[0])

    balance = find_sparse_balance(
        pc1,
        near_optimal_cosine_tolerance=config.near_optimal_cosine_tolerance,
        cache_candidate_directions_through_k=(
            config.cache_candidate_directions_through_k
        ),
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
) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series]:
    """Fit the complete one-feature-per-gene IsoVita balance for one gene."""
    gene_id = str(gene_id)
    prepared = make_effective_isoform_composition(
        gene_id,
        expression_df,
        effective_isoforms,
        sample_cols,
        metadata,
        pseudocount=config.clr_pseudocount,
        min_samples=config.min_samples,
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
        learned = _learn_balance_from_prepared(
            prepared,
            ages,
            config=config,
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
            "n_near_optimal": int(balance["n_near_optimal"]),
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one sparse-balance feature for every gene with >=2 effective isoforms."""
    effective_map = get_effective_isoforms(global_metrics)
    genes = [gene for gene, txs in effective_map.items() if len(txs) >= 2]

    summary_rows: List[Dict[str, object]] = []
    loading_tables: List[pd.DataFrame] = []
    feature_series: List[pd.Series] = []

    for i, gene_id in enumerate(genes):
        if progress_every and i % progress_every == 0:
            print(f"Balance: {i}/{len(genes)}", flush=True)

        summary, loadings, feature = fit_gene_balance(
            gene_id,
            expression_df,
            effective_map[gene_id],
            sample_cols,
            metadata,
            config=config,
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


def _permuted_max_step_f_statistics(
    y_sorted: np.ndarray,
    cut_positions: np.ndarray,
    *,
    n_permutations: int,
    random_seed: int,
) -> np.ndarray:
    """Repeat the full cut search under synchronized feature-value permutations."""
    y = np.asarray(y_sorted, dtype=float)
    cuts = np.asarray(cut_positions, dtype=int)
    n = len(y)

    perm_idx = _cached_permutation_indices(n, n_permutations, random_seed)
    yp = y[perm_idx]  # B x n

    csum = np.cumsum(yp, axis=1)
    cssq = np.cumsum(yp * yp, axis=1)
    total_sum = csum[:, -1][:, None]
    total_ssq = cssq[:, -1][:, None]

    left_idx = cuts - 1
    sum_left = csum[:, left_idx]
    ssq_left = cssq[:, left_idx]
    sum_right = total_sum - sum_left
    ssq_right = total_ssq - ssq_left

    n_left = cuts[None, :].astype(float)
    n_right = (n - cuts)[None, :].astype(float)

    rss_left = np.maximum(ssq_left - (sum_left * sum_left) / n_left, 0.0)
    rss_right = np.maximum(ssq_right - (sum_right * sum_right) / n_right, 0.0)
    rss1 = np.maximum(rss_left + rss_right, 0.0)

    rss0 = np.maximum(total_ssq - (total_sum * total_sum) / n, 0.0)
    delta = np.maximum(rss0 - rss1, 0.0)
    denom = rss1 / (n - 2)

    scale = np.maximum(1.0, rss0)
    eps = np.finfo(float).eps * scale * 32.0
    regular = denom > eps

    f_stats = np.zeros_like(rss1, dtype=float)
    f_stats[regular] = delta[regular] / denom[regular]
    f_stats[(~regular) & (delta > eps)] = np.inf

    return np.max(f_stats, axis=1)


def _fit_discrete_step_statistic(
    age: np.ndarray,
    y: np.ndarray,
    *,
    config: DiscreteConfig,
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

    order = np.argsort(age, kind="mergesort")
    age_sorted = age[order]
    y_sorted = y[order]

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


def _fit_discrete_step_test(
    age: np.ndarray,
    y: np.ndarray,
    *,
    config: DiscreteConfig,
) -> Dict[str, object]:
    """Legacy fixed-feature permutation test for one unknown pure step.

    This remains available for diagnostics or a feature specified independently
    of age. For a normal IsoVita learned balance, use the full-pipeline
    selection-aware inference functions below instead.
    """
    observed = _fit_discrete_step_statistic(age, y, config=config)
    observed["permutation_p_value"] = np.nan
    observed["p_value"] = np.nan

    if not observed["tested"]:
        return observed

    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(age) & np.isfinite(y)
    age = age[ok]
    y = y[ok]
    order = np.argsort(age, kind="mergesort")
    age_sorted = age[order]
    y_sorted = y[order]
    cut_positions, _ = _valid_change_point_positions(
        age_sorted,
        min_samples_side=config.min_samples_side,
        min_fraction_side=config.min_fraction_side,
    )

    null_max_f = _permuted_max_step_f_statistics(
        y_sorted,
        cut_positions,
        n_permutations=config.n_permutations,
        random_seed=config.random_seed,
    )
    best_f = float(observed["max_f_stat"])
    permutation_p = float(
        (1 + np.count_nonzero(null_max_f >= best_f))
        / (config.n_permutations + 1)
    )
    observed["permutation_p_value"] = permutation_p
    observed["p_value"] = permutation_p
    return observed

def run_discrete_feature_tests(
    balance_feature_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    config: DiscreteConfig = DiscreteConfig(),
    progress_every: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the pure-step change-point branch, one search-adjusted test per gene.

    The p-value is based on permutations that repeat the complete search over
    all valid age cut points. BH is then applied once across tested genes.

    Because the balance feature was itself learned using age, this permutation
    procedure corrects the change-point search but not the upstream
    age-informed feature-learning step.
    """
    rows: List[Dict[str, object]] = []

    for i, (gene_id, feature_row) in enumerate(balance_feature_matrix.iterrows()):
        if progress_every and i % progress_every == 0:
            print(
                f"Discrete feature test: {i}/{len(balance_feature_matrix)}",
                flush=True,
            )

        samples = [sample for sample in feature_row.index if sample in metadata.index]
        y = feature_row.loc[samples].to_numpy(dtype=float)
        age = metadata.loc[samples, "age"].to_numpy(dtype=float)

        result = _fit_discrete_step_test(age, y, config=config)
        result["gene_id"] = str(gene_id)
        rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = add_bh_fdr(
        out,
        "p_value",
        alpha=config.alpha_fdr,
        valid_mask=out["tested"],
        q_col="q_value",
        reject_col="fdr_significant",
    )

    return out.sort_values(
        ["q_value", "p_value", "abs_step_effect", "gene_id"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)


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


def run_continuous_feature_tests(
    balance_feature_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    config: FeatureTestConfig = FeatureTestConfig(),
    progress_every: Optional[int] = None,
) -> pd.DataFrame:
    """Run a fixed-feature continuous test for diagnostics.

    This conditional test is calibrated only when the supplied feature was
    specified independently of age. Normal IsoVita inference uses
    ``run_selection_aware_temporal_tests()`` instead.
    """
    rows = []

    for i, (gene_id, feature_row) in enumerate(balance_feature_matrix.iterrows()):
        if progress_every and i % progress_every == 0:
            print(
                f"Continuous feature test: {i}/{len(balance_feature_matrix)}",
                flush=True,
            )

        samples = [sample for sample in feature_row.index if sample in metadata.index]
        y = feature_row.loc[samples].to_numpy(dtype=float)
        age = metadata.loc[samples, "age"].to_numpy(dtype=float)

        result = _fit_nested_spline_test(
            age,
            y,
            spline_df=config.spline_df,
            spline_degree=config.spline_degree,
        )
        corr = _feature_correlations(age, y)
        result.update(corr)
        result["gene_id"] = str(gene_id)
        result["tested"] = bool(
            result["n"] >= config.min_samples_global
            and np.isfinite(result["p_value"])
        )
        if not result["tested"]:
            result["p_value"] = np.nan
        rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = add_bh_fdr(
        out,
        "p_value",
        alpha=config.alpha_fdr,
        valid_mask=out["tested"],
        q_col="q_value",
        reject_col="fdr_significant",
    )

    out["score_predictive_hybrid"] = (
        out["abs_pearson_r"].fillna(0.0)
        * out["fitted_robust_amp"].fillna(0.0)
    )

    return out.sort_values(
        ["q_value", "p_value", "score_predictive_hybrid"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def run_localized_feature_tests(
    balance_feature_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    windows: WindowConfig = WindowConfig(),
    config: FeatureTestConfig = FeatureTestConfig(),
    progress_every: Optional[int] = None,
) -> pd.DataFrame:
    """Run fixed-feature localized tests for diagnostics.

    This conditional test is calibrated only when the supplied feature was
    specified independently of age. Normal IsoVita inference uses
    ``run_selection_aware_temporal_tests()`` instead.
    """
    rows = []
    window_bins = np.asarray(windows.edges, dtype=float).copy()
    # Keep [a, b) windows while ensuring a sample exactly at the final edge
    # (e.g. age 85 for the default 40-85 window) is not silently dropped.
    window_bins[-1] = np.nextafter(window_bins[-1], np.inf)
    window_series = pd.cut(
        metadata["age"],
        bins=window_bins,
        labels=list(windows.labels),
        right=False,
        include_lowest=True,
    )

    for i, (gene_id, feature_row) in enumerate(balance_feature_matrix.iterrows()):
        if progress_every and i % progress_every == 0:
            print(
                f"Localized feature test: {i}/{len(balance_feature_matrix)}",
                flush=True,
            )

        for label in windows.labels:
            win_samples = metadata.index[window_series.astype(str).eq(label)]
            samples = [sample for sample in win_samples if sample in feature_row.index]

            y = feature_row.loc[samples].to_numpy(dtype=float)
            age = metadata.loc[samples, "age"].to_numpy(dtype=float)

            result = _fit_nested_spline_test(
                age,
                y,
                spline_df=config.spline_df,
                spline_degree=config.spline_degree,
            )
            corr = _feature_correlations(age, y)
            result.update(corr)
            result.update({
                "gene_id": str(gene_id),
                "window": label,
            })
            result["tested"] = bool(
                result["n"] >= config.min_samples_window
                and np.isfinite(result["p_value"])
            )
            if not result["tested"]:
                result["p_value"] = np.nan
            rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # The localized branch currently treats all tested gene-window hypotheses
    # as one BH family. Branch-level gene evidence is subsequently defined as
    # at least one FDR-significant window.
    out = add_bh_fdr(
        out,
        "p_value",
        alpha=config.alpha_fdr,
        valid_mask=out["tested"],
        q_col="q_value",
        reject_col="fdr_significant",
    )

    return out.sort_values(
        ["q_value", "p_value", "gene_id", "window"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SELECTION-AWARE FULL-PIPELINE PERMUTATION INFERENCE
# ---------------------------------------------------------------------------


def _localized_statistics_for_feature(
    age: np.ndarray,
    y: np.ndarray,
    *,
    windows: WindowConfig,
    config: FeatureTestConfig,
) -> Dict[str, Dict[str, object]]:
    """Compute window-specific statistics for one feature without multiplicity correction."""
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    if age.shape != y.shape:
        raise ValueError("age and y must have the same shape.")

    bins = np.asarray(windows.edges, dtype=float).copy()
    bins[-1] = np.nextafter(bins[-1], np.inf)
    labels = np.asarray(
        pd.cut(
            age,
            bins=bins,
            labels=list(windows.labels),
            right=False,
            include_lowest=True,
        ).astype(object),
        dtype=object,
    )

    out: Dict[str, Dict[str, object]] = {}
    for label in windows.labels:
        mask = labels == label
        age_w = age[mask]
        y_w = y[mask]
        result = _fit_nested_spline_test(
            age_w,
            y_w,
            spline_df=config.spline_df,
            spline_degree=config.spline_degree,
        )
        result.update(_feature_correlations(age_w, y_w))
        result["window"] = label
        result["tested"] = bool(
            result["n"] >= config.min_samples_window
            and np.isfinite(result["f_stat"])
        )
        if not result["tested"]:
            result["p_value"] = np.nan
        out[label] = result
    return out


def _selection_aware_permutation_p_value(
    observed_stat: float,
    null_stats: np.ndarray,
) -> float:
    """Monte Carlo upper-tail p-value with the standard +1 correction."""
    null_stats = np.asarray(null_stats, dtype=float)
    if not np.isfinite(observed_stat):
        return np.nan
    if null_stats.size == 0 or np.any(~np.isfinite(null_stats)):
        raise ValueError("All full-pipeline permutation statistics must be finite.")
    return float(
        (1 + np.count_nonzero(null_stats >= float(observed_stat)))
        / (len(null_stats) + 1)
    )


def _run_selection_aware_gene_tests(
    gene_id: str,
    prepared: Dict[str, object],
    observed_feature: np.ndarray,
    *,
    branches: Sequence[str],
    balance_config: BalanceConfig,
    discrete_config: DiscreteConfig,
    feature_config: FeatureTestConfig,
    windows: WindowConfig,
    inference_config: PermutationInferenceConfig,
) -> Dict[str, object]:
    """Run selected branches for one gene with full age-permutation relearning."""
    branches = normalize_branches(branches)
    gene_id = str(gene_id)

    ages = np.asarray(prepared["ages"], dtype=float)
    samples = np.asarray(prepared["samples"], dtype=str)
    observed_feature = np.asarray(observed_feature, dtype=float)
    if len(observed_feature) != len(ages):
        raise ValueError(
            f"Observed feature for {gene_id} does not align with prepared samples."
        )

    result: Dict[str, object] = {}

    # Observed statistics are computed from the balance learned with observed age.
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
            spline_df=feature_config.spline_df,
            spline_degree=feature_config.spline_degree,
        )
        continuous_obs.update(_feature_correlations(ages, observed_feature))
        continuous_obs["gene_id"] = gene_id
        continuous_obs["tested"] = bool(
            continuous_obs["n"] >= feature_config.min_samples_global
            and np.isfinite(continuous_obs["f_stat"])
        )
        continuous_obs["analytic_p_value"] = continuous_obs["p_value"]
        if not continuous_obs["tested"]:
            continuous_obs["p_value"] = np.nan
        result["continuous"] = continuous_obs

    if "localized" in branches:
        localized_obs = _localized_statistics_for_feature(
            ages,
            observed_feature,
            windows=windows,
            config=feature_config,
        )
        result["localized"] = localized_obs

    # Precompute every object that depends only on the multiset of ages. A
    # permutation changes which sample receives which age, but not these bases.
    clr_values = clr_transform(np.asarray(prepared["composition"], dtype=float))
    balance_workspace = _build_fixed_age_spline_workspace(
        ages,
        spline_df=balance_config.spline_df,
        spline_degree=balance_config.spline_degree,
        age_grid_size=balance_config.age_grid_size,
    )

    continuous_workspace = None
    if "continuous" in branches and bool(result["continuous"]["tested"]):
        continuous_workspace = _build_fixed_age_spline_workspace(
            ages,
            spline_df=feature_config.spline_df,
            spline_degree=feature_config.spline_degree,
        )

    localized_workspaces: Dict[str, Dict[str, object]] = {}
    if "localized" in branches:
        edges = np.asarray(windows.edges, dtype=float).copy()
        edges[-1] = np.nextafter(edges[-1], np.inf)
        for i, label in enumerate(windows.labels):
            if not bool(result["localized"][label]["tested"]):
                continue
            mask = (ages >= edges[i]) & (ages < edges[i + 1])
            localized_workspaces[label] = _build_fixed_age_spline_workspace(
                ages[mask],
                spline_df=feature_config.spline_df,
                spline_degree=feature_config.spline_degree,
            )
    else:
        edges = np.asarray([], dtype=float)

    B = int(inference_config.n_permutations)
    perm_idx = _cached_permutation_indices(
        len(samples),
        B,
        int(inference_config.random_seed),
    )

    discrete_null = np.empty(B, dtype=float) if "discrete" in branches else None
    continuous_null = np.empty(B, dtype=float) if "continuous" in branches else None
    localized_null = (
        {label: np.empty(B, dtype=float) for label in windows.labels}
        if "localized" in branches else None
    )

    # Each permutation repeats every age-dependent feature-learning step once.
    # The resulting permuted feature is then shared across all selected branches.
    for b in range(B):
        permuted_ages = ages[perm_idx[b]]
        learned = _learn_balance_from_prepared(
            prepared,
            permuted_ages,
            config=balance_config,
            spline_workspace=balance_workspace,
            clr_values=clr_values,
        )
        permuted_feature = np.asarray(learned["feature"], dtype=float)

        if discrete_null is not None:
            if bool(result["discrete"]["tested"]):
                stat = _fit_discrete_step_statistic(
                    permuted_ages,
                    permuted_feature,
                    config=discrete_config,
                )
                if not bool(stat["tested"]) or not np.isfinite(stat["max_f_stat"]):
                    raise RuntimeError(
                        f"Discrete permutation became untestable for {gene_id}."
                    )
                discrete_null[b] = float(stat["max_f_stat"])
            else:
                discrete_null[b] = 0.0

        if continuous_null is not None:
            if bool(result["continuous"]["tested"]):
                f_stat = _fit_spline_f_from_fixed_age_workspace(
                    permuted_ages,
                    permuted_feature,
                    continuous_workspace,
                )
                if not np.isfinite(f_stat):
                    raise RuntimeError(
                        f"Continuous permutation became untestable for {gene_id}."
                    )
                continuous_null[b] = float(f_stat)
            else:
                continuous_null[b] = 0.0

        if localized_null is not None:
            for i, label in enumerate(windows.labels):
                observed_window = result["localized"][label]
                if not bool(observed_window["tested"]):
                    localized_null[label][b] = 0.0
                    continue

                mask = (
                    (permuted_ages >= edges[i])
                    & (permuted_ages < edges[i + 1])
                )
                f_stat = _fit_spline_f_from_fixed_age_workspace(
                    permuted_ages[mask],
                    permuted_feature[mask],
                    localized_workspaces[label],
                )
                if not np.isfinite(f_stat):
                    raise RuntimeError(
                        f"Localized permutation became untestable for "
                        f"{gene_id}, window {label}."
                    )
                localized_null[label][b] = float(f_stat)

    inference_label = "full_pipeline_age_permutation"

    if discrete_null is not None:
        obs = result["discrete"]
        if bool(obs["tested"]):
            p = _selection_aware_permutation_p_value(
                float(obs["max_f_stat"]),
                discrete_null,
            )
        else:
            p = np.nan
        obs["selection_aware_p_value"] = p
        obs["permutation_p_value"] = p
        obs["p_value"] = p
        obs["n_permutations"] = B
        obs["inference_method"] = inference_label

    if continuous_null is not None:
        obs = result["continuous"]
        if bool(obs["tested"]):
            p = _selection_aware_permutation_p_value(
                float(obs["f_stat"]),
                continuous_null,
            )
        else:
            p = np.nan
        obs["selection_aware_p_value"] = p
        obs["p_value"] = p
        obs["n_permutations"] = B
        obs["inference_method"] = inference_label

    if localized_null is not None:
        for label in windows.labels:
            obs = result["localized"][label]
            obs["gene_id"] = gene_id
            obs["analytic_p_value"] = obs["p_value"]
            if bool(obs["tested"]):
                p = _selection_aware_permutation_p_value(
                    float(obs["f_stat"]),
                    localized_null[label],
                )
            else:
                p = np.nan
            obs["selection_aware_p_value"] = p
            obs["p_value"] = p
            obs["n_permutations"] = B
            obs["inference_method"] = inference_label

    return result

def run_selection_aware_temporal_tests(
    expression_df: pd.DataFrame,
    global_metrics: pd.DataFrame,
    sample_cols: Sequence[str],
    metadata: pd.DataFrame,
    *,
    branches: Sequence[str] = VALID_BRANCHES,
    observed_feature_matrix: Optional[pd.DataFrame] = None,
    balance_config: BalanceConfig = BalanceConfig(),
    discrete_config: DiscreteConfig = DiscreteConfig(),
    feature_config: FeatureTestConfig = FeatureTestConfig(),
    windows: WindowConfig = WindowConfig(),
    inference_config: PermutationInferenceConfig = PermutationInferenceConfig(),
    progress_every: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Run selection-aware inference for any requested IsoVita branches.

    For each gene, age is permuted and the complete age-informed balance is
    relearned before recomputing the selected branch statistic(s). The same
    relearned feature is shared by all selected branches within a permutation.

    Notes
    -----
    ``run_discrete_feature_tests()``, ``run_continuous_feature_tests()``, and
    ``run_localized_feature_tests()`` remain available as fixed-feature
    diagnostics. Normal IsoVita inference should use this function (or
    ``run_isovita_pipeline()``, which calls it automatically).
    """
    branches = normalize_branches(branches)
    effective_map = get_effective_isoforms(global_metrics)

    if observed_feature_matrix is not None:
        genes = [
            str(gene)
            for gene in observed_feature_matrix.index
            if str(gene) in effective_map and len(effective_map[str(gene)]) >= 2
        ]
    else:
        genes = [gene for gene, txs in effective_map.items() if len(txs) >= 2]

    rows: Dict[str, List[Dict[str, object]]] = {
        "discrete": [],
        "continuous": [],
        "localized": [],
    }

    # Monte Carlo p-values cannot be smaller than 1/(B+1). Warn when that
    # resolution is coarser than the first BH threshold for a selected family.
    min_possible_p = 1.0 / (int(inference_config.n_permutations) + 1.0)
    family_sizes = []
    if "discrete" in branches:
        family_sizes.append(("discrete", len(genes), discrete_config.alpha_fdr))
    if "continuous" in branches:
        family_sizes.append(("continuous", len(genes), feature_config.alpha_fdr))
    if "localized" in branches:
        family_sizes.append((
            "localized",
            len(genes) * len(windows.labels),
            feature_config.alpha_fdr,
        ))
    for branch_name, family_size, alpha in family_sizes:
        if family_size > 0 and min_possible_p > alpha / family_size:
            warnings.warn(
                f"{branch_name} permutation resolution may be coarse for BH: "
                f"minimum p={min_possible_p:.3g}, first BH threshold="
                f"{alpha / family_size:.3g}. Increase inference.n_permutations "
                "when precise tail resolution is required.",
                RuntimeWarning,
                stacklevel=2,
            )

    for i, gene_id in enumerate(genes):
        if progress_every and i % progress_every == 0:
            print(
                f"Selection-aware inference: {i}/{len(genes)}",
                flush=True,
            )

        prepared = make_effective_isoform_composition(
            gene_id,
            expression_df,
            effective_map[gene_id],
            sample_cols,
            metadata,
            pseudocount=balance_config.clr_pseudocount,
            min_samples=balance_config.min_samples,
        )
        if prepared is None:
            continue

        samples = np.asarray(prepared["samples"], dtype=str)
        if observed_feature_matrix is not None and gene_id in observed_feature_matrix.index:
            feature_row = observed_feature_matrix.loc[gene_id]
            missing = [sample for sample in samples if sample not in feature_row.index]
            if missing:
                raise ValueError(
                    f"Observed feature matrix is missing samples for {gene_id}: "
                    f"{missing[:5]}"
                )
            observed_feature = feature_row.loc[samples].to_numpy(dtype=float)
        else:
            learned = _learn_balance_from_prepared(
                prepared,
                np.asarray(prepared["ages"], dtype=float),
                config=balance_config,
            )
            observed_feature = np.asarray(learned["feature"], dtype=float)

        gene_result = _run_selection_aware_gene_tests(
            gene_id,
            prepared,
            observed_feature,
            branches=branches,
            balance_config=balance_config,
            discrete_config=discrete_config,
            feature_config=feature_config,
            windows=windows,
            inference_config=inference_config,
        )

        if "discrete" in branches:
            rows["discrete"].append(dict(gene_result["discrete"]))
        if "continuous" in branches:
            rows["continuous"].append(dict(gene_result["continuous"]))
        if "localized" in branches:
            rows["localized"].extend(
                dict(gene_result["localized"][label])
                for label in windows.labels
            )

    out: Dict[str, pd.DataFrame] = {
        "discrete": pd.DataFrame(rows["discrete"]),
        "continuous": pd.DataFrame(rows["continuous"]),
        "localized": pd.DataFrame(rows["localized"]),
    }

    if "discrete" in branches and not out["discrete"].empty:
        table = add_bh_fdr(
            out["discrete"],
            "p_value",
            alpha=discrete_config.alpha_fdr,
            valid_mask=out["discrete"]["tested"],
            q_col="q_value",
            reject_col="fdr_significant",
        )
        out["discrete"] = table.sort_values(
            ["q_value", "p_value", "abs_step_effect", "gene_id"],
            ascending=[True, True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    if "continuous" in branches and not out["continuous"].empty:
        table = add_bh_fdr(
            out["continuous"],
            "p_value",
            alpha=feature_config.alpha_fdr,
            valid_mask=out["continuous"]["tested"],
            q_col="q_value",
            reject_col="fdr_significant",
        )
        table["score_predictive_hybrid"] = (
            table["abs_pearson_r"].fillna(0.0)
            * table["fitted_robust_amp"].fillna(0.0)
        )
        out["continuous"] = table.sort_values(
            ["q_value", "p_value", "score_predictive_hybrid"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)

    if "localized" in branches and not out["localized"].empty:
        table = add_bh_fdr(
            out["localized"],
            "p_value",
            alpha=feature_config.alpha_fdr,
            valid_mask=out["localized"]["tested"],
            q_col="q_value",
            reject_col="fdr_significant",
        )
        out["localized"] = table.sort_values(
            ["q_value", "p_value", "gene_id", "window"],
            ascending=[True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# MERGE USER-SELECTED PARALLEL BRANCHES
# ---------------------------------------------------------------------------


def _trajectory_signature(row: pd.Series, branches: Sequence[str]) -> str:
    labels = [
        branch
        for branch in branches
        if bool(row.get(f"{branch}_fdr_significant", False))
    ]
    return "+".join(labels) if labels else "not_selected"


def _localized_gene_summary(
    localized_results: pd.DataFrame,
    gene_ids: pd.Series,
) -> pd.DataFrame:
    if localized_results.empty:
        return pd.DataFrame({"gene_id": gene_ids.astype(str)})

    win = localized_results.copy()
    win["is_sig"] = win["fdr_significant"].fillna(False).astype(bool)
    agg_rows = []
    for gene_id, sub in win.groupby("gene_id", sort=False):
        sig = sub.loc[sub["is_sig"]].sort_values(
            ["q_value", "p_value"], na_position="last"
        )
        best = sub.sort_values(
            ["q_value", "p_value"], na_position="last"
        ).iloc[0]
        agg_rows.append({
            "gene_id": str(gene_id),
            "n_tested_windows": int(sub["tested"].fillna(False).sum()),
            "n_significant_windows": int(sub["is_sig"].sum()),
            "significant_windows": "|".join(sig["window"].astype(str).tolist()),
            "best_window": str(best["window"]),
            "best_window_p_value": best["p_value"],
            "best_window_q_value": best["q_value"],
            "best_window_r_squared": best["r_squared"],
            "best_window_fitted_robust_amp": best["fitted_robust_amp"],
            "localized_fdr_significant": bool(sub["is_sig"].any()),
        })
    return pd.DataFrame(agg_rows)


def summarize_gene_trajectories(
    balance_summary: pd.DataFrame,
    discrete_results: pd.DataFrame,
    continuous_results: pd.DataFrame,
    localized_results: pd.DataFrame,
    *,
    branches: Sequence[str] = VALID_BRANCHES,
) -> pd.DataFrame:
    """Merge evidence from exactly the branches requested by the user."""
    branches = normalize_branches(branches)
    ok_balance = balance_summary.loc[balance_summary["status"].eq("ok")].copy()
    if ok_balance.empty:
        return ok_balance

    summary = ok_balance.copy()

    if "discrete" in branches:
        discrete_keep = discrete_results.copy()
        if not discrete_keep.empty:
            discrete_keep = discrete_keep.rename(columns={
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
            discrete_keep = discrete_keep[[
                c for c in keep_cols if c in discrete_keep.columns
            ]]
            summary = summary.merge(discrete_keep, on="gene_id", how="left")

    if "continuous" in branches:
        continuous_keep = continuous_results.copy()
        if not continuous_keep.empty:
            continuous_keep = continuous_keep.rename(columns={
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
            continuous_keep = continuous_keep[[
                c for c in keep_cols if c in continuous_keep.columns
            ]]
            summary = summary.merge(continuous_keep, on="gene_id", how="left")

    if "localized" in branches:
        localized_agg = _localized_gene_summary(
            localized_results, ok_balance["gene_id"]
        )
        summary = summary.merge(localized_agg, on="gene_id", how="left")

    for branch in VALID_BRANCHES:
        col = f"{branch}_fdr_significant"
        if col not in summary.columns:
            summary[col] = False
        summary[col] = summary[col].fillna(False).astype(bool)

    if "n_significant_windows" not in summary.columns:
        summary["n_significant_windows"] = 0
    summary["n_significant_windows"] = (
        summary["n_significant_windows"].fillna(0).astype(int)
    )

    selected_sig_cols = [f"{branch}_fdr_significant" for branch in branches]
    summary["n_detection_branches"] = (
        summary[selected_sig_cols].astype(int).sum(axis=1)
    )
    summary["trajectory_signature"] = summary.apply(
        lambda row: _trajectory_signature(row, branches), axis=1
    )

    q_col_map = {
        "discrete": "discrete_q_value",
        "continuous": "continuous_q_value",
        "localized": "best_window_q_value",
    }
    q_cols = [q_col_map[b] for b in branches if q_col_map[b] in summary.columns]
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
    branches: Sequence[str] = VALID_BRANCHES,
    config: SelectionConfig = SelectionConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selected-branch candidates and their unique gene-level union."""
    branches = normalize_branches(branches)
    if gene_summary.empty:
        empty = gene_summary.copy()
        return empty, empty, empty, empty

    candidate_tables: Dict[str, pd.DataFrame] = {}
    for branch in VALID_BRANCHES:
        sig_col = f"{branch}_fdr_significant"
        if branch in branches and sig_col in gene_summary.columns:
            candidate_tables[branch] = gene_summary.loc[
                gene_summary[sig_col].fillna(False)
            ].copy()
        else:
            candidate_tables[branch] = gene_summary.iloc[0:0].copy()

    selected_sig_cols = [f"{branch}_fdr_significant" for branch in branches]
    selected_mask = gene_summary[selected_sig_cols].fillna(False).any(axis=1)
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
        candidate_tables["discrete"].reset_index(drop=True),
        candidate_tables["continuous"].reset_index(drop=True),
        candidate_tables["localized"].reset_index(drop=True),
        selected.reset_index(drop=True),
    )


# ============================================================================
# END-TO-END ORCHESTRATOR
# ============================================================================


def run_isovita_pipeline(
    *,
    tpm: pd.DataFrame,
    metadata: pd.DataFrame,
    branches: Sequence[str] = VALID_BRANCHES,
    config: IsoVitaConfig = IsoVitaConfig(),
    progress_every: Optional[int] = None,
) -> IsoVitaPipelineResult:
    """Run IsoVita using only the requested temporal detection branches.

    Parameters
    ----------
    tpm
        Standardized transcript-level TPM table with ``gene_id``,
        ``transcript_id``, then one column per retained sample.
    metadata
        Standardized sample metadata with columns ``sample_id`` and ``age``.
        It must already reflect all dataset/cohort filtering.
    branches
        Any non-empty subset of ``("discrete", "continuous", "localized")``.
        Only selected branches are executed. The final gene list is their union.
    """
    selected_branches = normalize_branches(branches)
    full_tpm, meta, sample_cols = validate_core_inputs(tpm, metadata)

    isoplex_expression_df, isoplex_filter_summary = filter_tpm_for_isoplex(
        full_tpm, sample_cols, config=config.isoplex
    )
    sample_metrics, global_metrics = run_isoplex(isoplex_expression_df)
    gene_routes = route_genes_by_effective_isoform_count(global_metrics)

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
        )
    )

    # Final inference is selection-aware: every age permutation relearns the
    # age-informed balance before recomputing the selected branch statistics.
    branch_results = run_selection_aware_temporal_tests(
        expression_df=full_tpm,
        global_metrics=global_metrics,
        sample_cols=sample_cols,
        metadata=meta,
        branches=selected_branches,
        observed_feature_matrix=balance_feature_matrix,
        balance_config=config.balance,
        discrete_config=config.discrete,
        feature_config=config.feature_test,
        windows=config.windows,
        inference_config=config.inference,
        progress_every=progress_every,
    )

    discrete_results = branch_results["discrete"]
    continuous_results = branch_results["continuous"]
    localized_results = branch_results["localized"]

    gene_summary = summarize_gene_trajectories(
        balance_summary,
        discrete_results,
        continuous_results,
        localized_results,
        branches=selected_branches,
    )

    (
        discrete_candidates,
        continuous_candidates,
        localized_candidates,
        final_features,
    ) = select_final_gene_features(
        gene_summary,
        branches=selected_branches,
        config=config.selection,
    )

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
        localized_trajectory_results=localized_results,
        gene_trajectory_summary=gene_summary,
        discrete_trajectory_candidates=discrete_candidates,
        continuous_trajectory_candidates=continuous_candidates,
        localized_trajectory_candidates=localized_candidates,
        final_features=final_features,
    )


# ============================================================================
# OUTPUT HELPERS
# ============================================================================


def save_isovita_pipeline_outputs(
    result: IsoVitaPipelineResult,
    output_dir: str | Path,
) -> None:
    """Write shared outputs plus results for branches that were actually run."""
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

    branch_tables = {
        "discrete": (
            "discrete_trajectory_results.csv",
            result.discrete_trajectory_results,
            "discrete_trajectory_candidates.csv",
            result.discrete_trajectory_candidates,
        ),
        "continuous": (
            "continuous_trajectory_results.csv",
            result.continuous_trajectory_results,
            "continuous_trajectory_candidates.csv",
            result.continuous_trajectory_candidates,
        ),
        "localized": (
            "localized_trajectory_results.csv",
            result.localized_trajectory_results,
            "localized_trajectory_candidates.csv",
            result.localized_trajectory_candidates,
        ),
    }

    for branch in result.selected_branches:
        results_name, results_df, candidates_name, candidates_df = branch_tables[branch]
        tables[results_name] = results_df
        tables[candidates_name] = candidates_df

    for filename, table in tables.items():
        table.to_csv(
            output_dir / filename,
            index=(filename == "balance_feature_matrix.csv"),
        )

    (output_dir / "selected_branches.txt").write_text(
        "\n".join(result.selected_branches) + "\n"
    )

