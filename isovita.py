"""
IsoVita
============

Unified temporal isoform-trajectory analysis with one feature per gene and
three parallel detection branches.

Shared representation
---------------------
1. Prepare postnatal transcript-level TPM data.
2. Run IsoPlex and identify effective isoforms.
3. Exclude genes with <2 effective isoforms. Every gene with >=2 effective
   isoforms follows the SAME condensation procedure:
      a. renormalize TPM within the effective-isoform set to a composition;
      b. CLR-transform the effective-isoform composition;
      c. fit one smooth CLR trajectory per effective isoform versus age;
      d. center fitted trajectories over a common age grid and use SVD to
         obtain the dominant temporal direction (PC1);
      e. exhaustively search A/B/unused isoform assignments and choose the
         simplest balance whose cosine similarity is near the optimum;
      f. define one gene feature as log(GM(Group A) / GM(Group B)).
   For K=2 effective isoforms the same method reduces to the unique two-part
   log-ratio; a direct computational shortcut is used, but it is not a
   separate statistical branch.

Parallel temporal detection
---------------------------
The same one-feature-per-gene matrix is passed independently to three branches:

1. Discrete branch
   Detect one unknown pure level shift over age. Candidate change points are
   boundaries between distinct observed ages that satisfy configurable minimum
   sample-size constraints on both sides. The best cut maximizes the nested
   constant-vs-step F statistic. Significance is obtained by permuting feature
   values across ages and repeating the ENTIRE change-point search, so the
   gene-level permutation p-value accounts for selecting the cut point. BH is
   then applied across genes.

2. Continuous branch
   Test a global spline(age) model against an intercept-only model, followed
   by BH across genes.

3. Localized branch
   Test spline(age) models within configured age windows, followed by BH
   across successfully tested gene-window hypotheses.

The three branches are parallel detectors, not mutually exclusive trajectory
classes. Their FDR-significant evidence is merged at the gene level into a
trajectory signature such as discrete, continuous+localized, or
 discrete+continuous+localized. The final feature list is the union of genes
selected by any branch.

SPIT
----
IsoVita has no SPIT dependency. The former SPIT wrapper has been removed.

Numerical / inferential notes
-----------------------------
* CLR and geometric-mean log-ratios require strictly positive compositions.
  A small configurable pseudocount is therefore added to effective-isoform
  fractions and the composition is renormalized. The default is 1e-8.
* "Near-optimal" balance is configurable. By default, any balance within
  0.01 absolute cosine similarity of the best balance is considered
  near-optimal; among these, the balance using the fewest isoforms is chosen,
  then the highest-cosine balance breaks remaining ties.
* Candidate balance directions depend only on K, not on the gene. For modest
  K they are cached and reused across genes.
* The balance itself is learned using age. Therefore, p/q-values from all
  three downstream branches are screening/characterization statistics when
  the same samples are used for balance learning and testing. The discrete
  permutation test corrects the within-branch search over change points, but
  it does NOT correct for age-informed balance construction. Formal
  confirmatory inference would require a pre-condensation test or sample
  splitting/cross-fitting.
* This module defines functions only; it does not execute the pipeline on
  import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path
import re
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
    """Pure-step change-point test settings for the discrete branch."""

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
class SelectionConfig:
    """Final union/ranking settings after the three branches are merged."""

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
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass
class IsoVitaPipelineResult:
    """Container for shared intermediates and all three temporal branches."""

    postnatal_meta: pd.DataFrame
    full_expression_df: pd.DataFrame
    expression_df: pd.DataFrame
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


def detect_sample_id_column(df: pd.DataFrame) -> str:
    for candidate in ("sample_id", "id", "sample", "ID", "SampleID"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find sample ID column. Expected one of: "
        "sample_id, id, sample, ID, SampleID."
    )


def detect_age_column(df: pd.DataFrame) -> str:
    for candidate in ("age", "Age", "AGE"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find age column in metadata.")


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
# INPUT / PREPROCESSING
# ============================================================================


def prepare_metadata(
    metadata: pd.DataFrame,
    *,
    sample_col: Optional[str] = None,
    age_col: Optional[str] = None,
    postnatal_only: bool = True,
) -> pd.DataFrame:
    """Standardize metadata to index=sample with numeric age."""
    meta = metadata.copy()

    sample_col = sample_col or detect_sample_id_column(meta)
    age_col = age_col or detect_age_column(meta)

    meta[age_col] = pd.to_numeric(meta[age_col], errors="coerce")
    meta = meta.dropna(subset=[sample_col, age_col]).copy()
    meta[sample_col] = meta[sample_col].astype(str)

    if postnatal_only:
        meta = meta.loc[meta[age_col] >= 0].copy()

    meta = meta.set_index(sample_col, drop=True)
    meta.index.name = "sample"
    meta = meta.rename(columns={age_col: "age"})
    return meta


def normalize_tx2gene_table(
    tx2gene: pd.DataFrame,
    *,
    transcript_col: Optional[str] = None,
    gene_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a transcript-to-gene table to tx_id / gene_id."""
    mapping = tx2gene.copy()

    if transcript_col is None:
        for candidate in ("tx_id", "transcript_id", "transcript"):
            if candidate in mapping.columns:
                transcript_col = candidate
                break

    if gene_col is None:
        for candidate in ("gene_id", "gene"):
            if candidate in mapping.columns:
                gene_col = candidate
                break

    if transcript_col is None or gene_col is None:
        raise ValueError("Could not identify transcript and gene columns in tx2gene.")

    mapping = mapping[[transcript_col, gene_col]].copy()
    mapping = mapping.rename(
        columns={transcript_col: "tx_id", gene_col: "gene_id"}
    )
    mapping["tx_id"] = mapping["tx_id"].astype(str)
    mapping["gene_id"] = mapping["gene_id"].astype(str)

    return mapping.drop_duplicates("tx_id").reset_index(drop=True)


def prepare_full_expression(
    tpm: pd.DataFrame,
    tx2gene: pd.DataFrame,
    postnatal_meta: pd.DataFrame,
    *,
    tx_col: str = "tx_id",
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the canonical unfiltered postnatal transcript-level TPM table."""
    if tx_col not in tpm.columns:
        raise ValueError(f"TPM table must contain '{tx_col}'.")

    mapping = normalize_tx2gene_table(tx2gene)
    expr = tpm.copy()
    expr[tx_col] = expr[tx_col].astype(str)

    expr = expr.merge(
        mapping,
        left_on=tx_col,
        right_on="tx_id",
        how="left",
        suffixes=("", "_mapping"),
    )

    if tx_col != "tx_id":
        expr = expr.drop(columns=["tx_id"]).rename(columns={tx_col: "tx_id"})

    expr = expr.dropna(subset=["gene_id"]).copy()

    all_sample_cols = [
        col for col in expr.columns
        if col not in {"gene_id", "tx_id", "tx_id_mapping"}
    ]
    meta_samples = set(postnatal_meta.index.astype(str))
    sample_cols = [col for col in all_sample_cols if str(col) in meta_samples]

    if not sample_cols:
        raise ValueError("No overlapping postnatal samples between TPM and metadata.")

    expr[sample_cols] = (
        expr[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    expr = expr.rename(columns={"tx_id": "transcript_id"})
    expr = expr[["gene_id", "transcript_id"] + sample_cols].copy()

    return expr, sample_cols


def prepare_isoplex_expression(
    tpm: pd.DataFrame,
    tx2gene: pd.DataFrame,
    postnatal_meta: pd.DataFrame,
    *,
    config: IsoPlexConfig = IsoPlexConfig(),
    tx_col: str = "tx_id",
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Merge TPM with gene IDs and apply gentle IsoPlex input filters."""
    if tx_col not in tpm.columns:
        raise ValueError(f"TPM table must contain '{tx_col}'.")

    mapping = normalize_tx2gene_table(tx2gene)
    expr = tpm.copy()
    expr[tx_col] = expr[tx_col].astype(str)

    expr = expr.merge(
        mapping,
        left_on=tx_col,
        right_on="tx_id",
        how="left",
        suffixes=("", "_mapping"),
    )

    if tx_col != "tx_id":
        expr = expr.drop(columns=["tx_id"]).rename(columns={tx_col: "tx_id"})

    expr = expr.dropna(subset=["gene_id"]).copy()

    all_sample_cols = [
        col for col in expr.columns
        if col not in {"gene_id", "tx_id", "tx_id_mapping"}
    ]

    meta_samples = set(postnatal_meta.index.astype(str))
    sample_cols = [col for col in all_sample_cols if str(col) in meta_samples]

    if not sample_cols:
        raise ValueError("No overlapping postnatal samples between TPM and metadata.")

    expr[sample_cols] = (
        expr[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    summary_rows = [{
        "step": "raw_merged",
        "n_transcripts": int(expr["tx_id"].nunique()),
        "n_genes": int(expr["gene_id"].nunique()),
    }]

    keep_tx = (
        (expr[sample_cols] > config.tpm_threshold).sum(axis=1)
        >= config.min_samples_above_tpm
    )
    expr = expr.loc[keep_tx].copy()

    summary_rows.append({
        "step": "transcript_filter",
        "n_transcripts": int(expr["tx_id"].nunique()),
        "n_genes": int(expr["gene_id"].nunique()),
    })

    if config.require_positive_gene_total_every_sample:
        gene_total = expr.groupby("gene_id")[sample_cols].transform("sum")
        keep_gene = (gene_total > 0).all(axis=1)
        expr = expr.loc[keep_gene].copy()

    summary_rows.append({
        "step": "gene_total_filter",
        "n_transcripts": int(expr["tx_id"].nunique()),
        "n_genes": int(expr["gene_id"].nunique()),
    })

    expr = expr.rename(columns={"tx_id": "transcript_id"})
    expr = expr[["gene_id", "transcript_id"] + sample_cols].copy()

    return expr, sample_cols, pd.DataFrame(summary_rows)


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
    composition = np.asarray(prepared["composition"], dtype=float)

    try:
        clr = clr_transform(composition)
        age_grid, fitted_grid, centered_grid = _fit_clr_trajectories(
            ages,
            clr,
            spline_df=config.spline_df,
            spline_degree=config.spline_degree,
            age_grid_size=config.age_grid_size,
        )

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


def _fit_discrete_step_test(
    age: np.ndarray,
    y: np.ndarray,
    *,
    config: DiscreteConfig,
) -> Dict[str, object]:
    """Fit and permutation-test one unknown pure step for a single gene."""
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
        "permutation_p_value": np.nan,
        "p_value": np.nan,
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

    null_max_f = _permuted_max_step_f_statistics(
        y_sorted,
        cut_positions,
        n_permutations=config.n_permutations,
        random_seed=config.random_seed,
    )
    # +1 correction gives a valid Monte Carlo p-value and prevents p=0.
    permutation_p = float(
        (1 + np.count_nonzero(null_max_f >= best_f))
        / (config.n_permutations + 1)
    )

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
        "permutation_p_value": permutation_p,
        "p_value": permutation_p,
    })
    return base


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
    """Run the continuous branch: one global spline(age) test per gene."""
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
    """Run the localized branch: spline(age) tests within configured age windows."""
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
# MERGE THE THREE PARALLEL BRANCHES
# ---------------------------------------------------------------------------


def _trajectory_signature(row: pd.Series) -> str:
    labels = []
    if bool(row.get("discrete_fdr_significant", False)):
        labels.append("discrete")
    if bool(row.get("continuous_fdr_significant", False)):
        labels.append("continuous")
    if bool(row.get("localized_fdr_significant", False)):
        labels.append("localized")
    return "+".join(labels) if labels else "not_selected"


def summarize_gene_trajectories(
    balance_summary: pd.DataFrame,
    discrete_results: pd.DataFrame,
    continuous_results: pd.DataFrame,
    localized_results: pd.DataFrame,
) -> pd.DataFrame:
    """Merge independent discrete, continuous, and localized evidence per gene."""
    ok_balance = balance_summary.loc[balance_summary["status"].eq("ok")].copy()
    if ok_balance.empty:
        return ok_balance

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
    else:
        discrete_keep = pd.DataFrame({"gene_id": ok_balance["gene_id"]})

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
    else:
        continuous_keep = pd.DataFrame({"gene_id": ok_balance["gene_id"]})

    if localized_results.empty:
        localized_agg = pd.DataFrame({"gene_id": ok_balance["gene_id"]})
    else:
        win = localized_results.copy()
        win["is_sig"] = win["fdr_significant"].fillna(False).astype(bool)

        agg_rows = []
        for gene_id, sub in win.groupby("gene_id", sort=False):
            sig = sub.loc[sub["is_sig"]].sort_values(
                ["q_value", "p_value"],
                na_position="last",
            )
            best = sub.sort_values(
                ["q_value", "p_value"],
                na_position="last",
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
        localized_agg = pd.DataFrame(agg_rows)

    summary = ok_balance.merge(discrete_keep, on="gene_id", how="left")
    summary = summary.merge(continuous_keep, on="gene_id", how="left")
    summary = summary.merge(localized_agg, on="gene_id", how="left")

    for col in (
        "discrete_fdr_significant",
        "continuous_fdr_significant",
        "localized_fdr_significant",
    ):
        if col not in summary.columns:
            summary[col] = False
        summary[col] = summary[col].fillna(False).astype(bool)

    if "n_significant_windows" not in summary.columns:
        summary["n_significant_windows"] = 0
    summary["n_significant_windows"] = (
        summary["n_significant_windows"].fillna(0).astype(int)
    )

    summary["n_detection_branches"] = (
        summary[
            [
                "discrete_fdr_significant",
                "continuous_fdr_significant",
                "localized_fdr_significant",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )
    summary["trajectory_signature"] = summary.apply(
        _trajectory_signature,
        axis=1,
    )

    q_cols = [
        c for c in (
            "discrete_q_value",
            "continuous_q_value",
            "best_window_q_value",
        )
        if c in summary.columns
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
    config: SelectionConfig = SelectionConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return branch-specific candidate tables plus their unique union."""
    if gene_summary.empty:
        empty = gene_summary.copy()
        return empty, empty, empty, empty

    discrete = gene_summary.loc[
        gene_summary["discrete_fdr_significant"].fillna(False)
    ].copy()
    continuous = gene_summary.loc[
        gene_summary["continuous_fdr_significant"].fillna(False)
    ].copy()
    localized = gene_summary.loc[
        gene_summary["localized_fdr_significant"].fillna(False)
    ].copy()

    selected = gene_summary.loc[
        gene_summary["n_detection_branches"] > 0
    ].copy()
    selected = selected.sort_values(
        ["n_detection_branches", "min_branch_q_value", "gene_id"],
        ascending=[False, True, True],
        na_position="last",
    )

    if config.top_n_genes is not None:
        selected = selected.head(int(config.top_n_genes)).copy()

    selected.insert(0, "rank_gene", np.arange(1, len(selected) + 1))

    return (
        discrete.reset_index(drop=True),
        continuous.reset_index(drop=True),
        localized.reset_index(drop=True),
        selected.reset_index(drop=True),
    )


# ============================================================================
# END-TO-END ORCHESTRATOR
# ============================================================================


def run_isovita_pipeline(
    *,
    tpm: pd.DataFrame,
    tx2gene: pd.DataFrame,
    metadata: pd.DataFrame,
    config: IsoVitaConfig = IsoVitaConfig(),
    sample_col: Optional[str] = None,
    age_col: Optional[str] = None,
    progress_every: Optional[int] = None,
) -> IsoVitaPipelineResult:
    """Run the unified three-branch IsoVita workflow end-to-end."""
    postnatal_meta = prepare_metadata(
        metadata,
        sample_col=sample_col,
        age_col=age_col,
        postnatal_only=True,
    )

    full_expression_df, full_sample_cols = prepare_full_expression(
        tpm,
        tx2gene,
        postnatal_meta,
    )

    (
        expression_df,
        sample_cols,
        isoplex_filter_summary,
    ) = prepare_isoplex_expression(
        tpm,
        tx2gene,
        postnatal_meta,
        config=config.isoplex,
    )

    if list(full_sample_cols) != list(sample_cols):
        raise ValueError(
            "Canonical and IsoPlex expression tables do not have matching "
            "postnatal sample columns."
        )

    postnatal_meta = postnatal_meta.loc[list(sample_cols)].copy()

    sample_metrics, global_metrics = run_isoplex(expression_df)
    gene_routes = route_genes_by_effective_isoform_count(global_metrics)

    # Balance construction uses only IsoPlex-effective isoforms, renormalized
    # within that effective set. full_expression_df supplies their unfiltered
    # TPM values while IsoPlex determines membership in the effective set.
    balance_summary, balance_loadings, balance_feature_matrix = (
        build_one_feature_per_gene(
            expression_df=full_expression_df,
            global_metrics=global_metrics,
            sample_cols=sample_cols,
            metadata=postnatal_meta,
            config=config.balance,
            progress_every=progress_every,
        )
    )

    # Three completely parallel detectors on the SAME gene-feature matrix.
    discrete_results = run_discrete_feature_tests(
        balance_feature_matrix,
        postnatal_meta,
        config=config.discrete,
        progress_every=progress_every,
    )

    continuous_results = run_continuous_feature_tests(
        balance_feature_matrix,
        postnatal_meta,
        config=config.feature_test,
        progress_every=progress_every,
    )

    localized_results = run_localized_feature_tests(
        balance_feature_matrix,
        postnatal_meta,
        windows=config.windows,
        config=config.feature_test,
        progress_every=progress_every,
    )

    gene_summary = summarize_gene_trajectories(
        balance_summary,
        discrete_results,
        continuous_results,
        localized_results,
    )

    (
        discrete_candidates,
        continuous_candidates,
        localized_candidates,
        final_features,
    ) = select_final_gene_features(
        gene_summary,
        config=config.selection,
    )

    return IsoVitaPipelineResult(
        postnatal_meta=postnatal_meta,
        full_expression_df=full_expression_df,
        expression_df=expression_df,
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
    """Write major shared, branch-specific, and merged pipeline tables to CSV."""
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
        "discrete_trajectory_results.csv": result.discrete_trajectory_results,
        "continuous_trajectory_results.csv": result.continuous_trajectory_results,
        "localized_trajectory_results.csv": result.localized_trajectory_results,
        "gene_trajectory_summary.csv": result.gene_trajectory_summary,
        "discrete_trajectory_candidates.csv": result.discrete_trajectory_candidates,
        "continuous_trajectory_candidates.csv": result.continuous_trajectory_candidates,
        "localized_trajectory_candidates.csv": result.localized_trajectory_candidates,
        "final_features.csv": result.final_features,
    }

    for filename, table in tables.items():
        table.to_csv(
            output_dir / filename,
            index=(filename == "balance_feature_matrix.csv"),
        )