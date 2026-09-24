"""
prepare_isovita_inputs.py
=========================

Dataset-level preparation for the core IsoVita algorithm.

This module owns everything needed to turn raw transcript abundance plus
metadata and a transcript-to-gene mapping into the canonical inputs expected by
``isovita.run_isovita_pipeline()``. The core algorithm therefore starts from a
fully harmonized, gene-annotated TPM matrix rather than repeating dataset
preparation logic.

Preparation responsibilities
----------------------------
1. Load/filter sample metadata, including the postnatal restriction by default.
2. Extract a transcript x sample TPM matrix from AnnData (or accept one directly).
3. Parse and harmonize sample IDs.
4. Normalize and merge the transcript-to-gene mapping.
5. Restrict/reorder TPM columns to exactly the retained metadata samples.
6. Validate the canonical IsoVita input contract.
7. Save ``tpm.tsv``, ``metadata.tsv``, and a preparation summary.

Canonical output contract
-------------------------
``tpm`` contains ``gene_id``, ``transcript_id``, then one column per retained
sample. ``metadata`` contains canonical ``sample_id`` and numeric ``age`` and is
ordered identically to the TPM sample columns.

This module deliberately does not run IsoPlex, construct the one-feature-per-gene
balance, or run discrete/continuous/localized tests. Those are core IsoVita
operations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

try:
    import anndata as ad
except ImportError:  # Optional unless the AnnData adapter/CLI is used.
    ad = None

import numpy as np
import pandas as pd


# ============================================================================
# RESULT CONTAINER
# ============================================================================


@dataclass
class PreparedIsoVitaInputs:
    """Canonical TPM/metadata objects ready for ``run_isovita_pipeline()``."""

    tpm: pd.DataFrame
    metadata: pd.DataFrame
    summary: pd.DataFrame


# ============================================================================
# GENERIC I/O HELPERS
# ============================================================================


def read_table(
    path: str | Path,
    *,
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """Read a delimited table, inferring tab for .tsv/.txt and comma otherwise."""
    path = Path(path)

    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","

    return pd.read_csv(path, sep=sep)


def normalize_separator(value: str) -> str:
    """Convert CLI-friendly separator names to actual delimiter strings."""
    mapping = {
        "tab": "\t",
        "tsv": "\t",
        "\\t": "\t",
        "comma": ",",
        "csv": ",",
        ",": ",",
    }

    if value not in mapping:
        raise ValueError(
            f"Unsupported separator {value!r}. "
            "Use one of: tab, tsv, comma, csv."
        )

    return mapping[value]


# ============================================================================
# METADATA PREPARATION
# ============================================================================


def apply_metadata_filters(
    metadata: pd.DataFrame,
    filters: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Apply exact-match cohort filters to metadata."""
    out = metadata.copy()

    if not filters:
        return out

    for column, value in filters.items():
        if column not in out.columns:
            raise ValueError(f"Metadata filter column {column!r} is missing.")

        out = out.loc[out[column].astype(str).eq(str(value))].copy()

    return out


def prepare_metadata(
    metadata: pd.DataFrame,
    *,
    sample_col: str,
    age_col: str,
    filters: Optional[Mapping[str, object]] = None,
    canonical_sample_col: str = "sample_id",
    postnatal_only: bool = True,
) -> pd.DataFrame:
    """Filter and standardize metadata before it reaches the IsoVita core."""
    meta = apply_metadata_filters(metadata, filters)

    required = {sample_col, age_col}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    meta = meta.copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta[age_col] = pd.to_numeric(meta[age_col], errors="coerce")
    meta = meta.dropna(subset=[sample_col, age_col]).copy()

    if postnatal_only:
        meta = meta.loc[meta[age_col] >= 0].copy()

    if meta.empty:
        raise ValueError("No metadata rows remain after the requested filters.")

    if canonical_sample_col != sample_col:
        meta[canonical_sample_col] = meta[sample_col].astype(str)

    if age_col != "age":
        meta["age"] = meta[age_col]

    duplicate_mask = meta[canonical_sample_col].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = (
            meta.loc[duplicate_mask, canonical_sample_col]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Duplicate metadata sample IDs found after filtering. "
            f"Examples: {duplicate_ids[:10]}"
        )

    # Put the two canonical columns first while preserving useful dataset columns.
    first = [canonical_sample_col, "age"]
    remaining = [c for c in meta.columns if c not in first]
    return meta[first + remaining].reset_index(drop=True)


# ============================================================================
# SAMPLE-ID PARSING
# ============================================================================


def identity_sample_id(value: str) -> str:
    """Return the AnnData observation basename unchanged."""
    return Path(str(value)).name


def make_underscore_prefix_parser(n_parts: int = 2) -> Callable[[str], str]:
    """Build a parser retaining the first n underscore-delimited basename parts."""
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1.")

    def parser(value: str) -> str:
        basename = Path(str(value)).name
        parts = basename.split("_")

        if len(parts) < n_parts:
            raise ValueError(
                f"Cannot extract {n_parts} underscore-delimited parts from "
                f"sample name {value!r}."
            )

        return "_".join(parts[:n_parts])

    return parser


def resolve_sample_id_parser(
    mode: str,
    *,
    n_parts: int = 2,
) -> Callable[[str], str]:
    """Resolve a built-in sample-ID parser."""
    if mode == "identity":
        return identity_sample_id

    if mode == "underscore_prefix":
        return make_underscore_prefix_parser(n_parts)

    raise ValueError(
        f"Unknown sample_id_mode={mode!r}. "
        "Supported modes: identity, underscore_prefix."
    )


# ============================================================================
# EXPRESSION EXTRACTION
# ============================================================================


def extract_transcript_abundance(
    adata: ad.AnnData,
    *,
    abundance_key: str = "abundance",
    sample_id_parser: Optional[Callable[[str], str]] = None,
    transcript_id_col: str = "tx_id",
) -> pd.DataFrame:
    """
    Extract a transcript x sample abundance DataFrame from AnnData.obsm.

    Assumptions
    -----------
    - ``adata.obsm[abundance_key]`` has shape samples x transcripts.
    - ``adata.var_names`` identify transcripts.
    - ``adata.obs_names`` identify samples before optional parsing.
    """
    if abundance_key not in adata.obsm:
        raise ValueError(
            f"AnnData.obsm does not contain {abundance_key!r}. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    abundance = adata.obsm[abundance_key]

    if abundance.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            f"adata.obsm[{abundance_key!r}] has shape {abundance.shape}, "
            f"but expected ({adata.n_obs}, {adata.n_vars}) = "
            "(n_samples, n_transcripts)."
        )

    parser = sample_id_parser or identity_sample_id
    sample_ids = [str(parser(x)) for x in adata.obs_names]

    sample_index = pd.Index(sample_ids)
    if sample_index.has_duplicates:
        duplicated = sample_index[sample_index.duplicated()].unique().tolist()
        raise ValueError(
            "Parsed AnnData sample IDs are not unique. "
            f"Examples: {duplicated[:10]}"
        )

    transcript_ids = pd.Index(adata.var_names.astype(str))
    if transcript_ids.has_duplicates:
        duplicated = transcript_ids[transcript_ids.duplicated()].unique().tolist()
        raise ValueError(
            "AnnData transcript IDs are not unique. "
            f"Examples: {duplicated[:10]}"
        )

    if hasattr(abundance, "toarray"):
        abundance = abundance.toarray()

    expression = pd.DataFrame(
        np.asarray(abundance).T,
        index=transcript_ids,
        columns=sample_ids,
    )
    expression.index.name = transcript_id_col

    return expression.reset_index()


# ============================================================================
# SAMPLE HARMONIZATION
# ============================================================================


def match_expression_and_metadata(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    transcript_col: str = "tx_id",
    metadata_sample_col: str = "sample_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict expression and metadata to their shared samples.

    Metadata order is preserved and expression columns are reordered to match.
    """
    if transcript_col not in expression.columns:
        raise ValueError(
            f"Expression table is missing transcript column {transcript_col!r}."
        )

    if metadata_sample_col not in metadata.columns:
        raise ValueError(
            f"Metadata is missing sample column {metadata_sample_col!r}."
        )

    expr = expression.copy()
    rename_map = {
        column: str(column)
        for column in expr.columns
        if column != transcript_col
    }
    expr = expr.rename(columns=rename_map)

    expr_sample_cols = [column for column in expr.columns if column != transcript_col]
    if len(expr_sample_cols) != len(set(expr_sample_cols)):
        duplicated = pd.Index(expr_sample_cols)
        duplicated = duplicated[duplicated.duplicated()].unique().tolist()
        raise ValueError(
            "Expression sample columns are not unique after string conversion. "
            f"Examples: {duplicated[:10]}"
        )

    expr_samples = set(expr_sample_cols)
    meta = metadata.loc[
        metadata[metadata_sample_col].astype(str).isin(expr_samples)
    ].copy()

    if meta.empty:
        raise ValueError("No overlapping sample IDs between expression and metadata.")

    shared_ids = meta[metadata_sample_col].astype(str).tolist()
    expr = expr[[transcript_col] + shared_ids].copy()

    return expr, meta.reset_index(drop=True)


# ============================================================================
# TRANSCRIPT-TO-GENE MAPPING
# ============================================================================


def normalize_tx2gene(
    tx2gene: pd.DataFrame,
    *,
    transcript_col: Optional[str] = None,
    gene_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a transcript-to-gene table to columns ``tx_id`` / ``gene_id``."""
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
        raise ValueError(
            "Could not identify transcript and gene columns in tx2gene. "
            "Expected transcript column among tx_id/transcript_id/transcript "
            "and gene column among gene_id/gene, or supply names explicitly."
        )

    mapping = mapping[[transcript_col, gene_col]].copy()
    mapping = mapping.rename(
        columns={transcript_col: "tx_id", gene_col: "gene_id"}
    )
    mapping["tx_id"] = mapping["tx_id"].astype(str)
    mapping["gene_id"] = mapping["gene_id"].astype(str)

    if mapping["tx_id"].duplicated().any():
        duplicated = (
            mapping.loc[mapping["tx_id"].duplicated(keep=False), "tx_id"]
            .unique()
            .tolist()
        )
        raise ValueError(
            "tx2gene contains duplicate transcript IDs. "
            f"Examples: {duplicated[:10]}"
        )

    return mapping.reset_index(drop=True)


def annotate_tpm_with_genes(
    expression: pd.DataFrame,
    tx2gene: pd.DataFrame,
    *,
    transcript_col: str = "tx_id",
) -> pd.DataFrame:
    """Merge gene IDs into TPM and return canonical gene/transcript columns."""
    if transcript_col not in expression.columns:
        raise ValueError(
            f"Expression table is missing transcript column {transcript_col!r}."
        )

    expr = expression.copy()
    expr[transcript_col] = expr[transcript_col].astype(str)
    mapping = tx2gene[["tx_id", "gene_id"]].copy()

    merged = expr.merge(
        mapping,
        left_on=transcript_col,
        right_on="tx_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_mapping"),
    )

    if transcript_col == "tx_id":
        merged = merged.rename(columns={"tx_id": "transcript_id"})
    else:
        merged = merged.drop(columns=["tx_id"]).rename(
            columns={transcript_col: "transcript_id"}
        )

    missing_gene = merged["gene_id"].isna()
    if missing_gene.any():
        # Unmapped transcripts cannot be assigned to an IsoVita gene feature.
        merged = merged.loc[~missing_gene].copy()

    if merged.empty:
        raise ValueError("No TPM transcripts overlap the transcript-to-gene mapping.")

    merged["gene_id"] = merged["gene_id"].astype(str)
    merged["transcript_id"] = merged["transcript_id"].astype(str)
    sample_cols = [
        c for c in merged.columns
        if c not in {"gene_id", "transcript_id", "tx_id_mapping"}
    ]
    return merged[["gene_id", "transcript_id"] + sample_cols].copy()


# ============================================================================
# VALIDATION / SUMMARY
# ============================================================================


def validate_prepared_inputs(
    tpm: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    age_col: str = "age",
) -> None:
    """Validate the exact canonical contract consumed by the IsoVita core."""
    required_tpm = {"gene_id", "transcript_id"}
    missing_tpm = required_tpm - set(tpm.columns)
    if missing_tpm:
        raise ValueError(f"TPM missing canonical columns: {sorted(missing_tpm)}")
    if tpm["transcript_id"].duplicated().any():
        raise ValueError("TPM transcript_id values are not unique.")

    required_meta = {sample_col, age_col}
    missing_meta = required_meta - set(metadata.columns)
    if missing_meta:
        raise ValueError(
            f"Metadata missing required standardized columns: {sorted(missing_meta)}"
        )
    if metadata[sample_col].duplicated().any():
        raise ValueError("Metadata sample IDs are not unique.")

    age = pd.to_numeric(metadata[age_col], errors="coerce")
    if age.isna().any():
        raise ValueError("Standardized metadata age contains missing/non-numeric values.")

    sample_ids = metadata[sample_col].astype(str).tolist()
    tpm_sample_cols = [
        str(c) for c in tpm.columns if c not in {"gene_id", "transcript_id"}
    ]
    if tpm_sample_cols != sample_ids:
        raise ValueError(
            "TPM sample columns do not exactly match standardized metadata order."
        )

    numeric = tpm[tpm_sample_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("TPM contains missing/non-numeric sample values.")
    values = numeric.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("TPM contains non-finite sample values.")
    if np.any(values < 0):
        raise ValueError("TPM values must be non-negative.")


def build_preparation_summary(
    *,
    metadata: pd.DataFrame,
    tpm: pd.DataFrame,
    age_col: str = "age",
) -> pd.DataFrame:
    """Build a compact machine-readable summary of final core inputs."""
    age = pd.to_numeric(metadata[age_col], errors="coerce")
    rows = [
        ("retained_samples", int(len(metadata))),
        ("retained_transcripts", int(tpm["transcript_id"].nunique())),
        ("retained_genes", int(tpm["gene_id"].nunique())),
        ("age_min", float(age.min())),
        ("age_max", float(age.max())),
        ("n_negative_age", int((age < 0).sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# ============================================================================
# END-TO-END PREPARATION
# ============================================================================


def prepare_isovita_inputs_from_expression(
    *,
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    tx2gene: pd.DataFrame,
    metadata_sample_col: str,
    age_col: str,
    metadata_filters: Optional[Mapping[str, object]] = None,
    expression_transcript_col: str = "tx_id",
    tx2gene_transcript_col: Optional[str] = None,
    tx2gene_gene_col: Optional[str] = None,
    postnatal_only: bool = True,
) -> PreparedIsoVitaInputs:
    """Prepare canonical gene-annotated TPM and metadata from a table."""
    meta = prepare_metadata(
        metadata,
        sample_col=metadata_sample_col,
        age_col=age_col,
        filters=metadata_filters,
        canonical_sample_col="sample_id",
        postnatal_only=postnatal_only,
    )

    expr = expression.copy()
    if expression_transcript_col not in expr.columns:
        raise ValueError(
            f"Expression table is missing transcript column "
            f"{expression_transcript_col!r}."
        )
    if expression_transcript_col != "tx_id":
        expr = expr.rename(columns={expression_transcript_col: "tx_id"})
    expr["tx_id"] = expr["tx_id"].astype(str)
    if not expr["tx_id"].is_unique:
        raise ValueError("Expression transcript IDs must be unique.")

    expr, matched_meta = match_expression_and_metadata(
        expr,
        meta,
        transcript_col="tx_id",
        metadata_sample_col="sample_id",
    )

    mapping = normalize_tx2gene(
        tx2gene,
        transcript_col=tx2gene_transcript_col,
        gene_col=tx2gene_gene_col,
    )
    canonical_tpm = annotate_tpm_with_genes(expr, mapping, transcript_col="tx_id")

    validate_prepared_inputs(canonical_tpm, matched_meta)
    summary = build_preparation_summary(metadata=matched_meta, tpm=canonical_tpm)

    return PreparedIsoVitaInputs(
        tpm=canonical_tpm,
        metadata=matched_meta,
        summary=summary,
    )


def prepare_isovita_inputs(
    *,
    adata: ad.AnnData,
    metadata: pd.DataFrame,
    tx2gene: pd.DataFrame,
    metadata_sample_col: str,
    age_col: str,
    metadata_filters: Optional[Mapping[str, object]] = None,
    abundance_key: str = "abundance",
    sample_id_parser: Optional[Callable[[str], str]] = None,
    tx2gene_transcript_col: Optional[str] = None,
    tx2gene_gene_col: Optional[str] = None,
    postnatal_only: bool = True,
) -> PreparedIsoVitaInputs:
    """AnnData adapter for :func:`prepare_isovita_inputs_from_expression`."""
    expression = extract_transcript_abundance(
        adata,
        abundance_key=abundance_key,
        sample_id_parser=sample_id_parser,
        transcript_id_col="tx_id",
    )
    return prepare_isovita_inputs_from_expression(
        expression=expression,
        metadata=metadata,
        tx2gene=tx2gene,
        metadata_sample_col=metadata_sample_col,
        age_col=age_col,
        metadata_filters=metadata_filters,
        expression_transcript_col="tx_id",
        tx2gene_transcript_col=tx2gene_transcript_col,
        tx2gene_gene_col=tx2gene_gene_col,
        postnatal_only=postnatal_only,
    )


def save_prepared_inputs(
    result: PreparedIsoVitaInputs,
    output_dir: str | Path,
    *,
    sep: str = "\t",
) -> None:
    """Write exactly the canonical inputs consumed by ``run_isovita_pipeline``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.tpm.to_csv(output_dir / "tpm.tsv", sep=sep, index=False)
    result.metadata.to_csv(output_dir / "metadata.tsv", sep=sep, index=False)
    result.summary.to_csv(
        output_dir / "preparation_summary.tsv", sep=sep, index=False
    )


# ============================================================================
# CLI
# ============================================================================


def parse_filter_arguments(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeated ``--filter column=value`` CLI arguments."""
    filters: Dict[str, str] = {}

    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid filter {item!r}. Expected COLUMN=VALUE.")

        column, value = item.split("=", 1)
        column = column.strip()
        value = value.strip()

        if not column:
            raise ValueError(f"Invalid filter {item!r}: empty column name.")

        filters[column] = value

    return filters


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one standardized input set for IsoVita."
    )

    parser.add_argument(
        "--adata",
        required=True,
        help="Path to transcript-level AnnData .h5ad file.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to sample metadata table.",
    )
    parser.add_argument(
        "--tx2gene",
        required=True,
        help="Path to transcript-to-gene mapping table.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for prepared IsoVita inputs.",
    )

    parser.add_argument(
        "--metadata-sep",
        default="tab",
        choices=["tab", "tsv", "comma", "csv"],
        help="Metadata delimiter. Default: tab.",
    )
    parser.add_argument(
        "--tx2gene-sep",
        default="comma",
        choices=["tab", "tsv", "comma", "csv"],
        help="tx2gene delimiter. Default: comma.",
    )

    parser.add_argument(
        "--metadata-sample-col",
        default="sample_id",
        help="Metadata sample-ID column. Default: sample_id.",
    )
    parser.add_argument(
        "--age-col",
        default="age",
        help="Metadata age column. Default: age.",
    )

    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help=(
            "Exact-match metadata filter COLUMN=VALUE. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--abundance-key",
        default="abundance",
        help="AnnData.obsm key containing sample x transcript abundance.",
    )

    parser.add_argument(
        "--sample-id-mode",
        default="identity",
        choices=["identity", "underscore_prefix"],
        help=(
            "How to convert AnnData obs_names into metadata sample IDs. "
            "Default: identity."
        ),
    )
    parser.add_argument(
        "--sample-id-parts",
        type=int,
        default=2,
        help=(
            "Number of underscore-delimited basename components retained "
            "when --sample-id-mode=underscore_prefix. Default: 2."
        ),
    )

    parser.add_argument(
        "--tx2gene-transcript-col",
        default=None,
        help=(
            "Transcript column in tx2gene. If omitted, common names are "
            "auto-detected."
        ),
    )
    parser.add_argument(
        "--tx2gene-gene-col",
        default=None,
        help=(
            "Gene column in tx2gene. If omitted, common names are "
            "auto-detected."
        ),
    )

    parser.add_argument(
        "--include-prenatal",
        action="store_true",
        help=(
            "Retain negative ages. By default preparation restricts to age >= 0 "
            "so the core receives postnatal-only metadata."
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    metadata_sep = normalize_separator(args.metadata_sep)
    tx2gene_sep = normalize_separator(args.tx2gene_sep)

    metadata = read_table(args.metadata, sep=metadata_sep)
    tx2gene = read_table(args.tx2gene, sep=tx2gene_sep)
    if ad is None:
        raise ImportError(
            "The CLI/AnnData adapter requires the optional 'anndata' package."
        )
    adata = ad.read_h5ad(args.adata)

    filters = parse_filter_arguments(args.filter)
    sample_parser = resolve_sample_id_parser(
        args.sample_id_mode,
        n_parts=args.sample_id_parts,
    )

    result = prepare_isovita_inputs(
        adata=adata,
        metadata=metadata,
        tx2gene=tx2gene,
        metadata_sample_col=args.metadata_sample_col,
        age_col=args.age_col,
        metadata_filters=filters,
        abundance_key=args.abundance_key,
        sample_id_parser=sample_parser,
        tx2gene_transcript_col=args.tx2gene_transcript_col,
        tx2gene_gene_col=args.tx2gene_gene_col,
        postnatal_only=not args.include_prenatal,
    )

    save_prepared_inputs(result, args.output_dir)

    print("\nIsoVita input preparation complete.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print("\nPreparation summary:")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()