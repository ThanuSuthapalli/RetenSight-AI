"""Data validation, schema profiling, and leakage detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ColumnTypeProfile:
    """Detected column type breakdown."""
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    high_cardinality_cols: List[str] = field(default_factory=list)
    binary_cols: List[str] = field(default_factory=list)
    constant_cols: List[str] = field(default_factory=list)
    high_missing_cols: List[str] = field(default_factory=list)
    suspected_id_cols: List[str] = field(default_factory=list)
    suspected_leakage_cols: List[str] = field(default_factory=list)
    cleaned_dtypes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Outcome of dataset validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    target_col: Optional[str] = None
    positive_label: Optional[Any] = None
    negative_label: Optional[Any] = None
    id_col: Optional[str] = None
    profile: Optional[ColumnTypeProfile] = None


def sanitize_numeric_strings(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Check if an object series is mostly numeric (e.g., TotalCharges with blank spaces).

    Returns (converted_series, was_converted).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series, False

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        # Strip whitespace and test conversion
        stripped = series.astype(str).str.strip()
        # Replace empty strings with NaN
        cleaned = stripped.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "?": np.nan})
        converted = pd.to_numeric(cleaned, errors="coerce")
        # If at least 85% of non-null values converted cleanly to numbers, treat as numeric
        valid_orig_count = (cleaned.notna() & (cleaned != "")).sum()
        if valid_orig_count > 0:
            converted_valid_count = converted.notna().sum()
            if (converted_valid_count / valid_orig_count) >= 0.85:
                return converted, True

    return series, False


def profile_columns(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    high_cardinality_threshold: int = 20,
    missing_rate_threshold: float = 0.60,
) -> ColumnTypeProfile:
    """Analyze dataframe columns and categorize them."""
    profile = ColumnTypeProfile()
    n_rows = len(df)
    
    leakage_keywords = [
        "churn", "cancel", "cancellation", "exit", "leaving", "terminated",
        "disconnect", "end_date", "termination_date"
    ]

    for col in df.columns:
        if col == target_col or col == id_col:
            continue

        series = df[col]
        missing_rate = series.isna().mean()
        n_unique = series.nunique(dropna=True)

        # 1. Constant columns
        if n_unique <= 1:
            profile.constant_cols.append(col)
            continue

        # 2. High missing columns
        if missing_rate >= missing_rate_threshold:
            profile.high_missing_cols.append(col)

        # 3. Leakage keywords detection
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in leakage_keywords):
            profile.suspected_leakage_cols.append(col)

        # 4. Suspected ID columns (high uniqueness text/int)
        if n_unique / max(n_rows, 1) >= 0.95 and ("id" in col_lower or series.dtype == object):
            profile.suspected_id_cols.append(col)

        # 5. Check if numeric or object-convertible numeric
        cleaned_series, was_num = sanitize_numeric_strings(series)
        if was_num or pd.api.types.is_numeric_dtype(cleaned_series):
            # Check if actually binary 0/1 numeric
            unique_vals = set(cleaned_series.dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0}) and n_unique == 2:
                profile.binary_cols.append(col)
                profile.cleaned_dtypes[col] = "binary"
            else:
                profile.numeric_cols.append(col)
                profile.cleaned_dtypes[col] = "numeric"
            continue

        # 6. Categoricals / Binary
        if n_unique == 2:
            profile.binary_cols.append(col)
            profile.cleaned_dtypes[col] = "binary"
        elif n_unique > high_cardinality_threshold:
            profile.high_cardinality_cols.append(col)
            profile.cleaned_dtypes[col] = "high_cardinality"
        else:
            profile.categorical_cols.append(col)
            profile.cleaned_dtypes[col] = "categorical"

    return profile


def validate_dataset(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Any = None,
    id_col: Optional[str] = None,
    min_rows: int = 100,
    min_cols: int = 4,
) -> ValidationResult:
    """Validate dataset structure, target column, and flag risks."""
    errors: List[str] = []
    warnings: List[str] = []

    # Basic shape checks
    if df is None or df.empty:
        return ValidationResult(is_valid=False, errors=["Uploaded dataset is empty."])

    if len(df) < min_rows:
        warnings.append(
            f"Dataset has {len(df)} rows. Recommended minimum is {min_rows} rows for stable ML evaluation."
        )

    if df.shape[1] < min_cols:
        errors.append(
            f"Dataset has only {df.shape[1]} columns. At least {min_cols} columns are required."
        )

    # Target column validation
    if not target_col or target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found in dataset columns.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    target_series = df[target_col].dropna()
    unique_targets = list(pd.unique(target_series))

    if len(unique_targets) != 2:
        errors.append(
            f"Target column '{target_col}' must be binary (found {len(unique_targets)} classes: {unique_targets})."
        )
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    # Determine positive and negative classes
    resolved_pos = positive_label
    resolved_neg = None

    if resolved_pos is None:
        # Heuristic: looks for 'yes', '1', 'true', 'churn'
        for val in unique_targets:
            if str(val).strip().lower() in ("yes", "1", "true", "churn", "churned", "positive", "y"):
                resolved_pos = val
                break
        if resolved_pos is None:
            resolved_pos = unique_targets[1]

    neg_candidates = [val for val in unique_targets if str(val) != str(resolved_pos)]
    resolved_neg = neg_candidates[0] if neg_candidates else unique_targets[0]

    # Target class balance check
    pos_count = (target_series == resolved_pos).sum()
    pos_rate = pos_count / len(target_series)
    if pos_rate < 0.05 or pos_rate > 0.95:
        warnings.append(
            f"Extreme target imbalance detected: positive class '{resolved_pos}' accounts for "
            f"{pos_rate * 100:.1f}% of data. Model recall may be impacted."
        )

    # Profile remaining columns
    profile = profile_columns(df, target_col=target_col, id_col=id_col)

    if profile.suspected_leakage_cols:
        warnings.append(
            f"Potential target leakage in columns: {profile.suspected_leakage_cols}. "
            "Consider excluding them from features."
        )

    if profile.constant_cols:
        warnings.append(
            f"Constant columns detected (will be ignored): {profile.constant_cols}."
        )

    if profile.high_missing_cols:
        warnings.append(
            f"Columns with >60% missing values: {profile.high_missing_cols}."
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        target_col=target_col,
        positive_label=resolved_pos,
        negative_label=resolved_neg,
        id_col=id_col,
        profile=profile,
    )


def clean_and_prepare_df(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Any,
    id_col: Optional[str] = None,
    drop_high_missing: bool = True,
    missing_rate_threshold: float = 0.60,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Clean raw dataframe: fix dtypes, map target to binary 0/1, separate X and y.

    Returns (X, y, cleaning_report).
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "initial_cols": df.shape[1],
        "dropped_duplicates": 0,
        "dropped_columns": [],
        "imputed_numeric_type_conversions": [],
    }

    # Make working copy
    df_clean = df.copy()

    # Drop exact duplicates
    n_dups = int(df_clean.duplicated().sum())
    if n_dups > 0:
        df_clean = df_clean.drop_duplicates()
        report["dropped_duplicates"] = n_dups

    # Clean target
    df_clean = df_clean.dropna(subset=[target_col])
    y = (df_clean[target_col].astype(str).str.strip() == str(positive_label).strip()).astype(int)

    # Columns to drop
    cols_to_drop = [target_col]
    if id_col and id_col in df_clean.columns:
        cols_to_drop.append(id_col)

    # Convert numeric strings (e.g. TotalCharges)
    for col in df_clean.columns:
        if col in cols_to_drop:
            continue
        cleaned_series, was_converted = sanitize_numeric_strings(df_clean[col])
        if was_converted:
            df_clean[col] = cleaned_series
            report["imputed_numeric_type_conversions"].append(col)

    # Drop constant columns
    for col in df_clean.columns:
        if col in cols_to_drop:
            continue
        if df_clean[col].nunique(dropna=True) <= 1:
            cols_to_drop.append(col)

    # Drop high missing columns if requested
    if drop_high_missing:
        for col in df_clean.columns:
            if col in cols_to_drop:
                continue
            if df_clean[col].isna().mean() >= missing_rate_threshold:
                cols_to_drop.append(col)

    report["dropped_columns"] = list(set(cols_to_drop) - {target_col})
    X = df_clean.drop(columns=cols_to_drop, errors="ignore")

    report["final_features"] = list(X.columns)
    report["final_rows"] = len(X)
    return X, y, report

