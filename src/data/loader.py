"""Dataset loading and preview utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union
import pandas as pd


SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"


def load_dataset(file_source: Union[str, Path, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """Load dataset from a file path, file-like object, or buffer.

    Supports CSV and Excel formats.
    """
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ext = path.suffix.lower()
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path)

    # For Streamlit UploadedFile or BytesIO / StringIO
    filename = getattr(file_source, "name", "file.csv").lower()
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_source)
    
    # Try reading as CSV; fall back to trying encoding/delimiters
    try:
        return pd.read_csv(file_source)
    except Exception:
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        return pd.read_csv(file_source, encoding="latin1")


def load_sample_dataset(name: str = "telco") -> pd.DataFrame:
    """Load a bundled sample dataset by name ('telco' or 'bank')."""
    sample_files = {
        "telco": SAMPLE_DATA_DIR / "telco_churn.csv",
        "bank": SAMPLE_DATA_DIR / "bank_churn.csv",
    }
    target_path = sample_files.get(name.lower())
    if not target_path or not target_path.exists():
        raise FileNotFoundError(
            f"Sample dataset '{name}' not found at {target_path}. "
            f"Available: {list(sample_files.keys())}"
        )
    return pd.read_csv(target_path)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Compute high-level summary metadata for a dataframe."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "cols_with_missing": int((df.isna().sum() > 0).sum()),
    }


def preview_data(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Return top N rows preview."""
    return df.head(n)

