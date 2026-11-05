"""Unit tests for data loader, validator, and cleaner."""

import io
import pandas as pd
import pytest

from src.data.loader import load_dataset, load_sample_dataset, get_dataset_summary
from src.data.validator import validate_dataset, clean_and_prepare_df, profile_columns


def test_load_sample_dataset_telco():
    df = load_sample_dataset("telco")
    assert not df.empty
    assert "Churn" in df.columns
    assert len(df) > 5000


def test_load_sample_dataset_bank():
    df = load_sample_dataset("bank")
    assert not df.empty
    assert "Exited" in df.columns
    assert len(df) >= 1000


def test_get_dataset_summary():
    df = pd.DataFrame({
        "A": [1, 2, 3, None],
        "B": ["x", "y", "x", "x"],
    })
    summary = get_dataset_summary(df)
    assert summary["n_rows"] == 4
    assert summary["n_cols"] == 2
    assert summary["total_missing_cells"] == 1


def test_validate_dataset_valid():
    df = pd.DataFrame({
        "customerID": ["C1", "C2", "C3", "C4", "C5"] * 100,
        "tenure": [1, 12, 24, 36, 48] * 100,
        "MonthlyCharges": [29.9, 45.0, 70.0, 85.0, 100.0] * 100,
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year"] * 100,
        "Churn": ["Yes", "No", "No", "Yes", "No"] * 100,
    })
    res = validate_dataset(df, target_col="Churn", positive_label="Yes", id_col="customerID")
    assert res.is_valid
    assert res.positive_label == "Yes"
    assert res.negative_label == "No"


def test_validate_dataset_invalid_target():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
    })
    res = validate_dataset(df, target_col="NonExistentTarget")
    assert not res.is_valid
    assert any("not found" in err.lower() for err in res.errors)


def test_clean_and_prepare_df():
    df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "tenure": [0, 5, 10, 20],
        "TotalCharges": [" ", "100.5", "250.0", "500.0"],
        "Churn": ["No", "Yes", "No", "Yes"],
    })
    X, y, report = clean_and_prepare_df(df, target_col="Churn", positive_label="Yes", id_col="customerID")
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert len(y) == 4
    assert y.tolist() == [0, 1, 0, 1]
    # Check numeric conversion for TotalCharges
    assert pd.api.types.is_numeric_dtype(X["TotalCharges"])
    assert pd.isna(X["TotalCharges"].iloc[0])

