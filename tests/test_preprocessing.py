"""Unit tests for feature engineering and leak-free preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features import DomainFeatureEngineer
from src.preprocessing.pipeline import FullDataPreprocessor, identify_pipeline_columns


def test_domain_feature_engineer():
    df = pd.DataFrame({
        "tenure": [1, 15, 30, 60],
        "MonthlyCharges": [50.0, 70.0, 80.0, 90.0],
        "TotalCharges": [50.0, 1050.0, 2400.0, 5400.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        "OnlineSecurity": ["Yes", "No", "Yes", "Yes"],
        "TechSupport": ["No", "Yes", "No", "Yes"],
    })

    fe = DomainFeatureEngineer(enable_feature_engineering=True)
    out = fe.fit_transform(df)

    assert "tenure_bucket" in out.columns
    assert "avg_monthly_spend" in out.columns
    assert "monthly_charge_delta" in out.columns
    assert "num_addon_services" in out.columns
    assert "contract_payment_combo" in out.columns
    assert out["num_addon_services"].tolist() == [1, 1, 1, 2]


def test_full_data_preprocessor_anti_leakage():
    # Verify that preprocessor fit on train slice does not mutate based on test slice
    X_train = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })
    X_test = pd.DataFrame({
        "tenure": [5, 15],
        "MonthlyCharges": [40.0, 80.0],
        "Contract": ["Month-to-month", "UnknownContract"],  # New unseen category
    })

    prep = FullDataPreprocessor(enable_feature_engineering=True)
    X_train_proc = prep.fit_transform(X_train)
    X_test_proc = prep.transform(X_test)

    # Check output shapes are consistent
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert len(X_train_proc) == 3
    assert len(X_test_proc) == 2
    assert not np.isnan(X_test_proc).any()


def test_identify_pipeline_columns():
    df = pd.DataFrame({
        "numeric_1": [1.0, 2.0] * 15,
        "numeric_2": [10, 20] * 15,
        "cat_low": ["A", "B"] * 15,
        "cat_high": [f"val_{i}" for i in range(30)],
    })
    nums, cats, dropped = identify_pipeline_columns(df, high_cardinality_threshold=10)
    assert "numeric_1" in nums
    assert "numeric_2" in nums
    assert "cat_low" in cats
    assert "cat_high" in dropped
