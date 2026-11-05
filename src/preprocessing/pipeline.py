"""Leak-free Scikit-Learn preprocessing pipeline builder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from src.preprocessing.features import DomainFeatureEngineer


def identify_pipeline_columns(
    df: pd.DataFrame,
    high_cardinality_threshold: int = 25,
) -> Tuple[List[str], List[str], List[str]]:
    """Identify numeric, low-cardinality categorical, and high-cardinality columns.

    Returns (numeric_cols, categorical_cols, dropped_high_cardinality).
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    dropped_high_cardinality: List[str] = []

    for col in df.columns:
        series = df[col]
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            # If purely 0 and 1, still safe to treat as numeric or binary
            numeric_cols.append(col)
        else:
            # Object / categorical
            n_unique = series.nunique(dropna=True)
            if n_unique > high_cardinality_threshold:
                dropped_high_cardinality.append(col)
            else:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols, dropped_high_cardinality


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_imputer_strategy: str = "median",
    numeric_scaler: str = "standard",  # 'standard', 'minmax', or 'none'
    categorical_imputer_strategy: str = "most_frequent",
) -> ColumnTransformer:
    """Build a Scikit-Learn ColumnTransformer for numeric and categorical columns."""
    transformers = []

    # Numeric sub-pipeline
    if numeric_cols:
        num_steps = [("imputer", SimpleImputer(strategy=numeric_imputer_strategy))]
        if numeric_scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif numeric_scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        # If 'none', no scaling step
        transformers.append(("num", Pipeline(num_steps), numeric_cols))

    # Categorical sub-pipeline
    if categorical_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy=categorical_imputer_strategy, fill_value="Unknown")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


class FullDataPreprocessor:
    """End-to-end data preprocessor wrapping feature engineering and column transformations.

    Ensures transformers are fit strictly during .fit() without leaking test data.
    """

    def __init__(
        self,
        enable_feature_engineering: bool = True,
        numeric_scaler: str = "standard",
        numeric_imputer: str = "median",
        categorical_imputer: str = "most_frequent",
        high_cardinality_threshold: int = 25,
    ):
        self.enable_feature_engineering = enable_feature_engineering
        self.numeric_scaler = numeric_scaler
        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer
        self.high_cardinality_threshold = high_cardinality_threshold

        self.feature_engineer_ = DomainFeatureEngineer(
            enable_feature_engineering=self.enable_feature_engineering
        )
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.dropped_high_cardinality_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> FullDataPreprocessor:
        # Step 1: Fit feature engineer and transform training slice
        self.feature_engineer_.fit(X, y)
        X_engineered = self.feature_engineer_.transform(X)

        # Step 2: Dynamically detect column types on engineered dataset
        num_cols, cat_cols, dropped_high = identify_pipeline_columns(
            X_engineered,
            high_cardinality_threshold=self.high_cardinality_threshold,
        )
        self.numeric_cols_ = num_cols
        self.categorical_cols_ = cat_cols
        self.dropped_high_cardinality_ = dropped_high

        # Step 3: Build and fit ColumnTransformer strictly on X_engineered
        self.preprocessor_ = build_preprocessor(
            numeric_cols=self.numeric_cols_,
            categorical_cols=self.categorical_cols_,
            numeric_imputer_strategy=self.numeric_imputer,
            numeric_scaler=self.numeric_scaler,
            categorical_imputer_strategy=self.categorical_imputer,
        )
        self.preprocessor_.fit(X_engineered)

        # Step 4: Extract feature names out
        try:
            self.feature_names_out_ = list(self.preprocessor_.get_feature_names_out())
        except Exception:
            # Fallback if names cannot be retrieved
            self.feature_names_out_ = [f"f_{i}" for i in range(self.transform(X).shape[1])]

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        X_engineered = self.feature_engineer_.transform(X)
        return self.preprocessor_.transform(X_engineered)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_out_

    def get_summary_report(self) -> Dict[str, Any]:
        """Summary of transformation steps and feature counts."""
        return {
            "engineered_features_added": getattr(self.feature_engineer_, "created_features_", []),
            "numeric_columns_count": len(self.numeric_cols_),
            "numeric_columns": self.numeric_cols_,
            "categorical_columns_count": len(self.categorical_cols_),
            "categorical_columns": self.categorical_cols_,
            "dropped_high_cardinality": self.dropped_high_cardinality_,
            "final_feature_dimension": len(self.feature_names_out_),
            "feature_names_sample": self.feature_names_out_[:15],
        }

