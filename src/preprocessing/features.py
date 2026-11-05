"""Domain-specific feature engineering transformers."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates customer intelligence features if relevant columns exist.

    Features generated (when source columns exist):
    - tenure_bucket: Binned customer tenure (0-12, 13-24, 25-48, 49+ months)
    - avg_monthly_spend: TotalCharges / max(tenure, 1)
    - monthly_charge_delta: MonthlyCharges - avg_monthly_spend
    - charge_to_tenure_ratio: MonthlyCharges / (tenure + 1)
    - num_addon_services: Count of active digital/support services
    - contract_payment_combo: Interaction between Contract and PaymentMethod
    """

    ADDON_CANDIDATES = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def __init__(self, enable_feature_engineering: bool = True):
        self.enable_feature_engineering = enable_feature_engineering
        self.created_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enable_feature_engineering:
            return X.copy()

        df = X.copy()
        created: List[str] = []

        # Find columns (case-insensitive search)
        cols_lower = {str(c).lower(): c for c in df.columns}
        tenure_col = cols_lower.get("tenure")
        monthly_col = cols_lower.get("monthlycharges") or cols_lower.get("monthly_charges")
        total_col = cols_lower.get("totalcharges") or cols_lower.get("total_charges")
        contract_col = cols_lower.get("contract")
        payment_col = cols_lower.get("paymentmethod") or cols_lower.get("payment_method")

        # 1. Tenure Buckets
        if tenure_col is not None:
            t = pd.to_numeric(df[tenure_col], errors="coerce").fillna(0)
            bins = [-1, 12, 24, 48, 1000]
            labels = ["0-12m", "13-24m", "25-48m", "49m+"]
            df["tenure_bucket"] = pd.cut(t, bins=bins, labels=labels).astype(str)
            created.append("tenure_bucket")

        # 2. Average Monthly Spend & Deltas
        if tenure_col is not None and total_col is not None:
            t = pd.to_numeric(df[tenure_col], errors="coerce").fillna(0).clip(lower=1)
            tot = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
            avg_spend = tot / t
            df["avg_monthly_spend"] = avg_spend
            created.append("avg_monthly_spend")

            if monthly_col is not None:
                m = pd.to_numeric(df[monthly_col], errors="coerce").fillna(0)
                df["monthly_charge_delta"] = m - avg_spend
                df["charge_to_tenure_ratio"] = m / (t + 1)
                created.extend(["monthly_charge_delta", "charge_to_tenure_ratio"])

        elif tenure_col is not None and monthly_col is not None:
            t = pd.to_numeric(df[tenure_col], errors="coerce").fillna(0)
            m = pd.to_numeric(df[monthly_col], errors="coerce").fillna(0)
            df["charge_to_tenure_ratio"] = m / (t + 1)
            created.append("charge_to_tenure_ratio")

        # 3. Add-on Services Count
        addon_cols_present = [c for c in self.ADDON_CANDIDATES if c in df.columns]
        if addon_cols_present:
            addon_count = np.zeros(len(df), dtype=int)
            for c in addon_cols_present:
                addon_count += (df[c].astype(str).str.strip().str.lower() == "yes").astype(int)
            df["num_addon_services"] = addon_count
            created.append("num_addon_services")

        # 4. Contract x Payment Method interaction
        if contract_col is not None and payment_col is not None:
            df["contract_payment_combo"] = (
                df[contract_col].astype(str) + " & " + df[payment_col].astype(str)
            )
            created.append("contract_payment_combo")

        self.created_features_ = created
        return df

