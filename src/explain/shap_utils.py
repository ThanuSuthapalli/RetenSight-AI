"""SHAP explainability engine, caching, and plain-English risk translators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap


def build_shap_explainer(
    estimator: Any,
    X_background: np.ndarray,
    model_name: str = "",
) -> shap.Explainer:
    """Instantiate the optimal SHAP explainer based on model architecture."""
    name = (model_name or estimator.__class__.__name__).lower()

    # Subsample background for high-speed explanation (< 0.5s)
    if len(X_background) > 100:
        indices = np.random.RandomState(42).choice(len(X_background), 80, replace=False)
        bg = X_background[indices]
    else:
        bg = X_background

    if "random_forest" in name or "forest" in name or "xgboost" in name or "xgb" in name:
        try:
            return shap.TreeExplainer(estimator)
        except Exception:
            return shap.Explainer(estimator, bg)
    elif "logistic" in name or "linear" in name:
        try:
            return shap.LinearExplainer(estimator, bg)
        except Exception:
            return shap.Explainer(estimator, bg)
    else:
        return shap.Explainer(estimator, bg)


def compute_global_shap(
    explainer: shap.Explainer,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_samples: int = 150,
) -> Dict[str, Any]:
    """Compute global SHAP values, feature importance ranking, and beeswarm data."""
    if len(X_sample) > max_samples:
        indices = np.random.RandomState(42).choice(len(X_sample), max_samples, replace=False)
        X_eval = X_sample[indices]
    else:
        X_eval = X_sample

    raw_shap = explainer(X_eval)

    # Normalize values array across different SHAP versions and models
    if hasattr(raw_shap, "values"):
        vals = raw_shap.values
        # If binary classification returns (N, features, 2), pick class 1 (churn)
        if vals.ndim == 3 and vals.shape[2] == 2:
            vals = vals[:, :, 1]
    elif isinstance(raw_shap, list):
        vals = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    else:
        vals = np.asarray(raw_shap)

    # Mean absolute SHAP importance
    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    
    # Create ranked dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    return {
        "shap_values": vals,
        "X_eval": X_eval,
        "feature_names": feature_names,
        "importance_df": importance_df,
        "top_features": importance_df["Feature"].head(10).tolist(),
    }


def explain_single_customer(
    explainer: shap.Explainer,
    customer_vector_proc: np.ndarray,
    feature_names: List[str],
    customer_raw_dict: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Generate local SHAP explanation for a single customer instance."""
    # Ensure 2D shape (1, n_features)
    vec = np.atleast_2d(customer_vector_proc)
    explanation = explainer(vec)

    # Base value and shap values
    if hasattr(explanation, "values"):
        vals = explanation.values[0]
        base_val = float(explanation.base_values[0]) if hasattr(explanation, "base_values") else 0.0
        if vals.ndim == 2 and vals.shape[1] == 2:
            vals = vals[:, 1]
            base_val = (
                float(explanation.base_values[0][1])
                if hasattr(explanation, "base_values") and len(explanation.base_values[0]) > 1
                else base_val
            )
    else:
        vals = np.asarray(explanation)[0]
        base_val = 0.0

    df_contrib = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": vals,
        "Absolute_Impact": np.abs(vals),
        "Direction": np.where(vals > 0, "Increases Churn Risk", "Reduces Churn Risk"),
    }).sort_values(by="Absolute_Impact", ascending=False).reset_index(drop=True)

    # Top drivers
    risk_increasing = df_contrib[df_contrib["SHAP_Value"] > 0].head(top_k)
    risk_decreasing = df_contrib[df_contrib["SHAP_Value"] < 0].head(top_k)

    # Generate plain-English bullets
    bullet_reasons: List[str] = []
    for _, row in risk_increasing.iterrows():
        feat = row["Feature"]
        val = row["SHAP_Value"]
        bullet_reasons.append(f"• [Risk Up] **{feat}** pushes churn risk UP (+{val:.2f})")
    for _, row in risk_decreasing.iterrows():
        feat = row["Feature"]
        val = row["SHAP_Value"]
        bullet_reasons.append(f"• [Protects] **{feat}** helps PROTECT account ({val:.2f})")

    return {
        "base_value": base_val,
        "contributions_df": df_contrib,
        "risk_increasing": risk_increasing,
        "risk_decreasing": risk_decreasing,
        "bullet_reasons": bullet_reasons,
        "vals": vals,
    }
