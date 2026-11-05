"""Model training orchestrator, train/test splitting, and experiment runner."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.evaluate import compute_metrics_at_threshold, compare_models
from src.models.tune import tune_model
from src.preprocessing.pipeline import FullDataPreprocessor


AVAILABLE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]


def train_models_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    selected_models: Optional[List[str]] = None,
    enable_feature_engineering: bool = True,
    quick_mode: bool = True,
    class_imbalance: str = "balanced_weights",
    scoring_metric: str = "roc_auc",
    test_size: float = 0.20,
    random_state: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Train multiple models with leak-free preprocessing and cross-validation tuning.

    Returns comprehensive experiment bundle.
    """
    if selected_models is None:
        selected_models = AVAILABLE_MODELS.copy()

    start_total_time = time.time()

    if progress_callback:
        progress_callback(0.05, "Performing stratified train/test split...")
    else:
        print("  -> Performing stratified train/test split...", flush=True)

    # 1. Stratified Train / Test Split
    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    y_train = y_train_series.to_numpy().astype(int)
    y_test = y_test_series.to_numpy().astype(int)

    # 2. Fit Preprocessor STRICTLY on Training slice
    if progress_callback:
        progress_callback(0.15, "Fitting leak-free preprocessing pipeline on training data...")
    else:
        print("  -> Fitting leak-free preprocessing pipeline...", flush=True)

    preprocessor = FullDataPreprocessor(
        enable_feature_engineering=enable_feature_engineering,
    )
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)
    feature_names = preprocessor.get_feature_names_out()

    # 3. Train each model
    models_results: Dict[str, Any] = {}
    n_models = len(selected_models)

    for idx, model_name in enumerate(selected_models):
        step_base = 0.25 + (idx / max(n_models, 1)) * 0.65
        msg = f"Tuning and training {model_name} ({'Quick' if quick_mode else 'Thorough'} mode)..."
        if progress_callback:
            progress_callback(step_base, msg)
        else:
            print(f"  -> {msg}", flush=True)

        tune_res = tune_model(
            model_name=model_name,
            X_train=X_train_proc,
            y_train=y_train,
            quick_mode=quick_mode,
            class_imbalance=class_imbalance,
            scoring=scoring_metric,
        )

        estimator = tune_res["best_estimator"]

        # Predict probabilities on test set
        if hasattr(estimator, "predict_proba"):
            y_prob_test = estimator.predict_proba(X_test_proc)[:, 1]
            y_prob_train = estimator.predict_proba(X_train_proc)[:, 1]
        elif hasattr(estimator, "decision_function"):
            decision = estimator.decision_function(X_test_proc)
            y_prob_test = 1 / (1 + np.exp(-decision))
            decision_tr = estimator.decision_function(X_train_proc)
            y_prob_train = 1 / (1 + np.exp(-decision_tr))
        else:
            y_prob_test = estimator.predict(X_test_proc).astype(float)
            y_prob_train = estimator.predict(X_train_proc).astype(float)

        test_metrics = compute_metrics_at_threshold(y_test, y_prob_test, threshold=0.5)
        train_metrics = compute_metrics_at_threshold(y_train, y_prob_train, threshold=0.5)

        models_results[model_name] = {
            "estimator": estimator,
            "best_params": tune_res["best_params"],
            "cv_score_mean": tune_res["cv_score_mean"],
            "cv_score_std": tune_res["cv_score_std"],
            "train_time_sec": tune_res["train_time_sec"],
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "y_prob_test": y_prob_test,
            "y_prob_train": y_prob_train,
        }

    # 4. Compare models and pick best
    comparison_table = compare_models(models_results, X_test_proc, y_test, threshold=0.5)
    best_model_name = str(comparison_table.iloc[0]["Model"])

    if progress_callback:
        progress_callback(1.0, f"Training complete! Best model: {best_model_name}")

    total_duration = time.time() - start_total_time

    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "models": models_results,
        "comparison_table": comparison_table,
        "best_model_name": best_model_name,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "total_duration_sec": total_duration,
        "config": {
            "enable_feature_engineering": enable_feature_engineering,
            "quick_mode": quick_mode,
            "class_imbalance": class_imbalance,
            "scoring_metric": scoring_metric,
            "random_state": random_state,
        },
    }
