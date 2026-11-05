"""Hyperparameter search spaces and cross-validated tuning routines."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


def get_search_spaces(
    model_name: str,
    class_imbalance: str = "balanced_weights",
    pos_scale: float = 1.0,
    quick_mode: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Return an untuned base estimator and hyperparameter search distributions."""
    name = model_name.lower().replace(" ", "_").replace("-", "_")

    if "logistic" in name:
        base_estimator = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced" if class_imbalance == "balanced_weights" else None,
        )
        if quick_mode:
            param_dist = {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
            }
        else:
            param_dist = {
                "C": np.logspace(-3, 2, 8),
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
            }
        return base_estimator, param_dist

    elif "random_forest" in name or "forest" in name or name == "rf":
        base_estimator = RandomForestClassifier(
            random_state=42,
            n_jobs=1,
            class_weight="balanced" if class_imbalance == "balanced_weights" else None,
        )
        if quick_mode:
            param_dist = {
                "n_estimators": [50, 75],
                "max_depth": [6, 10],
                "min_samples_split": [2, 5],
            }
        else:
            param_dist = {
                "n_estimators": [100, 150, 200],
                "max_depth": [6, 10, 14, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            }
        return base_estimator, param_dist

    elif "xgboost" in name or "xgb" in name:
        scale_weight = pos_scale if class_imbalance == "balanced_weights" else 1.0
        base_estimator = XGBClassifier(
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
            scale_pos_weight=scale_weight,
        )
        if quick_mode:
            param_dist = {
                "n_estimators": [50, 75],
                "max_depth": [3, 4],
                "learning_rate": [0.1, 0.15],
                "subsample": [0.85, 1.0],
            }
        else:
            param_dist = {
                "n_estimators": [100, 150, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.08, 0.15],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [0.5, 1.0, 3.0],
            }
        return base_estimator, param_dist

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    quick_mode: bool = True,
    class_imbalance: str = "balanced_weights",
    scoring: str = "roc_auc",
    cv_folds: Optional[int] = None,
) -> Dict[str, Any]:
    """Tune a single model using StratifiedKFold CV and RandomizedSearchCV."""
    # Compute positive scale weight for XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_scale = float(n_neg / max(n_pos, 1))

    base_estimator, param_dist = get_search_spaces(
        model_name=model_name,
        class_imbalance=class_imbalance,
        pos_scale=pos_scale,
        quick_mode=quick_mode,
    )

    actual_cv = cv_folds if cv_folds is not None else (3 if quick_mode else 5)
    cv = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=42)
    n_iter = 3 if quick_mode else 12

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=True,
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    duration = time.time() - start_time

    best_idx = search.best_index_
    cv_mean = float(search.cv_results_["mean_test_score"][best_idx])
    cv_std = float(search.cv_results_["std_test_score"][best_idx])

    return {
        "model_name": model_name,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_score_mean": cv_mean,
        "cv_score_std": cv_std,
        "scoring_metric": scoring,
        "train_time_sec": duration,
    }
