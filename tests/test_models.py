"""Unit tests for model tuning, evaluation metrics, and registry."""

import numpy as np
import pytest

from src.models.evaluate import compute_metrics_at_threshold
from src.models.tune import tune_model


def test_compute_metrics_at_threshold():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9, 0.7])  # 1 false alarm at 0.9

    m = compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)
    assert m["total_samples"] == 6
    assert m["actual_churners"] == 3
    assert m["caught_churners"] == 3  # all 3 >= 0.5 caught
    assert m["recall"] == 1.0
    assert m["false_alarms"] == 1
    assert m["precision"] == 0.75
    assert m["roc_auc"] > 0.5


def test_tune_model_logistic():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], 100)

    res = tune_model(
        model_name="Logistic Regression",
        X_train=X,
        y_train=y,
        quick_mode=True,
        scoring="roc_auc",
        cv_folds=3,
    )

    assert "best_estimator" in res
    assert "cv_score_mean" in res
    assert hasattr(res["best_estimator"], "predict_proba")

