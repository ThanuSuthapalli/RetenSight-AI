"""Model evaluation, classification metrics, and dynamic threshold tuning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute standard and threshold-dependent classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Probabilistic metrics
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_pts, precision_pts)
    except Exception:
        pr_auc = 0.0

    brier = brier_score_loss(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
        "actual_churners": int(tp + fn),
        "caught_churners": int(tp),
        "missed_churners": int(fn),
        "false_alarms": int(fp),
    }


def get_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate ROC curve points."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_score = roc_auc_score(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": float(roc_score),
    }


def get_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate Precision-Recall curve points."""
    precision_pts, recall_pts, thresholds = precision_recall_curve(y_true, y_prob)
    pr_score = auc(recall_pts, precision_pts)
    return {
        "precision": precision_pts.tolist(),
        "recall": recall_pts.tolist(),
        "thresholds": thresholds.tolist(),
        "pr_auc": float(pr_score),
    }


def get_calibration_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Calculate reliability calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def compare_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compare multiple fitted models on the test set."""
    rows = []
    for model_name, model_info in models_dict.items():
        estimator = model_info["estimator"] if isinstance(model_info, dict) else model_info
        
        # Probabilities
        if hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            decision = estimator.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision))
        else:
            y_pred_direct = estimator.predict(X_test)
            y_prob = y_pred_direct.astype(float)

        metrics = compute_metrics_at_threshold(y_test, y_prob, threshold=threshold)
        
        cv_mean = model_info.get("cv_score_mean") if isinstance(model_info, dict) else None
        cv_std = model_info.get("cv_score_std") if isinstance(model_info, dict) else None
        train_time = model_info.get("train_time_sec") if isinstance(model_info, dict) else None

        rows.append({
            "Model": model_name,
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "PR-AUC": round(metrics["pr_auc"], 4),
            "Recall": round(metrics["recall"], 4),
            "Precision": round(metrics["precision"], 4),
            "F1-Score": round(metrics["f1"], 4),
            "Accuracy": round(metrics["accuracy"], 4),
            "CV Mean ± Std": f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_mean is not None and cv_std is not None else "N/A",
            "Train Time (s)": round(train_time, 2) if train_time is not None else "N/A",
        })

    df_comp = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return df_comp

