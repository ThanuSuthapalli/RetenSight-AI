"""Matplotlib and Seaborn visualization helpers for churn platform."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Clean modern aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {
    "primary": "#2563EB",     # Royal Blue
    "secondary": "#64748B",   # Slate Gray
    "danger": "#EF4444",      # Coral Red
    "warning": "#F59E0B",     # Amber
    "success": "#10B981",     # Emerald Green
    "dark": "#0F172A",
}


def plot_confusion_matrix(cm: List[List[int]], model_name: str = "Model") -> plt.Figure:
    """Render a beautifully annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=100)
    cm_arr = np.array(cm)

    # Calculate percentages
    cm_sum = cm_arr.sum()
    labels = []
    group_names = [["True Neg (Stay)", "False Pos (Alarm)"], ["False Neg (Miss)", "True Pos (Churn)"]]
    for i in range(2):
        row = []
        for j in range(2):
            val = cm_arr[i, j]
            pct = val / max(cm_sum, 1) * 100
            row.append(f"{group_names[i][j]}\n{val:,}\n({pct:.1f}%)")
        labels.append(row)

    sns.heatmap(
        cm_arr,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Predicted Stay", "Predicted Churn"],
        yticklabels=["Actual Stay", "Actual Churn"],
        annot_kws={"fontsize": 9, "fontweight": "medium"},
    )
    ax.set_title(f"Confusion Matrix · {model_name}", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def plot_roc_curves(roc_dict: Dict[str, Dict[str, Any]]) -> plt.Figure:
    """Overlaid ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
    colors = [PALETTE["primary"], PALETTE["danger"], PALETTE["warning"], PALETTE["success"]]

    for idx, (name, data) in enumerate(roc_dict.items()):
        color = colors[idx % len(colors)]
        auc_score = data.get("roc_auc", 0.0)
        ax.plot(
            data["fpr"],
            data["tpr"],
            label=f"{name} (AUC = {auc_score:.3f})",
            color=color,
            linewidth=2.2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess (0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    return fig


def plot_pr_curves(pr_dict: Dict[str, Dict[str, Any]], baseline_pos_rate: float = 0.26) -> plt.Figure:
    """Overlaid Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
    colors = [PALETTE["primary"], PALETTE["danger"], PALETTE["warning"], PALETTE["success"]]

    for idx, (name, data) in enumerate(pr_dict.items()):
        color = colors[idx % len(colors)]
        pr_auc = data.get("pr_auc", 0.0)
        ax.plot(
            data["recall"],
            data["precision"],
            label=f"{name} (PR-AUC = {pr_auc:.3f})",
            color=color,
            linewidth=2.2,
        )

    ax.axhline(
        baseline_pos_rate,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"Baseline Churn Rate ({baseline_pos_rate * 100:.1f}%)",
    )
    ax.set_xlabel("Recall (Caught Churners)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Precision (Accuracy of Alarms)", fontsize=10, fontweight="bold")
    ax.set_title("Precision-Recall Curves", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    return fig


def plot_feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """Horizontal bar chart for SHAP or Model feature importance."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
    plot_data = importance_df.head(top_n).iloc[::-1]

    val_col = "Mean_Abs_SHAP" if "Mean_Abs_SHAP" in plot_data.columns else plot_data.columns[1]
    y_col = "Feature"

    bars = ax.barh(plot_data[y_col], plot_data[val_col], color=PALETTE["primary"], height=0.65)
    ax.set_xlabel("Mean Absolute SHAP Impact (|Δ log-odds|)", fontsize=10, fontweight="bold")
    ax.set_title(f"Top {top_n} Global Drivers of Customer Churn", fontsize=12, fontweight="bold", pad=12)

    # Value annotations on bar tips
    max_val = plot_data[val_col].max()
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            fontsize=8.5,
            color=PALETTE["dark"],
        )
    ax.set_xlim([0, max_val * 1.18])
    plt.tight_layout()
    return fig


def plot_segment_churn_bar(segment_stats: pd.DataFrame, segment_col: str) -> plt.Figure:
    """Bar chart showing churn rate by business segment with benchmark line."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=100)
    avg_churn = segment_stats["churned_customers"].sum() / segment_stats["total_customers"].sum() * 100

    colors = [
        PALETTE["danger"] if rate > avg_churn else PALETTE["primary"]
        for rate in segment_stats["churn_rate_pct"]
    ]

    bars = ax.bar(
        segment_stats[segment_col].astype(str),
        segment_stats["churn_rate_pct"],
        color=colors,
        width=0.55,
    )
    ax.axhline(
        avg_churn,
        color=PALETTE["dark"],
        linestyle="--",
        linewidth=1.2,
        label=f"Avg Churn Rate ({avg_churn:.1f}%)",
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{h:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Churn Rate (%)", fontsize=10, fontweight="bold")
    ax.set_title(f"Churn Rate by {segment_col}", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="upper right")
    ax.set_ylim([0, max(segment_stats["churn_rate_pct"].max() + 8, 10)])
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Generate Pearson correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(max(5, numeric_df.shape[1] * 0.7), max(4, numeric_df.shape[1] * 0.6)), dpi=100)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Numeric Features Correlation Matrix", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

