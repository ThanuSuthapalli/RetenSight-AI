"""Automated business insight generation, revenue-at-risk analysis, and retention playbooks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_segment_churn_stats(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Any,
    segment_col: str,
) -> pd.DataFrame:
    """Calculate customer count, churn count, and churn rate by segment."""
    if segment_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    y_binary = (df[target_col].astype(str).str.strip() == str(positive_label).strip()).astype(int)
    temp = df[[segment_col]].copy()
    temp["is_churn"] = y_binary

    grouped = temp.groupby(segment_col).agg(
        total_customers=("is_churn", "count"),
        churned_customers=("is_churn", "sum"),
    ).reset_index()

    grouped["churn_rate_pct"] = (
        (grouped["churned_customers"] / grouped["total_customers"]) * 100
    ).round(1)

    return grouped.sort_values(by="churn_rate_pct", ascending=False).reset_index(drop=True)


def generate_executive_insights(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Any,
) -> List[Dict[str, Any]]:
    """Automatically detect high-impact churn drivers and generate executive insight cards."""
    insights: List[Dict[str, Any]] = []
    y_binary = (df[target_col].astype(str).str.strip() == str(positive_label).strip()).astype(int)
    overall_churn_rate = float(y_binary.mean() * 100)

    # 1. Overall baseline
    insights.append({
        "title": "Overall Churn Baseline",
        "category": "Baseline",
        "metric": f"{overall_churn_rate:.1f}%",
        "badge": "Company-wide",
        "description": f"Across {len(df):,} total analyzed customers, {y_binary.sum():,} ({overall_churn_rate:.1f}%) churned.",
        "icon": "📊",
    })

    # Look for Contract column
    contract_cols = [c for c in df.columns if "contract" in c.lower()]
    if contract_cols:
        col = contract_cols[0]
        stats = compute_segment_churn_stats(df, target_col, positive_label, col)
        if len(stats) >= 2:
            highest = stats.iloc[0]
            lowest = stats.iloc[-1]
            diff = highest["churn_rate_pct"] - lowest["churn_rate_pct"]
            insights.append({
                "title": f"Contract Commitment Disparity ({col})",
                "category": "Contract",
                "metric": f"{highest['churn_rate_pct']}% vs {lowest['churn_rate_pct']}%",
                "badge": f"+{diff:.1f}% Churn Gap",
                "description": (
                    f"Customers on '{highest[col]}' churn at {highest['churn_rate_pct']}%, "
                    f"compared to only {lowest['churn_rate_pct']}% on '{lowest[col]}'. "
                    "Long-term agreements drastically reduce turnover."
                ),
                "icon": "📝",
            })

    # Look for Tenure column
    tenure_cols = [c for c in df.columns if "tenure" in c.lower()]
    if tenure_cols:
        col = tenure_cols[0]
        t = pd.to_numeric(df[col], errors="coerce").fillna(0)
        early_churn = y_binary[t <= 12].mean() * 100 if (t <= 12).sum() > 0 else 0
        mature_churn = y_binary[t > 24].mean() * 100 if (t > 24).sum() > 0 else 0
        insights.append({
            "title": "Early-Tenure Onboarding Vulnerability",
            "category": "Tenure",
            "metric": f"{early_churn:.1f}% Churn in Year 1",
            "badge": "Onboarding Risk",
            "description": (
                f"First-year customers (tenure ≤ 12 months) churn at {early_churn:.1f}%, "
                f"versus {mature_churn:.1f}% for customers with >2 years tenure. "
                "The first 90-180 days are critical for user retention."
            ),
            "icon": "⏳",
        })

    # Look for Tech Support / Online Security
    support_cols = [c for c in df.columns if "support" in c.lower() or "security" in c.lower()]
    if support_cols:
        col = support_cols[0]
        stats = compute_segment_churn_stats(df, target_col, positive_label, col)
        no_supp = stats[stats[col].astype(str).str.lower().isin(["no", "none", "0"])]
        yes_supp = stats[stats[col].astype(str).str.lower().isin(["yes", "true", "1"])]
        if not no_supp.empty and not yes_supp.empty:
            no_rate = no_supp.iloc[0]["churn_rate_pct"]
            yes_rate = yes_supp.iloc[0]["churn_rate_pct"]
            insights.append({
                "title": f"Service Protection Impact ({col})",
                "category": "Customer Support",
                "metric": f"{no_rate:.1f}% without vs {yes_rate:.1f}% with",
                "badge": "Stickiness Driver",
                "description": (
                    f"Subscribers without {col} exhibit a {no_rate:.1f}% churn rate, "
                    f"whereas those with the service churn at just {yes_rate:.1f}%. "
                    "Support services create high platform stickiness."
                ),
                "icon": "🛡️",
            })

    # Look for Payment Method
    payment_cols = [c for c in df.columns if "payment" in c.lower() or "billing" in c.lower()]
    if payment_cols:
        col = payment_cols[0]
        stats = compute_segment_churn_stats(df, target_col, positive_label, col)
        if len(stats) >= 2:
            worst = stats.iloc[0]
            insights.append({
                "title": f"Payment Friction Point ({col})",
                "category": "Billing",
                "metric": f"{worst['churn_rate_pct']}% via {worst[col]}",
                "badge": "Billing Method",
                "description": (
                    f"Customers paying via '{worst[col]}' suffer the highest churn rate at "
                    f"{worst['churn_rate_pct']}%. Automated autopay options typically reduce involuntary churn."
                ),
                "icon": "💳",
            })

    return insights


def calculate_revenue_at_risk(
    df: pd.DataFrame,
    churn_probabilities: np.ndarray,
    high_risk_threshold: float = 0.60,
    medium_risk_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Calculate financial exposure and customer count across risk tiers."""
    probs = np.asarray(churn_probabilities)
    n_total = len(probs)

    # Find monthly revenue column
    cols_lower = {str(c).lower(): c for c in df.columns}
    rev_col = (
        cols_lower.get("monthlycharges")
        or cols_lower.get("monthly_charges")
        or cols_lower.get("balance")
        or cols_lower.get("revenue")
    )

    if rev_col is not None:
        revenue = pd.to_numeric(df[rev_col], errors="coerce").fillna(0).to_numpy()
        rev_name = rev_col
    else:
        # Fallback dummy revenue ₹50/mo per customer for calculation
        revenue = np.full(n_total, 50.0)
        rev_name = "Estimated (₹50/mo benchmark)"

    high_mask = probs >= high_risk_threshold
    med_mask = (probs >= medium_risk_threshold) & (probs < high_risk_threshold)
    low_mask = probs < medium_risk_threshold

    high_rev = float(revenue[high_mask].sum())
    med_rev = float(revenue[med_mask].sum())
    total_rev = float(revenue.sum())

    return {
        "revenue_column_used": rev_name,
        "total_revenue": total_rev,
        "high_risk_count": int(high_mask.sum()),
        "high_risk_pct": round(float(high_mask.mean() * 100), 1),
        "high_risk_monthly_loss": high_rev,
        "high_risk_annual_loss": high_rev * 12,
        "medium_risk_count": int(med_mask.sum()),
        "medium_risk_monthly_loss": med_rev,
        "low_risk_count": int(low_mask.sum()),
    }


def get_retention_playbook(top_risk_factors: List[str]) -> List[Dict[str, str]]:
    """Map top identified risk factors to prescriptive retention actions."""
    playbook_map = {
        "contract": {
            "action": "Offer Annual Upgrade Incentive",
            "detail": "Target month-to-month subscribers with a 15% discount or bundled upgrade to a 12-month commitment.",
            "target_team": "Retention & Lifecycle Marketing",
        },
        "support": {
            "action": "Proactive Tech Support Check-in",
            "detail": "Trigger a white-glove onboarding call or complimentary 60-day tech support trial.",
            "target_team": "Customer Success",
        },
        "tenure": {
            "action": "First 90-Day Nurture Sequence",
            "detail": "Deploy automated milestone rewards and satisfaction surveys at Day 14, 30, and 60.",
            "target_team": "Product & Growth",
        },
        "payment": {
            "action": "Autopay Switch Bonus",
            "detail": "Offer a one-time ₹500 incentive to transition manual or check payers to automated credit card / ACH.",
            "target_team": "Billing Operations",
        },
        "charge": {
            "action": "Value-Add Plan Review",
            "detail": "Conduct an account review to recommend optimal plan sizing or provide free premium add-ons.",
            "target_team": "Account Management",
        },
    }

    selected_actions = []
    seen_keys = set()
    for factor in top_risk_factors:
        f_lower = factor.lower()
        for key, rec in playbook_map.items():
            if key in f_lower and key not in seen_keys:
                selected_actions.append(rec)
                seen_keys.add(key)
                break

    # If no specific matches, provide high-value defaults
    if not selected_actions:
        selected_actions = [
            playbook_map["contract"],
            playbook_map["support"],
            playbook_map["charge"],
        ]

    return selected_actions

