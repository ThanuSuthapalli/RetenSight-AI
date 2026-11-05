"""Unit tests for SHAP explainability and business insights generation."""

import numpy as np
import pandas as pd
import pytest

from src.explain.insights import (
    generate_executive_insights,
    calculate_revenue_at_risk,
    get_retention_playbook,
)


def test_generate_executive_insights():
    df = pd.DataFrame({
        "Contract": ["Month-to-month", "Month-to-month", "Two year", "Two year"],
        "tenure": [2, 4, 36, 48],
        "MonthlyCharges": [70.0, 80.0, 40.0, 50.0],
        "Churn": ["Yes", "Yes", "No", "No"],
    })

    insights = generate_executive_insights(df, target_col="Churn", positive_label="Yes")
    assert len(insights) >= 2
    assert any("Contract" in ins["title"] for ins in insights)


def test_calculate_revenue_at_risk():
    df = pd.DataFrame({
        "MonthlyCharges": [100.0, 80.0, 50.0, 30.0],
    })
    probs = np.array([0.85, 0.65, 0.40, 0.10])

    res = calculate_revenue_at_risk(df, probs, high_risk_threshold=0.60)
    assert res["high_risk_count"] == 2
    assert res["high_risk_monthly_loss"] == 180.0
    assert res["high_risk_annual_loss"] == 180.0 * 12


def test_get_retention_playbook():
    actions = get_retention_playbook(["Contract_Month-to-month", "tenure", "MonthlyCharges"])
    assert len(actions) >= 2
    assert any("Annual" in a["action"] or "Contract" in a["action"] for a in actions)

