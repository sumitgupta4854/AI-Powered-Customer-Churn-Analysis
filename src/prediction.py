"""
prediction.py
-------------
Generate churn probability scores and risk categories for every customer.

Risk thresholds (configurable):
  LOW    : probability < low_threshold     (default 0.30)
  MEDIUM : low_threshold ≤ prob < high_threshold (default 0.60)
  HIGH   : probability ≥ high_threshold    (default 0.60)
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default risk thresholds
DEFAULT_LOW_THRESHOLD = 0.30
DEFAULT_HIGH_THRESHOLD = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# Risk categorisation
# ─────────────────────────────────────────────────────────────────────────────

def assign_risk_level(
    probability: float,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> str:
    if probability >= high_threshold:
        return "High Risk"
    elif probability >= low_threshold:
        return "Medium Risk"
    return "Low Risk"


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict_churn(
    df: pd.DataFrame,
    model,
    preprocessor,
    feature_cols: list,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate churn predictions for every customer in `df`.

    Parameters
    ----------
    df            : cleaned customer DataFrame
    model         : fitted sklearn / XGBoost model
    preprocessor  : fitted ColumnTransformer
    feature_cols  : list of feature column names used during training
    low_threshold / high_threshold : risk category thresholds

    Returns
    -------
    predictions_df : original df with added columns:
        Churn Probability, Risk Level, Predicted Churn
    """
    X = df[feature_cols].copy()
    X_transformed = preprocessor.transform(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_transformed)[:, 1]
    else:
        probabilities = model.predict(X_transformed).astype(float)

    predicted_churn = (probabilities >= 0.5).astype(int)

    result = df.copy()
    result["Churn Probability"] = np.round(probabilities, 4)
    result["Predicted Churn"] = predicted_churn
    result["Risk Level"] = [
        assign_risk_level(p, low_threshold, high_threshold) for p in probabilities
    ]

    logger.info(
        "Predictions generated. High=%.0f  Medium=%.0f  Low=%.0f",
        (result["Risk Level"] == "High Risk").sum(),
        (result["Risk Level"] == "Medium Risk").sum(),
        (result["Risk Level"] == "Low Risk").sum(),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Risk factor explanation per customer
# ─────────────────────────────────────────────────────────────────────────────

# Map of column → human-readable risk description
# Keys include both canonical and dataset-variant names
_RISK_FACTOR_RULES: Dict[str, Any] = {
    "Contract": {
        "check": lambda v: str(v).lower() in {"month-to-month"},
        "text": "month-to-month contract (no long-term commitment)",
    },
    "tenure": {
        "check": lambda v: pd.notna(v) and float(v) < 12,
        "text": "short customer tenure (< 12 months)",
    },
    "TenureMonths": {
        "check": lambda v: pd.notna(v) and float(v) < 12,
        "text": "short customer tenure (< 12 months)",
    },
    "MonthlyCharges": {
        "check": lambda v: pd.notna(v) and float(v) > 65,
        "text": "high monthly charges",
    },
    "TechSupport": {
        "check": lambda v: str(v).lower() in {"no"},
        "text": "no tech support subscription",
    },
    "OnlineSecurity": {
        "check": lambda v: str(v).lower() in {"no"},
        "text": "no online security subscription",
    },
    "InternetService": {
        "check": lambda v: str(v).lower() in {"fiber optic"},
        "text": "fiber optic internet (higher churn segment)",
    },
    "PaymentMethod": {
        "check": lambda v: str(v).lower() in {"electronic check"},
        "text": "electronic check payment (higher churn payment method)",
    },
    "SeniorCitizen": {
        "check": lambda v: pd.notna(v) and int(float(v)) == 1,
        "text": "senior citizen (higher-risk demographic)",
    },
    "Age": {
        "check": lambda v: pd.notna(v) and float(v) >= 60,
        "text": "senior age (60+, higher-risk demographic)",
    },
    "SupportCalls": {
        "check": lambda v: pd.notna(v) and float(v) >= 4,
        "text": "frequent support calls (indicates dissatisfaction)",
    },
    "Dependents": {
        "check": lambda v: str(v).lower() in {"no"},
        "text": "no dependents",
    },
    "Partner": {
        "check": lambda v: str(v).lower() in {"no"},
        "text": "no partner",
    },
}


def explain_customer_risk(row: pd.Series) -> str:
    """
    Generate a rule-based, data-grounded explanation for a single customer's
    churn risk.  Only reports factors actually present in the data row.
    """
    factors = []
    for col, rule in _RISK_FACTOR_RULES.items():
        if col in row.index:
            try:
                if rule["check"](row[col]):
                    factors.append(rule["text"])
            except (ValueError, TypeError):
                pass

    if not factors:
        return "No dominant risk factors identified from available features."

    if len(factors) == 1:
        return f"This customer has elevated churn risk due to: {factors[0]}."

    factor_list = "; ".join(factors[:-1]) + f"; and {factors[-1]}"
    return (
        f"This customer has elevated churn risk because of: {factor_list}."
    )


def add_risk_explanations(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised: add 'Risk Factors' column to predictions DataFrame."""
    pred_df = pred_df.copy()
    pred_df["Risk Factors"] = pred_df.apply(explain_customer_risk, axis=1)
    return pred_df


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def prediction_summary(pred_df: pd.DataFrame) -> Dict[str, Any]:
    """Return aggregate statistics over the predictions DataFrame."""
    total = len(pred_df)
    risk_counts = pred_df["Risk Level"].value_counts().to_dict()
    return {
        "total_customers": total,
        "high_risk": int(risk_counts.get("High Risk", 0)),
        "medium_risk": int(risk_counts.get("Medium Risk", 0)),
        "low_risk": int(risk_counts.get("Low Risk", 0)),
        "avg_churn_probability": round(float(pred_df["Churn Probability"].mean()), 4),
        "pct_predicted_churn": round(
            100 * pred_df["Predicted Churn"].mean(), 2
        ),
    }
