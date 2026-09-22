"""
utils.py
--------
Shared utilities: KPI calculation, SQL queries, download helpers,
dataset generation, and formatting.
"""

import io
import textwrap
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# KPI Calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_kpis(df: pd.DataFrame, pred_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Return a dictionary of business KPIs computed from the dataset."""
    kpis: Dict[str, Any] = {}

    kpis["total_customers"] = len(df)

    if "Churn" in df.columns:
        kpis["churned_customers"] = int(df["Churn"].sum())
        kpis["active_customers"] = int(len(df) - kpis["churned_customers"])
        kpis["churn_rate_pct"] = round(df["Churn"].mean() * 100, 2)
    else:
        kpis["churned_customers"] = "N/A"
        kpis["active_customers"] = "N/A"
        kpis["churn_rate_pct"] = "N/A"

    if "MonthlyCharges" in df.columns:
        kpis["avg_monthly_charges"] = round(df["MonthlyCharges"].mean(), 2)
    else:
        kpis["avg_monthly_charges"] = "N/A"

    if "TotalCharges" in df.columns:
        kpis["avg_total_charges"] = round(df["TotalCharges"].mean(), 2)
        if "Churn" in df.columns:
            kpis["revenue_at_risk"] = round(
                df[df["Churn"] == 1]["TotalCharges"].sum(), 2
            )
    else:
        kpis["avg_total_charges"] = "N/A"

    tenure_col = next((c for c in ["tenure", "TenureMonths"] if c in df.columns), None)
    if tenure_col:
        kpis["avg_tenure_months"] = round(df[tenure_col].mean(), 2)
    else:
        kpis["avg_tenure_months"] = "N/A"

    if pred_df is not None and "Risk Level" in pred_df.columns:
        risk_counts = pred_df["Risk Level"].value_counts().to_dict()
        kpis["high_risk_customers"] = int(risk_counts.get("High Risk", 0))
        kpis["medium_risk_customers"] = int(risk_counts.get("Medium Risk", 0))
        kpis["low_risk_customers"] = int(risk_counts.get("Low Risk", 0))
    else:
        kpis["high_risk_customers"] = "Run model"
        kpis["medium_risk_customers"] = "Run model"
        kpis["low_risk_customers"] = "Run model"

    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# SQL Query Bank
# ─────────────────────────────────────────────────────────────────────────────

SQL_QUERIES: Dict[str, Tuple[str, str]] = {
    "Total Customers": (
        "Count total customers in the dataset.",
        """SELECT COUNT(*) AS total_customers
FROM customers;""",
    ),
    "Total Churned Customers": (
        "Count customers who have churned.",
        """SELECT COUNT(*) AS churned_customers
FROM customers
WHERE Churn = 1;""",
    ),
    "Overall Churn Rate": (
        "Calculate the overall churn rate as a percentage.",
        """SELECT
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers;""",
    ),
    "Average Monthly Charges": (
        "Average monthly charges across all customers.",
        """SELECT
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(AVG(CASE WHEN Churn = 1 THEN MonthlyCharges END), 2) AS avg_charges_churned,
    ROUND(AVG(CASE WHEN Churn = 0 THEN MonthlyCharges END), 2) AS avg_charges_active
FROM customers;""",
    ),
    "Churn by Contract Type": (
        "Churn count and rate broken down by contract type.",
        """SELECT
    Contract,
    COUNT(*)                                      AS total_customers,
    SUM(Churn)                                    AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)       AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;""",
    ),
    "Churn by Payment Method": (
        "Churn rate per payment method.",
        """SELECT
    PaymentMethod,
    COUNT(*)                                      AS total_customers,
    SUM(Churn)                                    AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)       AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;""",
    ),
    "Churn by Internet Service": (
        "Churn statistics per internet service type.",
        """SELECT
    InternetService,
    COUNT(*)                                      AS total_customers,
    SUM(Churn)                                    AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)       AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;""",
    ),
    "Top Churn Segments": (
        "Customer segments (Contract + InternetService) with highest churn.",
        """SELECT
    Contract,
    InternetService,
    COUNT(*)                                      AS total_customers,
    SUM(Churn)                                    AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)       AS churn_rate_pct
FROM customers
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC
LIMIT 10;""",
    ),
    "Average Tenure: Churned vs Active": (
        "Compare average tenure between churned and retained customers.",
        """SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Active' END AS customer_status,
    COUNT(*)                                              AS count,
    ROUND(AVG(tenure), 2)                                 AS avg_tenure_months,
    ROUND(MIN(tenure), 2)                                 AS min_tenure,
    ROUND(MAX(tenure), 2)                                 AS max_tenure
FROM customers
GROUP BY Churn;""",
    ),
    "Revenue from Churned Customers": (
        "Total and average revenue associated with churned customers.",
        """SELECT
    ROUND(SUM(TotalCharges), 2)   AS total_revenue_churned,
    ROUND(AVG(TotalCharges), 2)   AS avg_revenue_per_churned,
    COUNT(*)                      AS churned_count
FROM customers
WHERE Churn = 1;""",
    ),
}


def get_sql_query(query_name: str) -> Tuple[str, str]:
    """Return (description, sql_text) for the given query name."""
    if query_name not in SQL_QUERIES:
        raise KeyError(f"Unknown SQL query: '{query_name}'")
    return SQL_QUERIES[query_name]


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame to UTF-8 CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """
    Write multiple DataFrames (one per sheet) to an Excel file in memory.
    Returns raw bytes.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in dfs.items():
            # Excel sheet names max 31 chars
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()


def generate_text_report(
    kpis: Dict[str, Any],
    insights: Dict[str, str],
    eval_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a plain-text summary report for download."""
    sep = "=" * 70
    lines = [
        sep,
        "AI-POWERED CUSTOMER CHURN ANALYSIS — SUMMARY REPORT",
        sep,
        "",
        "BUSINESS KPIs",
        "-" * 40,
    ]
    for k, v in kpis.items():
        label = k.replace("_", " ").title()
        lines.append(f"  {label:<35} {v}")

    lines += ["", "KEY FINDINGS", "-" * 40,
              insights.get("key_findings", ""), ""]
    lines += ["CHURN DRIVERS", "-" * 40,
              insights.get("churn_drivers", ""), ""]
    lines += ["SEGMENT INSIGHTS", "-" * 40,
              insights.get("segment_insights", ""), ""]
    lines += ["RETENTION RECOMMENDATIONS", "-" * 40,
              insights.get("recommendations", ""), ""]

    if eval_info:
        # Metrics are nested inside report_dict (keyed by class label)
        report = eval_info.get("report_dict", {})
        churned = report.get("Churned (1)", {})
        lines += [
            "MODEL EVALUATION",
            "-" * 40,
            f"  Best Model  : {eval_info.get('model_name', 'N/A')}",
            f"  Accuracy    : {report.get('accuracy', 'N/A')}",
            f"  Precision   : {churned.get('precision', 'N/A')}",
            f"  Recall      : {churned.get('recall', 'N/A')}",
            f"  F1 Score    : {churned.get('f1-score', 'N/A')}",
            f"  ROC-AUC     : {eval_info.get('auc', 'N/A')}",
            "",
            "MODEL LIMITATIONS",
            "-" * 40,
            "  - Model performance depends on the size and quality of training data.",
            "  - Class imbalance (more non-churners) may bias predictions.",
            "  - Historical data may not capture future market shifts.",
            "  - Predictions are probabilistic, not deterministic.",
            "  - Always validate predictions before acting on them.",
        ]

    lines += ["", sep, "Generated by AI-Powered Customer Churn Analysis", sep]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sample dataset generator (demo fallback when no CSV is uploaded)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_dataset(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer churn dataset that mirrors the structure
    of the IBM Telco Customer Churn dataset.

    Used only as a demo/fallback when the user has not uploaded a CSV.
    Clearly labelled as synthetic data throughout the dashboard.
    """
    rng = np.random.default_rng(random_state)

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    gender = rng.choice(["Male", "Female"], size=n)
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    tenure = rng.integers(1, 72, size=n)
    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = rng.choice(["Yes", "No", "No phone service"], size=n, p=[0.42, 0.48, 0.10])
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])
    online_security = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.28, 0.50, 0.22])
    online_backup = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.34, 0.44, 0.22])
    device_protection = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.34, 0.44, 0.22])
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.29, 0.49, 0.22])
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.38, 0.40, 0.22])
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.39, 0.39, 0.22])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21],
    )
    monthly_charges = np.round(rng.uniform(18, 118, size=n), 2)
    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.9, 1.1, size=n), 2)

    # Churn probability — weighted by known risk factors
    churn_score = (
        0.30 * (contract == "Month-to-month").astype(float)
        + 0.20 * (monthly_charges > 70).astype(float)
        + 0.15 * (tenure < 12).astype(float)
        + 0.10 * (internet == "Fiber optic").astype(float)
        + 0.10 * (online_security == "No").astype(float)
        + 0.05 * (tech_support == "No").astype(float)
        + 0.05 * (payment == "Electronic check").astype(float)
        + 0.05 * senior.astype(float)
    )
    churn_prob = churn_score / churn_score.max()
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Column display helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_display_columns(df: pd.DataFrame) -> list:
    """Return columns in a sensible display order, putting customerID first."""
    id_like = [c for c in df.columns if "id" in c.lower()]
    others = [c for c in df.columns if c not in id_like]
    return id_like + others


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_currency(value: float) -> str:
    return f"${value:,.2f}"
