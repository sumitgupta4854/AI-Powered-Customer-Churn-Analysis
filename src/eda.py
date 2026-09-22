"""
eda.py
------
Exploratory Data Analysis module.

All chart functions return a Plotly Figure object so they can be rendered
directly inside Streamlit with st.plotly_chart().

Functions are intentionally pure — they accept a DataFrame and return
a Figure.  No global state is used.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
CHURN_COLORS = {"Churned": "#EF4444", "Active": "#22C55E"}
PALETTE = px.colors.qualitative.Set2
LOW_COLOR = "#22C55E"
MED_COLOR = "#F59E0B"
HIGH_COLOR = "#EF4444"


# ─────────────────────────────────────────────────────────────────────────────
# 1. OVERALL CHURN DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def fig_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing churned vs active customers."""
    counts = df["Churn"].value_counts()
    labels = ["Active" if v == 0 else "Churned" for v in counts.index]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values,
            hole=0.55,
            marker_colors=[CHURN_COLORS.get(l, "#94A3B8") for l in labels],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} customers (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title="Overall Churn Distribution",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEMOGRAPHICS
# ─────────────────────────────────────────────────────────────────────────────

def fig_gender_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart — gender distribution."""
    gender_col = next((c for c in ["gender", "Gender"] if c in df.columns), None)
    counts = df[gender_col].value_counts().reset_index() if gender_col else pd.DataFrame()
    if counts.empty:
        return _empty_fig("Gender data not available")
    counts.columns = ["Gender", "Count"]
    fig = px.bar(counts, x="Gender", y="Count", color="Gender",
                 color_discrete_sequence=PALETTE,
                 title="Gender Distribution",
                 text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of Customers")
    return fig


def fig_senior_citizen(df: pd.DataFrame) -> go.Figure:
    """Donut — senior vs non-senior citizens."""
    if "SeniorCitizen" not in df.columns:
        return _empty_fig("SeniorCitizen column not available")
    counts = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"}).value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.5,
            marker_colors=["#6366F1", "#F59E0B"],
            textinfo="label+percent",
        )
    )
    fig.update_layout(title="Senior Citizen Distribution", showlegend=True)
    return fig


def fig_partner_dependents(df: pd.DataFrame) -> go.Figure:
    """Grouped bar — Partner & Dependents status."""
    cols = [c for c in ["Partner", "Dependents"] if c in df.columns]
    if not cols:
        return _empty_fig("Partner / Dependents data not available")
    rows = []
    for col in cols:
        vc = df[col].value_counts()
        for val, cnt in vc.items():
            rows.append({"Attribute": col, "Value": str(val), "Count": cnt})
    plot_df = pd.DataFrame(rows)
    fig = px.bar(plot_df, x="Attribute", y="Count", color="Value",
                 barmode="group", color_discrete_sequence=PALETTE,
                 title="Partner & Dependents Distribution",
                 text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Number of Customers")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. CUSTOMER BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────

def fig_tenure_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram — tenure in months, coloured by churn."""
    tenure_col = next((c for c in ["tenure", "TenureMonths"] if c in df.columns), None)
    if tenure_col is None:
        return _empty_fig("Tenure data not available")
    df_plot = df.copy()
    df_plot["Churn Label"] = df_plot["Churn"].map({1: "Churned", 0: "Active"})
    fig = px.histogram(
        df_plot, x=tenure_col, color="Churn Label",
        color_discrete_map=CHURN_COLORS,
        nbins=40, barmode="overlay", opacity=0.75,
        title="Tenure Distribution by Churn Status",
        labels={tenure_col: "Tenure (months)"},
    )
    fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Customer Count")
    return fig


def fig_monthly_charges_distribution(df: pd.DataFrame) -> go.Figure:
    """Box plot — monthly charges by churn."""
    if "MonthlyCharges" not in df.columns:
        return _empty_fig("MonthlyCharges data not available")
    df_plot = df.copy()
    df_plot["Churn Label"] = df_plot["Churn"].map({1: "Churned", 0: "Active"})
    fig = px.box(
        df_plot, x="Churn Label", y="MonthlyCharges",
        color="Churn Label", color_discrete_map=CHURN_COLORS,
        title="Monthly Charges vs Churn Status",
        labels={"MonthlyCharges": "Monthly Charges ($)"},
        points="outliers",
    )
    fig.update_layout(showlegend=False, xaxis_title="")
    return fig


def fig_total_charges_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram — total charges distribution."""
    if "TotalCharges" not in df.columns:
        return _empty_fig("TotalCharges data not available")
    df_plot = df.copy()
    df_plot["Churn Label"] = df_plot["Churn"].map({1: "Churned", 0: "Active"})
    fig = px.histogram(
        df_plot, x="TotalCharges", color="Churn Label",
        color_discrete_map=CHURN_COLORS,
        nbins=50, barmode="overlay", opacity=0.75,
        title="Total Charges Distribution by Churn Status",
        labels={"TotalCharges": "Total Charges ($)"},
    )
    fig.update_layout(xaxis_title="Total Charges ($)", yaxis_title="Customer Count")
    return fig


def fig_contract_distribution(df: pd.DataFrame) -> go.Figure:
    """Stacked bar — contract type vs churn."""
    if "Contract" not in df.columns:
        return _empty_fig("Contract data not available")
    ct = df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    ct["Churn Label"] = ct["Churn"].map({1: "Churned", 0: "Active"})
    fig = px.bar(
        ct, x="Contract", y="Count", color="Churn Label",
        color_discrete_map=CHURN_COLORS,
        barmode="stack", title="Churn by Contract Type",
        text="Count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(xaxis_title="Contract Type", yaxis_title="Customer Count")
    return fig


def fig_payment_method(df: pd.DataFrame) -> go.Figure:
    """Stacked bar — payment method vs churn."""
    if "PaymentMethod" not in df.columns:
        return _empty_fig("PaymentMethod data not available")
    ct = df.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
    ct["Churn Label"] = ct["Churn"].map({1: "Churned", 0: "Active"})
    fig = px.bar(
        ct, x="PaymentMethod", y="Count", color="Churn Label",
        color_discrete_map=CHURN_COLORS,
        barmode="stack", title="Churn by Payment Method",
        text="Count",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(xaxis_title="", yaxis_title="Customer Count",
                      xaxis_tickangle=-25)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHURN ANALYSIS CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fig_churn_by_gender(df: pd.DataFrame) -> go.Figure:
    """Grouped bar — churn rate by gender."""
    gender_col = next((c for c in ["gender", "Gender"] if c in df.columns), None)
    if gender_col is None:
        return _empty_fig("Gender data not available")
    return _churn_rate_bar(df, gender_col, "Churn Rate by Gender", "Gender")


def fig_churn_by_internet(df: pd.DataFrame) -> go.Figure:
    if "InternetService" not in df.columns:
        return _empty_fig("InternetService data not available")
    return _churn_rate_bar(df, "InternetService", "Churn Rate by Internet Service", "Internet Service")


def fig_churn_by_tech_support(df: pd.DataFrame) -> go.Figure:
    if "TechSupport" not in df.columns:
        return _empty_fig("TechSupport data not available")
    return _churn_rate_bar(df, "TechSupport", "Churn Rate by Tech Support", "Tech Support")


def fig_churn_by_online_security(df: pd.DataFrame) -> go.Figure:
    if "OnlineSecurity" not in df.columns:
        return _empty_fig("OnlineSecurity data not available")
    return _churn_rate_bar(df, "OnlineSecurity", "Churn Rate by Online Security", "Online Security")


def fig_churn_by_senior(df: pd.DataFrame) -> go.Figure:
    if "SeniorCitizen" not in df.columns:
        return _empty_fig("SeniorCitizen data not available")
    df_plot = df.copy()
    df_plot["Senior"] = df_plot["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    return _churn_rate_bar(df_plot, "Senior", "Churn Rate by Senior Citizen Status", "Segment")


def fig_churn_by_charges_bin(df: pd.DataFrame) -> go.Figure:
    """Bar chart — churn rate by monthly charges bucket."""
    if "MonthlyCharges" not in df.columns:
        return _empty_fig("MonthlyCharges data not available")
    df_plot = df.copy()
    df_plot["Charges Bin"] = pd.cut(
        df_plot["MonthlyCharges"],
        bins=[0, 30, 50, 70, 90, 120],
        labels=["$0–30", "$30–50", "$50–70", "$70–90", "$90+"],
    )
    return _churn_rate_bar(df_plot, "Charges Bin", "Churn Rate by Monthly Charges", "Monthly Charges Range")


def fig_churn_by_tenure_bin(df: pd.DataFrame) -> go.Figure:
    """Bar — churn rate by tenure bucket."""
    tenure_col = next((c for c in ["tenure", "TenureMonths"] if c in df.columns), None)
    if tenure_col is None:
        return _empty_fig("Tenure data not available")
    df_plot = df.copy()
    df_plot["Tenure Bin"] = pd.cut(
        df_plot[tenure_col],
        bins=[0, 12, 24, 36, 48, 72],
        labels=["0-12m", "12-24m", "24-36m", "36-48m", "48m+"],
    )
    return _churn_rate_bar(df_plot, "Tenure Bin", "Churn Rate by Customer Tenure", "Tenure")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def fig_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of Pearson correlations for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return _empty_fig("Not enough numeric columns for correlation heatmap")
    corr = num_df.corr(numeric_only=True)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Correlation Heatmap (Numeric Features)",
        xaxis_tickangle=-45,
        height=500,
        margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. RISK DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def fig_risk_distribution(pred_df: pd.DataFrame) -> go.Figure:
    """Donut — Low / Medium / High risk customers."""
    if "Risk Level" not in pred_df.columns:
        return _empty_fig("Risk Level column not available")
    counts = pred_df["Risk Level"].value_counts()
    color_map = {"Low Risk": LOW_COLOR, "Medium Risk": MED_COLOR, "High Risk": HIGH_COLOR}
    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            marker_colors=[color_map.get(l, "#94A3B8") for l in counts.index],
            textinfo="label+percent",
        )
    )
    fig.update_layout(title="Customer Risk Distribution", showlegend=True)
    return fig


def fig_churn_probability_histogram(
    pred_df: pd.DataFrame,
    low_threshold: float = 0.30,
    high_threshold: float = 0.60,
) -> go.Figure:
    """Histogram — churn probability distribution."""
    if "Churn Probability" not in pred_df.columns:
        return _empty_fig("Churn Probability column not available")
    fig = px.histogram(
        pred_df, x="Churn Probability", nbins=30,
        color_discrete_sequence=["#6366F1"],
        title="Churn Probability Distribution",
        labels={"Churn Probability": "Predicted Churn Probability"},
    )
    fig.add_vline(x=low_threshold, line_dash="dash", line_color=MED_COLOR,
                  annotation_text=f"Medium Risk ({low_threshold})")
    fig.add_vline(x=high_threshold, line_dash="dash", line_color=HIGH_COLOR,
                  annotation_text=f"High Risk ({high_threshold})")
    fig.update_layout(yaxis_title="Customer Count")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODEL EVALUATION CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def fig_roc_curve(fpr, tpr, auc_score: float) -> go.Figure:
    """ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_score:.3f})",
                   line=dict(color="#6366F1", width=2))
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                   line=dict(color="#94A3B8", dash="dash"), name="Random Classifier")
    )
    fig.update_layout(
        title=f"ROC Curve — AUC = {auc_score:.3f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
    )
    return fig


def fig_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    labels = ["Active (0)", "Churned (1)"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=380,
    )
    return fig


def fig_feature_importance(feature_names: list, importances: list, top_n: int = 20) -> go.Figure:
    """Horizontal bar — top-N most important features."""
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    features, scores = zip(*pairs)
    fig = go.Figure(
        go.Bar(x=list(scores), y=list(features), orientation="h",
               marker_color="#6366F1",
               hovertemplate="%{y}: %{x:.4f}<extra></extra>")
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=max(350, top_n * 22),
        margin=dict(l=160),
    )
    return fig


def fig_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Grouped bar comparing multiple ML models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    metrics = [m for m in metrics if m in results_df.columns]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(
            go.Bar(name=metric, x=results_df["Model"], y=results_df[metric],
                   text=results_df[metric].round(3), textposition="outside")
        )
    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _churn_rate_bar(df: pd.DataFrame, group_col: str, title: str, x_label: str) -> go.Figure:
    """Generic helper: compute churn rate per group, return bar chart."""
    agg = (
        df.groupby(group_col)["Churn"]
        .agg(total="count", churned="sum")
        .reset_index()
    )
    agg["Churn Rate (%)"] = (agg["churned"] / agg["total"] * 100).round(2)
    fig = px.bar(
        agg, x=group_col, y="Churn Rate (%)",
        color="Churn Rate (%)",
        color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
        range_color=[0, 100],
        title=title,
        text=agg["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        hover_data={"total": True, "churned": True},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title=x_label, yaxis_title="Churn Rate (%)",
                      coloraxis_showscale=False)
    return fig


def _empty_fig(message: str) -> go.Figure:
    """Return a blank Plotly figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#94A3B8"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white", height=300,
    )
    return fig
