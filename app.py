"""
app.py
------
AI-Powered Customer Churn Analysis — Streamlit Dashboard

Run with:
    streamlit run app.py

Pages
-----
1. Executive Overview   — KPIs + key charts
2. Customer Analysis    — filterable customer table with risk levels
3. Churn Drivers        — feature importance + churn breakdown charts
4. AI Insights          — AI-generated findings and recommendations
5. AI Data Analyst      — natural language Q&A
"""

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress noisy sklearn / xgboost warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# Make src/ importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_cleaning import clean_data, audit_dataframe, outlier_report
from eda import (
    fig_churn_distribution, fig_gender_distribution, fig_senior_citizen,
    fig_partner_dependents, fig_tenure_distribution,
    fig_monthly_charges_distribution, fig_total_charges_distribution,
    fig_contract_distribution, fig_payment_method,
    fig_churn_by_gender, fig_churn_by_internet, fig_churn_by_tech_support,
    fig_churn_by_online_security, fig_churn_by_senior,
    fig_churn_by_charges_bin, fig_churn_by_tenure_bin,
    fig_correlation_heatmap, fig_risk_distribution,
    fig_churn_probability_histogram, fig_roc_curve,
    fig_confusion_matrix, fig_feature_importance, fig_model_comparison,
)
from preprocessing import prepare_data
from model_training import train_all_models, select_best_model, detailed_evaluation, save_model
from prediction import predict_churn, add_risk_explanations, prediction_summary
from ai_insights import generate_insights, answer_question
from utils import (
    calculate_kpis, SQL_QUERIES, df_to_csv_bytes, df_to_excel_bytes,
    generate_text_report, generate_sample_dataset, format_pct, format_currency,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Churn Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #F8F9FB; }
    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 16px 16px 16px;
        text-align: center;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1F2937; }
    .kpi-label { font-size: 0.82rem; color: #6B7280; margin-top: 4px; }
    .kpi-delta { font-size: 0.78rem; margin-top: 6px; }
    .risk-high  { color: #EF4444; font-weight: 600; }
    .risk-med   { color: #F59E0B; font-weight: 600; }
    .risk-low   { color: #22C55E; font-weight: 600; }
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
        border-bottom: 2px solid #6366F1;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    /* Sidebar */
    .css-1d391kg { background-color: #1E293B !important; }
    /* Insight boxes */
    .insight-box {
        background: #F0F4FF;
        border-left: 4px solid #6366F1;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .warning-box {
        background: #FFF7ED;
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "clean_report": None,
        "prep_data": None,
        "results_df": None,
        "trained_models": None,
        "best_model_name": None,
        "best_model": None,
        "eval_info": None,
        "pred_df": None,
        "insights": None,
        "is_demo": False,
        "low_threshold": 0.30,
        "high_threshold": 0.60,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Data Loading & Pipeline Controls
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
        st.title("Churn Analysis")
        st.caption("AI-Powered Customer Intelligence")
        st.divider()

        # ── Dataset source ──────────────────────────────────────────────
        st.subheader("📂 Dataset")

        _BUNDLED_CSV = os.path.join(os.path.dirname(__file__), "data", "raw", "ai_customer_churn_dataset.csv")
        _has_bundled = os.path.exists(_BUNDLED_CSV)

        _source_options = ["Use Project Dataset", "Upload CSV", "Use Demo Dataset"] if _has_bundled else ["Upload CSV", "Use Demo Dataset"]
        source = st.radio("Choose data source", _source_options, index=0)

        if source == "Use Project Dataset":
            if st.button("Load Dataset", use_container_width=True):
                try:
                    raw = pd.read_csv(_BUNDLED_CSV)
                    st.session_state["raw_df"] = raw
                    st.session_state["is_demo"] = False
                    for k in ["clean_df", "clean_report", "prep_data", "results_df",
                              "trained_models", "best_model_name", "best_model",
                              "eval_info", "pred_df", "insights"]:
                        st.session_state[k] = None
                    st.success(f"Loaded {len(raw):,} rows \u00d7 {raw.shape[1]} columns from project dataset.")
                except Exception as e:
                    st.error(f"Could not read project dataset: {e}")

        elif source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload customer CSV", type=["csv"],
                help="Must contain a 'Churn' column (Yes/No or 1/0)."
            )
            if uploaded:
                try:
                    raw = pd.read_csv(uploaded)
                    st.session_state["raw_df"] = raw
                    st.session_state["is_demo"] = False
                    st.success(f"Loaded {len(raw):,} rows \u00d7 {raw.shape[1]} columns")
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
        else:
            n_rows = st.slider("Demo dataset size", 500, 5000, 1000, step=500)
            if st.button("Generate Demo Data"):
                demo = generate_sample_dataset(n=n_rows)
                st.session_state["raw_df"] = demo
                st.session_state["is_demo"] = True
                st.success(f"Generated {n_rows:,} synthetic customers.")
                # Reset downstream state
                for k in ["clean_df", "clean_report", "prep_data", "results_df",
                          "trained_models", "best_model_name", "best_model",
                          "eval_info", "pred_df", "insights"]:
                    st.session_state[k] = None

        # ── Pipeline ────────────────────────────────────────────────────
        st.divider()
        st.subheader("⚙️ Pipeline")

        if st.button("1. Clean Data", use_container_width=True,
                     disabled=st.session_state["raw_df"] is None):
            with st.spinner("Cleaning data…"):
                clean_df, report = clean_data(st.session_state["raw_df"])
                st.session_state["clean_df"] = clean_df
                st.session_state["clean_report"] = report
                st.success("Data cleaned.")

        if st.button("2. Train Models", use_container_width=True,
                     disabled=st.session_state["clean_df"] is None):
            with st.spinner("Training models (this may take a minute)…"):
                _run_training()

        if st.button("3. Generate Insights", use_container_width=True,
                     disabled=st.session_state["pred_df"] is None):
            with st.spinner("Generating AI insights…"):
                insights = generate_insights(
                    st.session_state["clean_df"],
                    st.session_state["pred_df"],
                )
                st.session_state["insights"] = insights
                st.success("Insights ready.")

        # ── Risk thresholds ─────────────────────────────────────────────
        st.divider()
        st.subheader("🎚️ Risk Thresholds")
        low_t = st.slider("Low / Medium boundary", 0.10, 0.50, 0.30, 0.05)
        high_t = st.slider("Medium / High boundary", 0.40, 0.90, 0.60, 0.05)
        if low_t >= high_t:
            st.warning("Low threshold must be < High threshold.")
        else:
            st.session_state["low_threshold"] = low_t
            st.session_state["high_threshold"] = high_t

        # ── Navigation ──────────────────────────────────────────────────
        st.divider()
        page = st.radio(
            "Navigate",
            [
                "📊 Executive Overview",
                "👥 Customer Analysis",
                "🔍 Churn Drivers",
                "🤖 AI Insights",
                "💬 AI Data Analyst",
                "🔬 Data Cleaning Report",
                "🗄️ SQL Analysis",
            ],
        )
        return page


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_training():
    try:
        prep = prepare_data(st.session_state["clean_df"])
        results_df, trained = train_all_models(
            prep["X_train"], prep["X_test"],
            prep["y_train"], prep["y_test"],
        )
        best_name, best_model = select_best_model(results_df, trained)
        eval_info = detailed_evaluation(
            best_model,
            prep["X_test"],
            prep["y_test"],
            feature_names=prep["feature_names_transformed"],
        )
        eval_info["model_name"] = best_name

        pred_df = predict_churn(
            st.session_state["clean_df"],
            best_model,
            prep["preprocessor"],
            prep["feature_cols"],
            st.session_state["low_threshold"],
            st.session_state["high_threshold"],
        )
        pred_df = add_risk_explanations(pred_df)

        # Persist
        st.session_state.update({
            "prep_data": prep,
            "results_df": results_df,
            "trained_models": trained,
            "best_model_name": best_name,
            "best_model": best_model,
            "eval_info": eval_info,
            "pred_df": pred_df,
        })

        # Save model artifacts
        os.makedirs("models", exist_ok=True)
        save_model(best_model, {
            "model_name": best_name,
            "feature_cols": prep["feature_cols"],
            "feature_names_transformed": prep["feature_names_transformed"],
        })
        st.success(f"Best model: **{best_name}** (ROC-AUC = {eval_info['auc']:.4f})")
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# KPI card renderer
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_card(col, label: str, value, colour: str = "#6366F1", delta: str = ""):
    with col:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{colour}">{value}</div>
                <div class="kpi-label">{label}</div>
                {"<div class='kpi-delta'>" + delta + "</div>" if delta else ""}
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Executive Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview():
    st.title("📊 Executive Overview")

    if st.session_state["clean_df"] is None:
        st.info("👈 Load a dataset and click **Clean Data** to begin.")
        _show_getting_started()
        return

    df = st.session_state["clean_df"]
    pred_df = st.session_state["pred_df"]
    kpis = calculate_kpis(df, pred_df)

    if st.session_state["is_demo"]:
        st.warning("⚠️ Displaying **synthetic demo data**.  Upload a real CSV for production analysis.")

    # ── KPI Cards ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Business KPIs</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _kpi_card(c1, "Total Customers", f"{kpis['total_customers']:,}", "#6366F1")
    _kpi_card(c2, "Churned", f"{kpis['churned_customers']:,}", "#EF4444")
    _kpi_card(c3, "Active", f"{kpis['active_customers']:,}", "#22C55E")
    _kpi_card(c4, "Churn Rate", f"{kpis['churn_rate_pct']}%", "#F59E0B")
    _kpi_card(c5, "Avg Monthly Charges", format_currency(kpis["avg_monthly_charges"]), "#0EA5E9")
    _kpi_card(c6, "Avg Tenure (months)", str(kpis["avg_tenure_months"]), "#8B5CF6")

    st.markdown("<br>", unsafe_allow_html=True)

    if pred_df is not None:
        c7, c8, c9 = st.columns(3)
        _kpi_card(c7, "🔴 High Risk", f"{kpis['high_risk_customers']:,}", "#EF4444")
        _kpi_card(c8, "🟡 Medium Risk", f"{kpis['medium_risk_customers']:,}", "#F59E0B")
        _kpi_card(c9, "🟢 Low Risk", f"{kpis['low_risk_customers']:,}", "#22C55E")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_churn_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(fig_monthly_charges_distribution(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_contract_distribution(df), use_container_width=True)
    with col4:
        st.plotly_chart(fig_churn_by_tenure_bin(df), use_container_width=True)

    if pred_df is not None:
        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(fig_risk_distribution(pred_df), use_container_width=True)
        with col6:
            st.plotly_chart(
                fig_churn_probability_histogram(
                    pred_df,
                    low_threshold=st.session_state["low_threshold"],
                    high_threshold=st.session_state["high_threshold"],
                ),
                use_container_width=True,
            )

    # ── Model summary ──────────────────────────────────────────────────────
    if st.session_state["eval_info"]:
        eval_info = st.session_state["eval_info"]
        st.markdown('<p class="section-header">Model Performance Summary</p>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        _kpi_card(mc1, "Best Model", eval_info["model_name"], "#6366F1")
        _kpi_card(mc2, "Accuracy", f"{eval_info['report_dict']['accuracy']:.3f}", "#6366F1")
        _kpi_card(mc3, "Recall", f"{eval_info['report_dict']['Churned (1)']['recall']:.3f}", "#EF4444")
        _kpi_card(mc4, "F1 Score", f"{eval_info['report_dict']['Churned (1)']['f1-score']:.3f}", "#F59E0B")
        _kpi_card(mc5, "ROC-AUC", f"{eval_info['auc']:.4f}", "#22C55E")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Customer Analysis
# ─────────────────────────────────────────────────────────────────────────────

def page_customer_analysis():
    st.title("👥 Customer Analysis")

    if st.session_state["pred_df"] is None:
        st.info("Run the full pipeline (Clean → Train) to see customer-level predictions.")
        return

    pred_df = st.session_state["pred_df"]

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Filters</p>', unsafe_allow_html=True)
    fc = st.columns(6)

    risk_filter = fc[0].multiselect(
        "Risk Level", ["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"],
    )
    _gender_col = next((c for c in ["gender", "Gender"] if c in pred_df.columns), None)
    gender_opts = sorted(pred_df[_gender_col].dropna().unique().tolist()) if _gender_col else []
    gender_filter = fc[1].multiselect("Gender", gender_opts, default=gender_opts)

    contract_opts = sorted(pred_df["Contract"].dropna().unique().tolist()) if "Contract" in pred_df.columns else []
    contract_filter = fc[2].multiselect("Contract", contract_opts, default=contract_opts)

    internet_opts = sorted(pred_df["InternetService"].dropna().unique().tolist()) if "InternetService" in pred_df.columns else []
    internet_filter = fc[3].multiselect("Internet", internet_opts, default=internet_opts)

    payment_opts = sorted(pred_df["PaymentMethod"].dropna().unique().tolist()) if "PaymentMethod" in pred_df.columns else []
    payment_filter = fc[4].multiselect("Payment", payment_opts, default=payment_opts)

    senior_opts = [0, 1] if "SeniorCitizen" in pred_df.columns else []
    senior_filter = fc[5].multiselect("Senior", senior_opts, default=senior_opts,
                                       format_func=lambda x: "Senior" if x == 1 else "Non-Senior")

    # Apply filters
    mask = pred_df["Risk Level"].isin(risk_filter)
    if gender_filter and _gender_col:
        mask &= pred_df[_gender_col].isin(gender_filter)
    if contract_filter and "Contract" in pred_df.columns:
        mask &= pred_df["Contract"].isin(contract_filter)
    if internet_filter and "InternetService" in pred_df.columns:
        mask &= pred_df["InternetService"].isin(internet_filter)
    if payment_filter and "PaymentMethod" in pred_df.columns:
        mask &= pred_df["PaymentMethod"].isin(payment_filter)
    if senior_filter and "SeniorCitizen" in pred_df.columns:
        mask &= pred_df["SeniorCitizen"].isin(senior_filter)

    filtered = pred_df[mask].copy()

    st.caption(f"Showing **{len(filtered):,}** of {len(pred_df):,} customers")

    # ── Customer table ─────────────────────────────────────────────────────
    id_col = next((c for c in pred_df.columns if "id" in c.lower()), None)
    display_cols = (
        [id_col] if id_col else []
    ) + [
        c for c in ["Gender", "gender", "Age", "SeniorCitizen", "Contract",
                    "tenure", "TenureMonths", "MonthlyCharges", "InternetService",
                    "PaymentMethod", "SupportCalls",
                    "Churn Probability", "Risk Level", "Risk Factors"]
        if c in filtered.columns
    ]

    # Colour risk column
    def colour_risk(val):
        colours = {"High Risk": "color: #EF4444; font-weight:600",
                   "Medium Risk": "color: #F59E0B; font-weight:600",
                   "Low Risk": "color: #22C55E; font-weight:600"}
        return colours.get(val, "")

    styled = (
        filtered[display_cols]
        .sort_values("Churn Probability", ascending=False)
        .style
        .applymap(colour_risk, subset=["Risk Level"])
        .format({"Churn Probability": "{:.2%}"})
        .background_gradient(subset=["Churn Probability"], cmap="RdYlGn_r")
    )
    st.dataframe(styled, use_container_width=True, height=450)

    # ── Download ───────────────────────────────────────────────────────────
    st.divider()
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "⬇️ Download Filtered Predictions (CSV)",
            data=df_to_csv_bytes(filtered[display_cols]),
            file_name="filtered_predictions.csv",
            mime="text/csv",
        )
    with dl2:
        high_risk = pred_df[pred_df["Risk Level"] == "High Risk"]
        st.download_button(
            "⬇️ Download High-Risk Customers (CSV)",
            data=df_to_csv_bytes(high_risk),
            file_name="high_risk_customers.csv",
            mime="text/csv",
        )
    with dl3:
        st.download_button(
            "⬇️ Download All Predictions (CSV)",
            data=df_to_csv_bytes(pred_df),
            file_name="all_predictions.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Churn Drivers
# ─────────────────────────────────────────────────────────────────────────────

def page_churn_drivers():
    st.title("🔍 Churn Drivers")

    if st.session_state["clean_df"] is None:
        st.info("Load and clean a dataset first.")
        return

    df = st.session_state["clean_df"]

    # ── Feature importance ─────────────────────────────────────────────────
    if st.session_state["eval_info"] and st.session_state["eval_info"].get("feature_importances") is not None:
        st.markdown('<p class="section-header">Feature Importance (Best Model)</p>', unsafe_allow_html=True)
        ei = st.session_state["eval_info"]
        st.plotly_chart(
            fig_feature_importance(ei["feature_names"], ei["feature_importances"]),
            use_container_width=True,
        )
    else:
        st.info("Train a model to see feature importance.")

    st.divider()

    # ── Churn analysis charts ──────────────────────────────────────────────
    st.markdown('<p class="section-header">Churn Breakdown</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_contract_distribution(df), use_container_width=True)
        st.plotly_chart(fig_churn_by_internet(df), use_container_width=True)
        st.plotly_chart(fig_churn_by_senior(df), use_container_width=True)
    with col2:
        st.plotly_chart(fig_churn_by_tenure_bin(df), use_container_width=True)
        st.plotly_chart(fig_churn_by_charges_bin(df), use_container_width=True)
        st.plotly_chart(fig_churn_by_tech_support(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_payment_method(df), use_container_width=True)
    with col4:
        st.plotly_chart(fig_churn_by_online_security(df), use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_correlation_heatmap(df), use_container_width=True)

    # ── Model comparison ───────────────────────────────────────────────────
    if st.session_state["results_df"] is not None:
        st.divider()
        st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
        st.plotly_chart(fig_model_comparison(st.session_state["results_df"]), use_container_width=True)
        st.dataframe(st.session_state["results_df"], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — AI Insights
# ─────────────────────────────────────────────────────────────────────────────

def page_ai_insights():
    st.title("🤖 AI Insights")

    if st.session_state["clean_df"] is None:
        st.info("Load and clean a dataset first.")
        return

    df = st.session_state["clean_df"]
    pred_df = st.session_state["pred_df"]

    # Auto-generate if not already done
    if st.session_state["insights"] is None:
        with st.spinner("Generating insights…"):
            st.session_state["insights"] = generate_insights(df, pred_df)

    insights = st.session_state["insights"]
    mode = insights.get("mode", "Rule-Based")

    st.caption(f"Insight mode: **{mode}**  {'🤖 LLM' if 'LLM' in mode else '📐 Statistical rule-based'}")

    if "LLM" not in mode:
        st.info(
            "ℹ️ Running in **rule-based mode** (no AI API key configured).  "
            "Add `AI_API_KEY=your_key` to a `.env` file to enable LLM-enhanced insights."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Key Findings", "🔍 Churn Drivers", "👥 Segment Insights", "💡 Recommendations"]
    )

    with tab1:
        st.markdown(insights.get("key_findings", "No findings available."))
    with tab2:
        st.markdown(insights.get("churn_drivers", "No churn driver info available."))
    with tab3:
        st.markdown(insights.get("segment_insights", "No segment insights available."))
    with tab4:
        st.markdown(
            '<div class="warning-box">⚠️ The following are <strong>data-informed suggestions</strong>, '
            "not guaranteed outcomes.  Always test initiatives before full rollout.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(insights.get("recommendations", "No recommendations available."))

    st.divider()

    # ── Model Evaluation ───────────────────────────────────────────────────
    if st.session_state["eval_info"]:
        st.markdown('<p class="section-header">Model Evaluation</p>', unsafe_allow_html=True)
        ei = st.session_state["eval_info"]

        ec1, ec2 = st.columns(2)
        with ec1:
            st.plotly_chart(
                fig_confusion_matrix(ei["confusion_matrix"]),
                use_container_width=True,
            )
        with ec2:
            st.plotly_chart(
                fig_roc_curve(ei["fpr"], ei["tpr"], ei["auc"]),
                use_container_width=True,
            )

        with st.expander("📋 Full Classification Report"):
            st.text(ei["report_text"])

        with st.expander("⚠️ Model Limitations"):
            st.markdown(
                """
- **Dataset size** — model accuracy depends on having sufficient training data.
- **Class imbalance** — churned customers are a minority; recall may still miss some.
- **Historical data** — past patterns may not fully predict future behaviour.
- **No causal inference** — high feature importance ≠ the cause of churn.
- **Data leakage risk** — if TotalCharges encodes tenure information, it may inflate metrics.
- **Model generalisation** — performance on new populations may differ from test set.
                """
            )

    # ── Downloads ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Download Reports</p>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    kpis = calculate_kpis(df, pred_df)
    report_text = generate_text_report(
        kpis, insights,
        eval_info=st.session_state.get("eval_info"),
    )
    with dl1:
        st.download_button(
            "⬇️ Download Text Report",
            data=report_text.encode("utf-8"),
            file_name="churn_analysis_report.txt",
            mime="text/plain",
        )
    with dl2:
        st.download_button(
            "⬇️ Download Cleaned Data (CSV)",
            data=df_to_csv_bytes(df),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    if pred_df is not None:
        excel_data = df_to_excel_bytes({
            "Predictions": pred_df,
            "High Risk": pred_df[pred_df["Risk Level"] == "High Risk"],
            "Cleaned Data": df,
        })
        with dl3:
            st.download_button(
                "⬇️ Download Excel Workbook",
                data=excel_data,
                file_name="churn_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — AI Data Analyst
# ─────────────────────────────────────────────────────────────────────────────

def page_ai_analyst():
    st.title("💬 AI Data Analyst")
    st.caption(
        "Ask natural-language questions about your dataset.  "
        "Answers are based **only** on actual calculated statistics — no hallucinations."
    )

    if st.session_state["clean_df"] is None:
        st.info("Load and clean a dataset first.")
        return

    # ── Example questions ──────────────────────────────────────────────────
    st.markdown("**Example questions:**")
    examples = [
        "What is the current churn rate?",
        "Which contract type has the highest churn?",
        "What are the main reasons customers are leaving?",
        "How many high-risk customers are there?",
        "What is the average monthly charge for churned customers?",
        "Which segment should the company investigate first?",
        "What payment methods are associated with high churn?",
        "What is the average tenure of churned vs active customers?",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i % 4].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["qa_input"] = ex

    st.divider()

    # ── Question input ─────────────────────────────────────────────────────
    question = st.text_input(
        "Ask a question about the data:",
        value=st.session_state.get("qa_input", ""),
        placeholder="e.g. What is the churn rate for fiber optic customers?",
        key="qa_text_input",
    )
    if question:
        st.session_state["qa_input"] = ""
        with st.spinner("Analysing…"):
            answer = answer_question(
                question,
                st.session_state["clean_df"],
                st.session_state["pred_df"],
            )
        st.markdown(
            f'<div class="insight-box"><strong>Q:</strong> {question}<br><br>'
            f"<strong>A:</strong> {answer}</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Data Cleaning Report
# ─────────────────────────────────────────────────────────────────────────────

def page_cleaning_report():
    st.title("🔬 Data Cleaning Report")

    if st.session_state["raw_df"] is None:
        st.info("Load a dataset first.")
        return

    if st.session_state["clean_df"] is None:
        st.info("Click **Clean Data** in the sidebar to run the cleaning pipeline.")
        return

    raw_df = st.session_state["raw_df"]
    clean_df = st.session_state["clean_df"]
    report = st.session_state["clean_report"]

    # ── Before / after summary ─────────────────────────────────────────────
    st.markdown('<p class="section-header">Before vs After Cleaning</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, "Rows Before", f"{report['rows_before']:,}", "#6366F1")
    _kpi_card(c2, "Rows After", f"{report['rows_after']:,}", "#22C55E")
    _kpi_card(c3, "Nulls Before", f"{report['nulls_before']:,}", "#EF4444")
    _kpi_card(c4, "Nulls After", f"{report['nulls_after']:,}", "#22C55E")

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    _kpi_card(c5, "Duplicate Rows Removed", str(report["duplicates_removed"]), "#F59E0B")
    _kpi_card(c6, "Churn Column Encoded", "Yes" if report["churn_encoded"] else "No", "#0EA5E9")

    st.divider()

    # ── Tabs: raw / clean ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Cleaned Data", "Missing Values", "Outlier Report"])

    with tab1:
        st.caption(f"Shape: {raw_df.shape}")
        st.dataframe(raw_df.head(100), use_container_width=True)

    with tab2:
        st.caption(f"Shape: {clean_df.shape}")
        st.dataframe(clean_df.head(100), use_container_width=True)

    with tab3:
        raw_nulls = raw_df.isnull().sum().reset_index()
        raw_nulls.columns = ["Column", "Missing (Raw)"]
        clean_nulls = clean_df.isnull().sum().reset_index()
        clean_nulls.columns = ["Column", "Missing (Clean)"]
        null_df = raw_nulls.merge(clean_nulls, on="Column", how="outer").fillna(0)
        null_df["Resolved?"] = null_df["Missing (Clean)"] == 0
        st.dataframe(null_df, use_container_width=True)

    with tab4:
        outliers = outlier_report(clean_df)
        st.dataframe(outliers, use_container_width=True)
        st.info(
            "Outliers in customer data (e.g. very high TotalCharges, long tenure) are "
            "**retained** — they represent real customer behaviour and carry predictive value."
        )

    # ── EDA charts ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_gender_distribution(clean_df), use_container_width=True)
        st.plotly_chart(fig_tenure_distribution(clean_df), use_container_width=True)
    with col2:
        st.plotly_chart(fig_senior_citizen(clean_df), use_container_width=True)
        st.plotly_chart(fig_total_charges_distribution(clean_df), use_container_width=True)

    st.plotly_chart(fig_partner_dependents(clean_df), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — SQL Analysis
# ─────────────────────────────────────────────────────────────────────────────

def page_sql_analysis():
    st.title("🗄️ SQL Analysis")
    st.caption(
        "Business SQL queries for the churn dataset.  "
        "Results are computed via pandas on the cleaned dataset."
    )

    if st.session_state["clean_df"] is None:
        st.info("Load and clean a dataset first.")
        return

    df = st.session_state["clean_df"]

    for query_name, (description, sql_text) in SQL_QUERIES.items():
        with st.expander(f"📌 {query_name}"):
            st.caption(description)
            st.code(sql_text, language="sql")
            # Execute via pandas equivalent
            result = _execute_query(query_name, df)
            if result is not None:
                st.dataframe(result, use_container_width=True)


def _execute_query(name: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """Execute a named query against the pandas DataFrame."""
    try:
        if name == "Total Customers":
            return pd.DataFrame({"total_customers": [len(df)]})

        if name == "Total Churned Customers":
            return pd.DataFrame({"churned_customers": [int(df["Churn"].sum())]})

        if name == "Overall Churn Rate":
            return pd.DataFrame({"churn_rate_pct": [round(df["Churn"].mean() * 100, 2)]})

        if name == "Average Monthly Charges":
            return pd.DataFrame({
                "avg_monthly_charges": [round(df["MonthlyCharges"].mean(), 2)],
                "avg_charges_churned": [round(df[df["Churn"] == 1]["MonthlyCharges"].mean(), 2)],
                "avg_charges_active": [round(df[df["Churn"] == 0]["MonthlyCharges"].mean(), 2)],
            })

        if name == "Churn by Contract Type" and "Contract" in df.columns:
            g = df.groupby("Contract").agg(
                total_customers=("Churn", "count"),
                churned=("Churn", "sum"),
            ).reset_index()
            g["churn_rate_pct"] = (g["churned"] / g["total_customers"] * 100).round(2)
            return g.sort_values("churn_rate_pct", ascending=False)

        if name == "Churn by Payment Method" and "PaymentMethod" in df.columns:
            g = df.groupby("PaymentMethod").agg(
                total_customers=("Churn", "count"),
                churned=("Churn", "sum"),
            ).reset_index()
            g["churn_rate_pct"] = (g["churned"] / g["total_customers"] * 100).round(2)
            return g.sort_values("churn_rate_pct", ascending=False)

        if name == "Churn by Internet Service" and "InternetService" in df.columns:
            g = df.groupby("InternetService").agg(
                total_customers=("Churn", "count"),
                churned=("Churn", "sum"),
            ).reset_index()
            g["churn_rate_pct"] = (g["churned"] / g["total_customers"] * 100).round(2)
            return g.sort_values("churn_rate_pct", ascending=False)

        if name == "Top Churn Segments" and "Contract" in df.columns and "InternetService" in df.columns:
            g = df.groupby(["Contract", "InternetService"]).agg(
                total_customers=("Churn", "count"),
                churned=("Churn", "sum"),
            ).reset_index()
            g["churn_rate_pct"] = (g["churned"] / g["total_customers"] * 100).round(2)
            return g.sort_values("churn_rate_pct", ascending=False).head(10)

        if name == "Average Tenure: Churned vs Active":
            tenure_col = next((c for c in ["tenure", "TenureMonths"] if c in df.columns), None)
            if tenure_col is None:
                return None
            g = df.groupby("Churn").agg(
                count=(tenure_col, "count"),
                avg_tenure_months=(tenure_col, "mean"),
                min_tenure=(tenure_col, "min"),
                max_tenure=(tenure_col, "max"),
            ).reset_index()
            g["customer_status"] = g["Churn"].map({1: "Churned", 0: "Active"})
            g = g.drop(columns=["Churn"])
            for col in ["avg_tenure_months", "min_tenure", "max_tenure"]:
                g[col] = g[col].round(2)
            return g

        if name == "Revenue from Churned Customers" and "TotalCharges" in df.columns:
            churned = df[df["Churn"] == 1]
            return pd.DataFrame({
                "total_revenue_churned": [round(churned["TotalCharges"].sum(), 2)],
                "avg_revenue_per_churned": [round(churned["TotalCharges"].mean(), 2)],
                "churned_count": [len(churned)],
            })

    except Exception as e:
        st.error(f"Query error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Getting started helper
# ─────────────────────────────────────────────────────────────────────────────

def _show_getting_started():
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    st.markdown(
        """
**Step 1 — Load Data**

- In the sidebar, choose **Use Demo Dataset** (no file required) and click **Generate Demo Data**.
- Or select **Upload CSV** and upload the IBM Telco Customer Churn CSV.

**Step 2 — Clean Data**

- Click **1. Clean Data** in the sidebar.

**Step 3 — Train Models**

- Click **2. Train Models**.  This runs Logistic Regression, Random Forest, Gradient Boosting, and XGBoost (if installed).

**Step 4 — Generate Insights**

- Click **3. Generate Insights** to produce AI/statistical recommendations.

**Step 5 — Explore the Dashboard**

- Navigate through the pages using the sidebar.

---
**Dataset**:  [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main router
# ─────────────────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "📊 Executive Overview":
        page_overview()
    elif page == "👥 Customer Analysis":
        page_customer_analysis()
    elif page == "🔍 Churn Drivers":
        page_churn_drivers()
    elif page == "🤖 AI Insights":
        page_ai_insights()
    elif page == "💬 AI Data Analyst":
        page_ai_analyst()
    elif page == "🔬 Data Cleaning Report":
        page_cleaning_report()
    elif page == "🗄️ SQL Analysis":
        page_sql_analysis()


if __name__ == "__main__":
    main()
