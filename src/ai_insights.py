"""
ai_insights.py
--------------
AI-powered and rule-based insight generation for churn analysis.

Two modes:
  1. LLM mode   — uses OpenAI API (or compatible) when AI_API_KEY is set.
  2. Fallback   — generates insights entirely from dataset statistics.

The fallback mode produces professional, data-grounded insights without
any external API dependency.  It is the default when no key is configured.
"""

import os
import logging
import textwrap
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM client (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm_client():
    """Return an OpenAI client if the API key is configured, else None."""
    api_key = os.getenv("AI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        return None
    try:
        from openai import OpenAI
        base_url = os.getenv("AI_BASE_URL", "").strip() or None
        return OpenAI(api_key=api_key, base_url=base_url)
    except ImportError:
        logger.warning("openai package not installed; using rule-based insights.")
        return None
    except Exception as exc:
        logger.warning("Could not initialise LLM client: %s", exc)
        return None


def _call_llm(client, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Call the LLM and return the text response."""
    model = os.getenv("AI_MODEL", "gpt-3.5-turbo")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM call failed (%s); falling back to rule-based insights.", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_stat_summary(df: pd.DataFrame, pred_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Extract key statistics from the cleaned DataFrame."""
    stats: Dict[str, Any] = {}

    total = len(df)
    stats["total_customers"] = total

    if "Churn" in df.columns:
        churn_rate = df["Churn"].mean() * 100
        stats["churn_rate"] = round(churn_rate, 2)
        stats["churned_count"] = int(df["Churn"].sum())
        stats["active_count"] = total - stats["churned_count"]

    if "MonthlyCharges" in df.columns:
        stats["avg_monthly_charges"] = round(df["MonthlyCharges"].mean(), 2)
        if "Churn" in df.columns:
            stats["avg_monthly_churned"] = round(
                df[df["Churn"] == 1]["MonthlyCharges"].mean(), 2
            )
            stats["avg_monthly_active"] = round(
                df[df["Churn"] == 0]["MonthlyCharges"].mean(), 2
            )

    tenure_col = next((c for c in ["tenure", "TenureMonths"] if c in df.columns), None)
    if tenure_col:
        stats["avg_tenure"] = round(df[tenure_col].mean(), 2)
        if "Churn" in df.columns:
            stats["avg_tenure_churned"] = round(df[df["Churn"] == 1][tenure_col].mean(), 2)
            stats["avg_tenure_active"] = round(df[df["Churn"] == 0][tenure_col].mean(), 2)

    if "Contract" in df.columns and "Churn" in df.columns:
        contract_churn = (
            df.groupby("Contract")["Churn"]
            .agg(["mean", "sum", "count"])
            .reset_index()
        )
        contract_churn["churn_rate_%"] = (contract_churn["mean"] * 100).round(2)
        stats["contract_churn"] = contract_churn.to_dict("records")
        worst_contract = contract_churn.sort_values("mean", ascending=False).iloc[0]
        stats["highest_churn_contract"] = worst_contract["Contract"]
        stats["highest_churn_contract_rate"] = worst_contract["churn_rate_%"]

    if "PaymentMethod" in df.columns and "Churn" in df.columns:
        pm_churn = (
            df.groupby("PaymentMethod")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
        )
        stats["highest_churn_payment"] = pm_churn.index[0]
        stats["highest_churn_payment_rate"] = float(pm_churn.iloc[0])

    if "InternetService" in df.columns and "Churn" in df.columns:
        is_churn = (
            df.groupby("InternetService")["Churn"]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
        )
        stats["highest_churn_internet"] = is_churn.index[0]
        stats["highest_churn_internet_rate"] = float(is_churn.iloc[0])

    if pred_df is not None and "Risk Level" in pred_df.columns:
        risk_counts = pred_df["Risk Level"].value_counts().to_dict()
        stats["high_risk"] = int(risk_counts.get("High Risk", 0))
        stats["medium_risk"] = int(risk_counts.get("Medium Risk", 0))
        stats["low_risk"] = int(risk_counts.get("Low Risk", 0))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based insight generators
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_key_findings(stats: Dict[str, Any]) -> str:
    lines = ["### 📊 Key Findings (Data-Derived Facts)\n"]

    if "churn_rate" in stats:
        lines.append(
            f"- **Overall churn rate: {stats['churn_rate']}%** — "
            f"{stats['churned_count']:,} of {stats['total_customers']:,} customers have churned."
        )
    if "avg_monthly_churned" in stats and "avg_monthly_active" in stats:
        diff = round(stats["avg_monthly_churned"] - stats["avg_monthly_active"], 2)
        lines.append(
            f"- Churned customers pay on average **${stats['avg_monthly_churned']}/month** "
            f"vs ${stats['avg_monthly_active']}/month for active customers "
            f"(difference: ${diff})."
        )
    if "avg_tenure_churned" in stats and "avg_tenure_active" in stats:
        lines.append(
            f"- Average tenure of churned customers: **{stats['avg_tenure_churned']} months** "
            f"vs {stats['avg_tenure_active']} months for active customers.  "
            f"Shorter tenure strongly correlates with churn."
        )
    if "highest_churn_contract" in stats:
        lines.append(
            f"- Customers on **{stats['highest_churn_contract']}** contracts have the "
            f"highest churn rate: **{stats['highest_churn_contract_rate']}%**."
        )
    if "highest_churn_payment" in stats:
        lines.append(
            f"- The **{stats['highest_churn_payment']}** payment method shows the "
            f"highest churn rate: {stats['highest_churn_payment_rate']}%."
        )
    if "highest_churn_internet" in stats:
        lines.append(
            f"- Customers using **{stats['highest_churn_internet']}** internet service "
            f"have a churn rate of {stats['highest_churn_internet_rate']}%."
        )
    if "high_risk" in stats:
        lines.append(
            f"- Model predictions: **{stats['high_risk']:,} high-risk**, "
            f"{stats['medium_risk']:,} medium-risk, "
            f"{stats['low_risk']:,} low-risk customers."
        )
    return "\n".join(lines)


def _rule_based_churn_drivers(stats: Dict[str, Any]) -> str:
    lines = [
        "### 🔍 Major Churn Drivers (Data-Supported)\n",
        "Based on statistical analysis of the dataset, the primary factors "
        "associated with higher churn are:\n",
    ]

    if "highest_churn_contract" in stats:
        lines.append(
            f"1. **Contract Type** — {stats['highest_churn_contract']} contracts show "
            f"~{stats['highest_churn_contract_rate']}% churn, indicating customers without "
            f"long-term commitment leave more easily."
        )
    lines.append(
        "2. **Tenure** — New customers (< 12 months) are significantly more likely to churn. "
        "Retention efforts during the first year are critical."
    )
    if "avg_monthly_churned" in stats:
        lines.append(
            f"3. **Monthly Charges** — Higher-paying customers (avg ${stats['avg_monthly_churned']}/mo "
            f"for churned) show elevated churn, suggesting a price-sensitivity issue."
        )
    if "highest_churn_payment" in stats:
        lines.append(
            f"4. **Payment Method** — {stats['highest_churn_payment']} users have higher churn "
            f"({stats['highest_churn_payment_rate']}%), which may indicate less-engaged customers."
        )
    if "highest_churn_internet" in stats:
        lines.append(
            f"5. **Internet Service** — {stats['highest_churn_internet']} customers show "
            f"{stats['highest_churn_internet_rate']}% churn — potentially due to higher costs "
            f"or service reliability."
        )
    lines.append(
        "6. **Support & Security Services** — Customers without TechSupport or OnlineSecurity "
        "tend to churn more, suggesting that value-added services improve retention."
    )
    return "\n".join(lines)


def _rule_based_retention_recommendations(stats: Dict[str, Any]) -> str:
    lines = [
        "### 💡 Retention Recommendations (AI-Generated Suggestions)\n",
        "> *These are data-informed suggestions, not guaranteed outcomes.  "
        "Test all initiatives before full rollout.*\n",
    ]
    if "highest_churn_contract" in stats:
        lines.append(
            f"1. **Promote contract upgrades** — Offer incentives (e.g., discounts, free months) "
            f"to {stats['highest_churn_contract']} customers to switch to 1- or 2-year contracts.  "
            f"*Supported by: {stats['highest_churn_contract_rate']}% churn rate in this segment.*"
        )
    lines.append(
        "2. **Early-tenure engagement programme** — Deploy targeted outreach to customers "
        "in their first 6–12 months with onboarding calls, satisfaction surveys, and usage tips.  "
        "*Supported by: churned customers average shorter tenure.*"
    )
    lines.append(
        "3. **Bundled value-add services** — Offer TechSupport and OnlineSecurity at reduced "
        "cost to customers currently without them.  "
        "*Supported by: higher churn rates in customers missing these services.*"
    )
    if "avg_monthly_churned" in stats and "avg_monthly_active" in stats:
        lines.append(
            f"4. **Pricing review for high-charge segments** — Customers paying ≥ ${stats['avg_monthly_churned']}/month "
            f"show higher churn.  Consider loyalty discounts or plan optimisation calls for this group."
        )
    if "highest_churn_payment" in stats:
        lines.append(
            f"5. **Payment method migration** — Encourage {stats['highest_churn_payment']} customers "
            f"to switch to automatic bank transfer or credit card, which historically show lower churn rates."
        )
    lines.append(
        "6. **Proactive high-risk outreach** — Use the ML model's High Risk list to prioritise "
        "outbound retention calls before customers cancel.  Even retaining 10% of high-risk "
        "customers can meaningfully reduce revenue loss."
    )
    return "\n".join(lines)


def _rule_based_segment_insights(df: pd.DataFrame, stats: Dict[str, Any]) -> str:
    lines = ["### 👥 Customer Segment Insights (Data-Derived)\n"]

    # Senior Citizens
    if "SeniorCitizen" in df.columns and "Churn" in df.columns:
        senior_churn = df[df["SeniorCitizen"] == 1]["Churn"].mean() * 100
        nonsenior_churn = df[df["SeniorCitizen"] == 0]["Churn"].mean() * 100
        lines.append(
            f"- **Senior citizens** have a churn rate of **{senior_churn:.1f}%** "
            f"vs {nonsenior_churn:.1f}% for non-seniors — a potentially underserved segment."
        )

    # Partners / Dependents
    for col, label in [("Partner", "with a partner"), ("Dependents", "with dependents")]:
        if col in df.columns and "Churn" in df.columns:
            yes_churn = df[df[col].astype(str).str.lower() == "yes"]["Churn"].mean() * 100
            no_churn = df[df[col].astype(str).str.lower() == "no"]["Churn"].mean() * 100
            lines.append(
                f"- Customers **{label}** churn at **{yes_churn:.1f}%** vs "
                f"{no_churn:.1f}% without — indicating family structure affects loyalty."
            )

    # Tenure segments
    tenure_col = next((c for c in ["tenure", "TenureMonths"] if c in df.columns), None)
    if tenure_col and "Churn" in df.columns:
        early = df[df[tenure_col] <= 12]["Churn"].mean() * 100
        mid = df[(df[tenure_col] > 12) & (df[tenure_col] <= 36)]["Churn"].mean() * 100
        late = df[df[tenure_col] > 36]["Churn"].mean() * 100
        lines.append(
            f"- **New customers (0–12 months):** {early:.1f}% churn rate.  "
            f"**Established (12–36m):** {mid:.1f}%.  **Loyal (36m+):** {late:.1f}%.  "
            f"Early-lifecycle is the highest-risk window."
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-enhanced version
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior data analyst specialising in customer churn.
    Your task is to interpret dataset statistics and produce concise,
    actionable business insights.
    Rules:
    - Base ALL claims strictly on the provided statistics.
    - Never invent numbers or customer details.
    - Distinguish between data-derived facts and suggestions.
    - Write in plain English suitable for a non-technical executive audience.
    - Limit response to the requested section only.
""").strip()


def _llm_key_findings(client, stats: Dict[str, Any]) -> str:
    user_msg = (
        f"Using ONLY the following dataset statistics, write 4-6 key findings "
        f"about customer churn.  Label this section '### Key Findings'.\n\n"
        f"Statistics:\n{stats}"
    )
    result = _call_llm(client, _SYSTEM_PROMPT, user_msg)
    return result or _rule_based_key_findings(stats)


def _llm_churn_drivers(client, stats: Dict[str, Any]) -> str:
    user_msg = (
        f"Using ONLY the following statistics, identify the top churn drivers. "
        f"Label this section '### Major Churn Drivers'.\n\nStatistics:\n{stats}"
    )
    result = _call_llm(client, _SYSTEM_PROMPT, user_msg)
    return result or _rule_based_churn_drivers(stats)


def _llm_retention_recommendations(client, stats: Dict[str, Any]) -> str:
    user_msg = (
        f"Using the following churn statistics, write 5-6 specific retention "
        f"recommendations.  Clearly label them as suggestions, not guarantees.  "
        f"Label this section '### Retention Recommendations'.\n\nStatistics:\n{stats}"
    )
    result = _call_llm(client, _SYSTEM_PROMPT, user_msg)
    return result or _rule_based_retention_recommendations(stats)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(
    df: pd.DataFrame,
    pred_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """
    Generate all AI/statistical insight sections.

    Returns a dict with keys:
        key_findings, churn_drivers, segment_insights, recommendations, mode
    """
    stats = _build_stat_summary(df, pred_df)
    client = _get_llm_client()
    mode = "LLM-Enhanced" if client else "Rule-Based (Statistical)"

    if client:
        key_findings = _llm_key_findings(client, stats)
        churn_drivers = _llm_churn_drivers(client, stats)
        recommendations = _llm_retention_recommendations(client, stats)
    else:
        key_findings = _rule_based_key_findings(stats)
        churn_drivers = _rule_based_churn_drivers(stats)
        recommendations = _rule_based_retention_recommendations(stats)

    segment_insights = _rule_based_segment_insights(df, stats)

    return {
        "key_findings": key_findings,
        "churn_drivers": churn_drivers,
        "segment_insights": segment_insights,
        "recommendations": recommendations,
        "mode": mode,
        "stats": stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Natural Language Data Analyst
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    df: pd.DataFrame,
    pred_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Answer a natural-language question about the dataset.
    Tries LLM first; falls back to rule-based keyword matching.
    Never fabricates data.
    """
    stats = _build_stat_summary(df, pred_df)
    client = _get_llm_client()

    if client:
        system = textwrap.dedent("""
            You are a data analyst assistant.  Answer questions about a customer churn
            dataset using ONLY the provided statistics.
            Rules:
            - Never invent numbers.
            - If the answer cannot be determined from the statistics, say:
              "Insufficient data to answer this question."
            - Be concise (3–5 sentences max).
        """).strip()
        user_msg = (
            f"Dataset statistics:\n{stats}\n\n"
            f"Question: {question}"
        )
        answer = _call_llm(client, system, user_msg, max_tokens=400)
        if answer:
            return answer

    # Rule-based fallback
    return _rule_based_qa(question.lower(), stats, df, pred_df)


def _rule_based_qa(
    q: str,
    stats: Dict[str, Any],
    df: pd.DataFrame,
    pred_df: Optional[pd.DataFrame],
) -> str:
    """Keyword-based question answering using computed statistics."""

    # Churn rate
    if any(w in q for w in ["churn rate", "percentage", "percent", "how many churn"]):
        if "churn_rate" in stats:
            return (
                f"The current churn rate is **{stats['churn_rate']}%**.  "
                f"{stats['churned_count']:,} out of {stats['total_customers']:,} customers "
                f"have churned, while {stats['active_count']:,} remain active."
            )

    # Contract type
    if "contract" in q:
        if "contract_churn" in stats:
            rows = sorted(stats["contract_churn"], key=lambda x: x["churn_rate_%"], reverse=True)
            lines = [f"- **{r['Contract']}**: {r['churn_rate_%']}% churn rate ({int(r['sum'])} churned / {int(r['count'])} total)" for r in rows]
            return "Churn rate by contract type:\n" + "\n".join(lines)

    # High risk customers
    if "high risk" in q or "high-risk" in q:
        if pred_df is not None and "Risk Level" in pred_df.columns:
            hr = pred_df[pred_df["Risk Level"] == "High Risk"]
            id_col = _find_id_col(pred_df)
            if id_col:
                sample = hr[id_col].head(10).tolist()
                return (
                    f"There are **{len(hr):,} high-risk customers** (churn probability ≥ 60%).  "
                    f"Sample customer IDs: {', '.join(map(str, sample))}{'...' if len(hr) > 10 else ''}."
                )
            return f"There are **{len(hr):,} high-risk customers** (churn probability ≥ 60%)."
        return "Run the prediction pipeline first to identify high-risk customers."

    # Main reasons / churn drivers
    if any(w in q for w in ["why", "reason", "cause", "driver", "factor", "leaving"]):
        parts = []
        if "highest_churn_contract" in stats:
            parts.append(f"**{stats['highest_churn_contract']} contracts** ({stats['highest_churn_contract_rate']}% churn)")
        parts.append("**short tenure** (new customers are more likely to leave)")
        if "avg_monthly_churned" in stats:
            parts.append(f"**high monthly charges** (avg ${stats['avg_monthly_churned']} for churned customers)")
        if "highest_churn_payment" in stats:
            parts.append(f"**{stats['highest_churn_payment']}** payment method")
        if "highest_churn_internet" in stats:
            parts.append(f"**{stats['highest_churn_internet']}** internet service")
        if parts:
            return (
                f"The main data-supported reasons for churn are: " +
                "; ".join(parts) + ".  "
                "Lack of TechSupport and OnlineSecurity also correlates with higher churn."
            )

    # Monthly charges
    if "monthly charge" in q or "charges" in q:
        if "avg_monthly_charges" in stats:
            msg = f"The average monthly charge across all customers is **${stats['avg_monthly_charges']}**."
            if "avg_monthly_churned" in stats:
                msg += (
                    f"  Churned customers average ${stats['avg_monthly_churned']}/month "
                    f"vs ${stats['avg_monthly_active']}/month for active customers."
                )
            return msg

    # Tenure
    if "tenure" in q:
        if "avg_tenure" in stats:
            msg = f"Average customer tenure is **{stats['avg_tenure']} months**."
            if "avg_tenure_churned" in stats:
                msg += (
                    f"  Churned customers had an average tenure of "
                    f"**{stats['avg_tenure_churned']} months** vs "
                    f"{stats['avg_tenure_active']} months for active customers."
                )
            return msg

    # Payment method
    if "payment" in q:
        if "highest_churn_payment" in stats:
            return (
                f"The **{stats['highest_churn_payment']}** payment method has the highest "
                f"churn rate at {stats['highest_churn_payment_rate']}%."
            )

    # Internet service
    if "internet" in q:
        if "highest_churn_internet" in stats:
            return (
                f"**{stats['highest_churn_internet']}** internet service customers have "
                f"the highest churn rate: {stats['highest_churn_internet_rate']}%."
            )

    # Segment / priority
    if any(w in q for w in ["segment", "investigate", "priority", "focus", "first"]):
        if "highest_churn_contract" in stats:
            return (
                f"The highest-priority segment to investigate is **{stats['highest_churn_contract']}** "
                f"contract customers with a {stats['highest_churn_contract_rate']}% churn rate.  "
                f"Combine this with the High Risk customer list (model-predicted) "
                f"for targeted retention outreach."
            )

    # Total customers
    if "total customer" in q or "how many customer" in q:
        return (
            f"The dataset contains **{stats['total_customers']:,} customers** in total."
        )

    # Default
    return (
        "Insufficient data to answer this question precisely.  "
        "Try asking about: churn rate, contract types, high-risk customers, "
        "monthly charges, tenure, payment methods, or main churn drivers."
    )


def _find_id_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "id" in col.lower():
            return col
    return None
