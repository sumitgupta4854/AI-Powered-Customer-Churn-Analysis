"""
data_cleaning.py
----------------
Complete data-cleaning pipeline for the Customer Churn dataset.

Steps performed:
1. Load CSV / accept a pre-loaded DataFrame.
2. Audit raw data (shape, dtypes, nulls, duplicates).
3. Fix column names (strip whitespace).
4. Remove duplicate rows.
5. Handle empty strings → NaN.
6. Convert TotalCharges (and similar) to numeric.
7. Fill / drop missing values with a transparent strategy.
8. Standardise categorical values (strip, title-case).
9. Cast SeniorCitizen to int if present.
10. Encode binary target column (Churn: Yes→1, No→0).
11. Return clean DataFrame + audit report dict.
"""

import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Read a CSV file and return a raw DataFrame."""
    try:
        df = pd.read_csv(filepath)
        logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file: {exc}") from exc


def audit_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dictionary of raw-data quality metrics."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Core cleaning pipeline
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer data (from CSV or file upload).

    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned, type-corrected DataFrame ready for EDA / ML.
    report : dict
        Detailed before/after audit information.
    """
    df = df.copy()

    # ── 1. Audit before ────────────────────────────────────────────────────
    before = audit_dataframe(df)

    # ── 2. Normalise column names ──────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]
    df = _normalise_column_names(df)

    # ── 3. Drop exact duplicate rows ──────────────────────────────────────
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        df = df.drop_duplicates()
        logger.info("Removed %d duplicate rows.", n_dupes)

    # ── 4. Replace empty strings with NaN ─────────────────────────────────
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # ── 5. Numeric coercion ────────────────────────────────────────────────
    numeric_candidates = _detect_numeric_candidates(df)
    conversion_results: Dict[str, str] = {}
    for col in numeric_candidates:
        original_nulls = int(df[col].isnull().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = int(df[col].isnull().sum())
        conversion_results[col] = (
            f"converted (introduced {new_nulls - original_nulls} new NaNs)"
            if new_nulls > original_nulls
            else "converted"
        )
        logger.info("Column '%s': %s", col, conversion_results[col])

    # ── 6. Handle missing values ───────────────────────────────────────────
    missing_strategy: Dict[str, str] = {}
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            continue
        pct = n_missing / len(df)
        if pct > 0.5:
            # Drop columns with >50 % missing
            df.drop(columns=[col], inplace=True)
            missing_strategy[col] = "dropped (>50% missing)"
            logger.warning("Dropped column '%s' — %.0f%% missing.", col, pct * 100)
        elif pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            missing_strategy[col] = f"filled with median ({median_val:.2f})"
        else:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            missing_strategy[col] = f"filled with mode ('{fill_val}')"

    # ── 7. Standardise categorical values ─────────────────────────────────
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].str.strip()

    # ── 8. SeniorCitizen: keep as int (0/1) ───────────────────────────────
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # ── 9. Encode Churn target column ─────────────────────────────────────
    churn_col = _find_churn_column(df)
    churn_encoded = False
    if churn_col:
        # Handle any string dtype (object, ArrowStringArray, StringDtype, etc.)
        col_dtype = df[churn_col].dtype
        is_text = (
            col_dtype == object
            or pd.api.types.is_string_dtype(col_dtype)
            or str(col_dtype) in {"string", "str"}
        )
        if is_text:
            mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
            df[churn_col] = (
                df[churn_col]
                .astype(str)
                .str.lower()
                .map(mapping)
            )
            churn_encoded = True
        df[churn_col] = pd.to_numeric(df[churn_col], errors="coerce").fillna(0).astype(int)
        # Rename to canonical "Churn" if different
        if churn_col != "Churn":
            df.rename(columns={churn_col: "Churn"}, inplace=True)

    # ── 10. Audit after ────────────────────────────────────────────────────
    after = audit_dataframe(df)

    report = {
        "rows_before": before["n_rows"],
        "rows_after": after["n_rows"],
        "cols_before": before["n_cols"],
        "cols_after": after["n_cols"],
        "nulls_before": before["total_nulls"],
        "nulls_after": after["total_nulls"],
        "duplicates_removed": n_dupes,
        "numeric_conversions": conversion_results,
        "missing_value_strategy": missing_strategy,
        "churn_encoded": churn_encoded,
        "dtypes_after": after["dtypes"],
        "columns_after": after["columns"],
    }

    logger.info(
        "Cleaning complete: %d→%d rows, %d→%d nulls.",
        before["n_rows"], after["n_rows"],
        before["total_nulls"], after["total_nulls"],
    )
    return df, report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------

# Map of known variant column names → canonical name used throughout the app
_COLUMN_ALIASES = {
    # tenure variants
    "tenuremonths": "tenure",
    "tenure_months": "tenure",
    "months": "tenure",
    # customer id variants
    "customerid": "customerID",
    "customer_id": "customerID",
    "custid": "customerID",
    # age variants (keep as-is but also alias)
    # charges variants
    "monthly_charges": "MonthlyCharges",
    "monthlycharge": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "totalcharge": "TotalCharges",
    # support calls
    "supportcalls": "SupportCalls",
    "support_calls": "SupportCalls",
    "numcalls": "SupportCalls",
}


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column name variants to canonical names."""
    rename_map = {}
    for col in df.columns:
        # Normalise to lowercase, no spaces, no underscores — matches _COLUMN_ALIASES keys
        normalised = col.lower().replace(" ", "").replace("_", "")
        canonical = _COLUMN_ALIASES.get(normalised, None)
        if canonical and col != canonical:
            rename_map[col] = canonical
            logger.info("Renamed column '%s' → '%s'", col, canonical)
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _detect_numeric_candidates(df: pd.DataFrame) -> list:
    """
    Return object-dtype columns that are likely numeric.
    Heuristic: at least 60 % of non-null values parse as float.
    Always includes 'TotalCharges' and 'MonthlyCharges' if present.
    """
    forced = {"TotalCharges", "MonthlyCharges", "tenure", "TenureMonths"}
    candidates = []
    for col in df.select_dtypes(include="object").columns:
        if col in forced:
            candidates.append(col)
            continue
        sample = df[col].dropna().head(200)
        if len(sample) == 0:
            continue
        try:
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            if numeric_count / len(sample) >= 0.60:
                candidates.append(col)
        except Exception:
            pass
    return candidates


def _find_churn_column(df: pd.DataFrame) -> str | None:
    """Case-insensitive search for the churn target column."""
    for col in df.columns:
        if col.lower() in {"churn", "churned", "is_churn", "ischurn", "churn_flag"}:
            return col
    return None


# ---------------------------------------------------------------------------
# Outlier report (non-destructive — for display only)
# ---------------------------------------------------------------------------

def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising potential outliers in numeric columns
    using the IQR method.  Does NOT remove any rows — outliers in customer
    data often represent genuinely extreme (and valuable) customers.
    """
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append(
            {
                "Column": col,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Fence": round(lower, 2),
                "Upper Fence": round(upper, 2),
                "Outlier Count": n_outliers,
                "Outlier %": round(100 * n_outliers / len(df), 2),
                "Decision": (
                    "Retain — business outlier (e.g. long-tenure, high-spend customer)"
                    if n_outliers > 0
                    else "None"
                ),
            }
        )
    return pd.DataFrame(rows)
