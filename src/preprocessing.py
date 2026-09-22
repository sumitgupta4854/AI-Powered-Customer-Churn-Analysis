"""
preprocessing.py
----------------
Feature engineering and ML-ready preprocessing.

Steps
-----
1. Identify feature columns vs target.
2. Separate numeric and categorical features.
3. Scale numeric features (StandardScaler).
4. One-hot encode categorical features.
5. Return train/test splits + fitted transformers.
"""

import logging
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# Columns that should never be used as model features
_NON_FEATURE_COLS = {
    "customerid", "customer_id", "customerID", "churn", "churned",
    "is_churn", "ischurn", "churn_flag",
    "churn probability", "risk level", "risk_level",
    "predicted churn", "risk factors",
}


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Auto-detect feature columns and target column.

    Returns
    -------
    features : list of column names
    target   : name of the target column ('Churn')
    """
    target = None
    for col in df.columns:
        if col.lower() in {"churn", "churned", "is_churn", "ischurn", "churn_flag"}:
            target = col
            break
    if target is None:
        raise ValueError(
            "No churn target column found.  Expected a column named 'Churn', "
            "'Churned', 'is_churn', or similar."
        )

    features = [
        c for c in df.columns
        if c.lower() not in {x.lower() for x in _NON_FEATURE_COLS}
        and c not in _NON_FEATURE_COLS
    ]
    return features, target


def _is_numeric_col(series: pd.Series) -> bool:
    """Return True if the series holds numeric data (int or float)."""
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical_col(series: pd.Series) -> bool:
    """Return True if the series holds string/object/categorical data."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer:
      - numeric  → StandardScaler
      - category → OneHotEncoder (ignore unknown)
    """
    num_cols = [c for c in feature_cols if _is_numeric_col(df[c])]
    cat_cols = [c for c in feature_cols if _is_categorical_col(df[c])]

    logger.info("Numeric features  : %s", num_cols)
    logger.info("Categorical features: %s", cat_cols)

    transformers = []
    if num_cols:
        # Impute then scale — handles any residual NaNs after cleaning
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_feature_names_after_transform(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> List[str]:
    """Return the final feature names after the ColumnTransformer is fitted."""
    num_cols = [c for c in feature_cols if _is_numeric_col(df[c])]
    cat_cols = [c for c in feature_cols if _is_categorical_col(df[c])]

    names = list(num_cols)
    if cat_cols:
        cat_transformer = preprocessor.named_transformers_.get("cat")
        if cat_transformer is not None:
            # May be a Pipeline (cat_pipeline) or a bare OHE
            if hasattr(cat_transformer, "named_steps"):
                ohe = cat_transformer.named_steps.get("ohe")
            else:
                ohe = cat_transformer
            if ohe is not None:
                names += list(ohe.get_feature_names_out(cat_cols))
    return names


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline.

    Returns a dict with:
        X_train, X_test, y_train, y_test   – numpy arrays
        preprocessor                        – fitted ColumnTransformer
        feature_cols                        – original feature column names
        feature_names_transformed           – post-transform feature names
        target_col                          – name of the target column
        X_train_raw, X_test_raw             – raw DataFrames (for display)
    """
    feature_cols, target_col = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).values

    # Train / test split — stratified to preserve churn ratio
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(df, feature_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names_after_transform(preprocessor, df, feature_cols)

    logger.info(
        "Train: %d rows | Test: %d rows | Features: %d",
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "feature_names_transformed": feature_names,
        "target_col": target_col,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
    }


def save_preprocessor(preprocessor: ColumnTransformer, path: str = "models/preprocessor.pkl") -> None:
    joblib.dump(preprocessor, path)
    logger.info("Preprocessor saved to %s", path)


def load_preprocessor(path: str = "models/preprocessor.pkl") -> ColumnTransformer:
    return joblib.load(path)
