"""
model_training.py
-----------------
Train, evaluate, and compare multiple churn prediction models.

Models trained
--------------
1. Logistic Regression
2. Random Forest
3. Gradient Boosting (sklearn)
4. XGBoost (if available)

Each model is evaluated with:
  Accuracy, Precision, Recall, F1 Score, ROC-AUC

The best model is selected using ROC-AUC (better for imbalanced data).
"""

import logging
import os
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

MODEL_PATH = "models/churn_model.pkl"
METADATA_PATH = "models/model_metadata.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: try to import XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Build model registry
# ─────────────────────────────────────────────────────────────────────────────

def _get_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }
    if _xgboost_available():
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            eval_metric="logloss",
            random_state=42, scale_pos_weight=1,
        )
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train all models and return:
      - results_df : DataFrame of per-model metrics
      - trained    : dict mapping model name → fitted model
    """
    models = _get_models()
    results = []
    trained: Dict[str, Any] = {}

    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        trained[name] = model
        logger.info(
            "  %-25s Accuracy=%.3f  Recall=%.3f  ROC-AUC=%.3f",
            name, metrics["Accuracy"], metrics["Recall"], metrics["ROC-AUC"],
        )

    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ]
    return results_df, trained


def select_best_model(results_df: pd.DataFrame, trained: Dict[str, Any]) -> Tuple[str, Any]:
    """Select model with highest ROC-AUC (primary) and F1 Score (tie-break)."""
    best_row = results_df.sort_values(
        ["ROC-AUC", "F1 Score"], ascending=False
    ).iloc[0]
    best_name = best_row["Model"]
    logger.info("Best model selected: %s (ROC-AUC=%.4f)", best_name, best_row["ROC-AUC"])
    return best_name, trained[best_name]


# ─────────────────────────────────────────────────────────────────────────────
# Detailed evaluation for best model
# ─────────────────────────────────────────────────────────────────────────────

def detailed_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return a detailed evaluation dict for the selected model:
      cm, fpr, tpr, auc, report_text, feature_importances
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(
        y_test, y_pred,
        target_names=["Active (0)", "Churned (1)"],
        output_dict=True,
    )
    report_text = classification_report(
        y_test, y_pred,
        target_names=["Active (0)", "Churned (1)"],
    )

    # Feature importances
    importances = None
    names = feature_names or []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    return {
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": round(float(auc), 4),
        "report_dict": report,
        "report_text": report_text,
        "feature_importances": importances,
        "feature_names": names,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, metadata: Dict[str, Any], model_path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(metadata, METADATA_PATH)
    logger.info("Model saved to %s", model_path)


def load_model(model_path: str = MODEL_PATH) -> Tuple[Any, Dict[str, Any]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    return model, metadata
