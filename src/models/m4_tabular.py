"""
m4_tabular.py
=============
Module 4: Tabular Diabetes Risk Prediction
Dataset : NHANES (primary) + Pima Indians Diabetes (secondary/validation)
Model   : XGBoost (primary) with Random Forest fallback
XAI     : SHAP TreeExplainer

Paper equations:
  - XGBoost objective: L(phi) = SUM_i l(y_hat_i, y_i) + SUM_k Omega(f_k)
      where Omega(f) = gamma*T + (1/2)*lambda*||w||^2               (Eq. M4.1)
  - Gradient boosting: y_hat_i^(t) = y_hat_i^(t-1) + f_t(x_i)    (Eq. M4.2)
  - SHAP (Shapley formula — see shap_explainer.py for full formulation)

Reference: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System,"
KDD 2016. https://arxiv.org/abs/1603.02754
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import M4, SEED

# ─────────────────────────────────────────────
# 1. MODEL BUILDERS
# ─────────────────────────────────────────────

def build_xgboost_model():
    """Build an XGBClassifier with config hyperparameters."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(**M4["xgb"])
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")


def build_random_forest():
    """Build a RandomForestClassifier with config hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**M4["rf"])


def build_model(model_type: str = M4["model_type"]):
    """Factory function: return the configured model."""
    if model_type == "xgboost":
        return build_xgboost_model()
    elif model_type == "random_forest":
        return build_random_forest()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ─────────────────────────────────────────────
# 2. TRAINING
# ─────────────────────────────────────────────

def train_tabular_model(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame,   y_val: pd.Series,
                         model_type: str = M4["model_type"],
                         save_path: str = M4["model_path"]):
    """
    Train M4 model with early stopping (XGBoost) or full fit (RF).

    Returns
    -------
    model : fitted classifier
    """
    model = build_model(model_type)

    if model_type == "xgboost":
        # Use early stopping on validation set
        model.set_params(
            early_stopping_rounds=20,
            eval_metric="logloss",
            n_estimators=500,           # will stop early
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        print(f"[M4 XGB] Best iteration: {model.best_iteration}")
    else:
        model.fit(X_train, y_train)
        print("[M4 RF] Fitted successfully.")

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[M4] Model saved to {save_path}")

    # Val accuracy
    val_acc = (model.predict(X_val) == y_val).mean()
    print(f"[M4] Val Accuracy: {val_acc:.4f}")
    return model


# ─────────────────────────────────────────────
# 3. INFERENCE & RISK SCORES
# ─────────────────────────────────────────────

def extract_risk_scores(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return continuous risk probability (r_M4) in [0, 1].

    Returns
    -------
    risk_scores : (N,) float array — P(diabetes=1)
    """
    return model.predict_proba(X)[:, 1]


def predict_binary(model, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Binary diabetes prediction at a given probability threshold."""
    probs = extract_risk_scores(model, X)
    return (probs >= threshold).astype(int)


def load_trained_model(model_path: str = M4["model_path"]):
    """Load a saved M4 model from pickle."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[M4] Loaded model from {model_path}")
    return model


# ─────────────────────────────────────────────
# 4. NHANES DATA PREPARATION (pipeline)
# ─────────────────────────────────────────────

def build_nhanes_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the diabetes label from NHANES features.
    Label = 1 if fasting_glucose >= 126 mg/dL OR hba1c >= 6.5%
    (ADA diagnostic criteria: https://diabetes.org/about-diabetes/diagnosis)
    This creates the target column when it doesn't exist in the raw data.
    """
    df = df.copy()
    has_glucose = "fasting_glucose" in df.columns
    has_hba1c   = "hba1c" in df.columns

    if has_glucose and has_hba1c:
        df["diabetes"] = ((df["fasting_glucose"] >= 126) | (df["hba1c"] >= 6.5)).astype(int)
    elif has_glucose:
        df["diabetes"] = (df["fasting_glucose"] >= 126).astype(int)
    elif has_hba1c:
        df["diabetes"] = (df["hba1c"] >= 6.5).astype(int)
    else:
        raise ValueError("Cannot create diabetes label: no glucose or HbA1c column found.")

    pos_rate = df["diabetes"].mean()
    print(f"[NHANES] Derived diabetes label: positive rate = {pos_rate:.3f}")
    return df


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE SUMMARY
# ─────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Return feature importances as a sorted DataFrame.
    Works for both XGBoost (gain-based) and RandomForest (impurity-based).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise AttributeError("Model does not expose feature_importances_")

    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    return df


# ─────────────────────────────────────────────
# 6. PIMA VALIDATION PIPELINE
# ─────────────────────────────────────────────

def validate_on_pima(model, X_pima: pd.DataFrame, y_pima: pd.Series) -> dict:
    """
    Quick cross-dataset validation of M4 on Pima Indians dataset.
    The Pima features must match the training features (or a subset).
    Returns a dict of accuracy, f1, roc_auc.
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Only use features present in both datasets
    shared_features = [f for f in M4["pima_features"] if f in X_pima.columns]
    if not shared_features:
        print("[Pima Validation] No shared features with M4 training set — skipping.")
        return {}

    X_pima_shared = X_pima[shared_features]
    y_prob = model.predict_proba(X_pima_shared)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    results = {
        "accuracy": accuracy_score(y_pima, y_pred),
        "f1":       f1_score(y_pima, y_pred),
        "roc_auc":  roc_auc_score(y_pima, y_prob),
    }
    print(f"[Pima Validation] Acc: {results['accuracy']:.4f} | "
          f"F1: {results['f1']:.4f} | AUC: {results['roc_auc']:.4f}")
    return results
