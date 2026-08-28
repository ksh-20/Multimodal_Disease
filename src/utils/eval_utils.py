"""
eval_utils.py
=============
Metric computation, classification reports, ROC/AUC, and comparison tables
for all 5 modules of the Multimodal Disease Risk Framework.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)


# ─────────────────────────────────────────────
# 1. CORE METRIC COMPUTATION
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None, average="weighted",
                    module_name: str = "Module") -> dict:
    """
    Compute classification metrics for a single module.

    Parameters
    ----------
    y_true  : array-like, ground truth labels
    y_pred  : array-like, predicted labels
    y_prob  : array-like, predicted probabilities (for ROC-AUC); optional
    average : str, sklearn averaging strategy
    module_name : str, display label

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc (if y_prob given)
    """
    metrics = {
        "module":    module_name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            # Binary
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc"] = roc_auc_score(y_true, prob)
            else:
                # Multiclass
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
        except ValueError:
            metrics["roc_auc"] = np.nan

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>12}: {v:.4f}")
    return metrics


def compute_binary_roc(y_true, y_prob) -> tuple:
    """Return (fpr, tpr, thresholds, auc) for ROC curve plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return fpr, tpr, thresholds, auc


def print_classification_report(y_true, y_pred, class_names=None):
    """Pretty-print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


# ─────────────────────────────────────────────
# 2. AGGREGATE RESULTS TABLE
# ─────────────────────────────────────────────

def build_results_table(all_metrics: list[dict]) -> pd.DataFrame:
    """
    Build the Section 4.3 comparison table from a list of metric dicts.

    Parameters
    ----------
    all_metrics : list of dicts returned by compute_metrics()

    Returns
    -------
    pd.DataFrame with one row per module, formatted for paper Table
    """
    df = pd.DataFrame(all_metrics)
    numeric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    df.set_index("module", inplace=True)
    return df


def save_results_csv(all_metrics: list[dict], output_path: str):
    """Save results table to CSV for reproducibility."""
    df = pd.DataFrame(all_metrics)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Eval] Results saved to {output_path}")


# ─────────────────────────────────────────────
# 3. ABLATION STUDY UTILITIES
# ─────────────────────────────────────────────

def run_ablation_study(fusion_model_fn, X_all: np.ndarray, y_true: np.ndarray,
                       modality_names: list[str]) -> pd.DataFrame:
    """
    Drop-one-modality ablation for M5 fusion.

    Parameters
    ----------
    fusion_model_fn : callable, takes X (n_samples, n_modalities) -> y_prob
    X_all           : np.ndarray, shape (n_samples, n_modalities) — per-modality scores
    y_true          : np.ndarray, ground truth binary labels
    modality_names  : list of str, names for each column of X_all

    Returns
    -------
    pd.DataFrame with columns [dropped_modality, accuracy, f1, roc_auc]
    """
    n_modalities = X_all.shape[1]
    results = []

    # Full model baseline
    y_prob = fusion_model_fn(X_all)
    y_pred = (y_prob >= 0.5).astype(int)
    m = compute_metrics(y_true, y_pred, y_prob=y_prob, module_name="All modalities")
    results.append({
        "dropped_modality": "None (full model)",
        "accuracy": m["accuracy"], "f1": m["f1"],
        "roc_auc": m.get("roc_auc", np.nan)
    })

    # Drop each modality
    for i, name in enumerate(modality_names):
        X_masked = X_all.copy()
        X_masked[:, i] = 0.0  # zero out modality i

        y_prob_i = fusion_model_fn(X_masked)
        y_pred_i = (y_prob_i >= 0.5).astype(int)
        m_i = compute_metrics(y_true, y_pred_i, y_prob=y_prob_i,
                               module_name=f"Drop {name}")
        results.append({
            "dropped_modality": name,
            "accuracy": m_i["accuracy"],
            "f1": m_i["f1"],
            "roc_auc": m_i.get("roc_auc", np.nan)
        })
        print(f"  Drop [{name}] -> Acc: {m_i['accuracy']:.4f} | F1: {m_i['f1']:.4f}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING SUMMARY TABLE
# ─────────────────────────────────────────────

def build_hyperparam_table() -> pd.DataFrame:
    """
    Returns a DataFrame documenting the hyperparameters explored vs. chosen.
    Used in Section 4.6 of the paper.
    """
    rows = [
        # M1 Retinopathy
        {"Module": "M1 (Retinopathy)", "Parameter": "Learning Rate",
         "Values Tried": "1e-3, 1e-4, 5e-5", "Chosen": "1e-4"},
        {"Module": "M1", "Parameter": "Batch Size",
         "Values Tried": "16, 32, 64", "Chosen": "32"},
        {"Module": "M1", "Parameter": "Epochs",
         "Values Tried": "10, 20, 30", "Chosen": "20"},
        {"Module": "M1", "Parameter": "Dropout",
         "Values Tried": "0.2, 0.4, 0.5", "Chosen": "0.4"},
        # M2 Acanthosis
        {"Module": "M2 (Acanthosis Proxy)", "Parameter": "Learning Rate",
         "Values Tried": "1e-4, 5e-5", "Chosen": "5e-5"},
        {"Module": "M2", "Parameter": "Freeze Backbone",
         "Values Tried": "True, False", "Chosen": "True"},
        # M3 Stress
        {"Module": "M3 (Stress)", "Parameter": "Learning Rate",
         "Values Tried": "1e-2, 1e-3, 5e-4", "Chosen": "1e-3"},
        {"Module": "M3", "Parameter": "Epochs",
         "Values Tried": "20, 30, 50", "Chosen": "30"},
        {"Module": "M3", "Parameter": "LR Step Gamma",
         "Values Tried": "0.1, 0.5, 0.7", "Chosen": "0.5"},
        # M4 Tabular
        {"Module": "M4 (Tabular XGB)", "Parameter": "Max Depth",
         "Values Tried": "4, 6, 8", "Chosen": "6"},
        {"Module": "M4", "Parameter": "Learning Rate",
         "Values Tried": "0.01, 0.05, 0.1", "Chosen": "0.05"},
        {"Module": "M4", "Parameter": "n_estimators",
         "Values Tried": "100, 200, 300", "Chosen": "300"},
        # M5 Fusion
        {"Module": "M5 (Fusion MLP)", "Parameter": "Hidden Dims",
         "Values Tried": "[32], [16,8], [64,32]", "Chosen": "[16, 8]"},
        {"Module": "M5", "Parameter": "Learning Rate",
         "Values Tried": "1e-3, 5e-4, 1e-4", "Chosen": "5e-4"},
        {"Module": "M5", "Parameter": "Epochs",
         "Values Tried": "30, 50, 100", "Chosen": "50"},
    ]
    return pd.DataFrame(rows)
