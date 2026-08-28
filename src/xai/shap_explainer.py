"""
shap_explainer.py
=================
SHAP wrappers for tabular (M4) and fusion (M5) modules.

Based on: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions,"
NeurIPS 2017. https://arxiv.org/abs/1705.07874

SHAP value definition (Shapley formula):
  phi_i(f, x) = SUM_{S ⊆ F\{i}} [ |S|!(|F|-|S|-1)! / |F|! ] * [f(S∪{i}) - f(S)]

where:
  - phi_i   : SHAP value (contribution) of feature i
  - F       : full set of features
  - S       : subset of features not containing i
  - f(S)    : model prediction with only features in S (others marginalized)

For tree models (XGBoost, RF): uses TreeExplainer — exact, O(TLD^2) computation.
For neural net fusion (MLP): uses DeepExplainer or GradientExplainer.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# 1. TABULAR SHAP (M4 — XGBoost / Random Forest)
# ─────────────────────────────────────────────

class TabularSHAPExplainer:
    """
    SHAP explainer for tree-based classifiers (XGBoost, RandomForest).
    Uses shap.TreeExplainer for exact Shapley values.

    Parameters
    ----------
    model        : fitted sklearn / XGBoost model
    X_background : pd.DataFrame, background dataset for SHAP (typically X_train)
    feature_names: list of str, column names
    """

    def __init__(self, model, X_background: pd.DataFrame, feature_names: list[str]):
        import shap
        self.model         = model
        self.feature_names = feature_names
        self.explainer     = shap.TreeExplainer(model, data=X_background)
        print("[SHAP] TreeExplainer initialized.")

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for X.

        Returns
        -------
        shap_values : np.ndarray, shape (n_samples, n_features)
                      For binary classification, returns values for class 1.
        """
        sv = self.explainer.shap_values(X)
        # For binary classifiers, shap_values is a list [class0, class1]
        if isinstance(sv, list):
            sv = sv[1]
        return sv

    def plot_summary(self, X: pd.DataFrame, shap_values: np.ndarray = None,
                     max_display: int = 15, save_path: str = None):
        """Beeswarm summary plot (requires shap library)."""
        import shap
        import matplotlib.pyplot as plt

        if shap_values is None:
            shap_values = self.compute_shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names,
                          max_display=max_display, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"[SHAP] Summary plot saved to {save_path}")

    def plot_waterfall(self, X_row: pd.DataFrame, shap_values_row: np.ndarray = None,
                       save_path: str = None):
        """Waterfall plot for a single instance."""
        import shap
        import matplotlib.pyplot as plt

        if shap_values_row is None:
            shap_values_row = self.compute_shap_values(X_row)[0]

        explanation = shap.Explanation(
            values        = shap_values_row,
            base_values   = self.explainer.expected_value,
            data          = X_row.values[0],
            feature_names = self.feature_names,
        )
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"[SHAP] Waterfall plot saved to {save_path}")

    def mean_abs_shap(self, X: pd.DataFrame) -> pd.Series:
        """Return mean absolute SHAP values per feature (for bar chart)."""
        sv = self.compute_shap_values(X)
        return pd.Series(np.abs(sv).mean(axis=0), index=self.feature_names).sort_values(ascending=False)


# ─────────────────────────────────────────────
# 2. FUSION SHAP (M5 — MLP over 4 risk scores)
# ─────────────────────────────────────────────

class FusionSHAPExplainer:
    """
    SHAP explainer for the M5 fusion MLP.
    Uses shap.KernelExplainer (model-agnostic) since the MLP is a PyTorch net.

    For the fusion layer, feature names are the modality names:
      ["Retinopathy (M1)", "Acanthosis Nigricans (M2)", "Stress (M3)", "Tabular Risk (M4)"]

    The SHAP values directly quantify each modality's marginal contribution
    to the final composite risk score — this is the key fusion-level XAI result
    reported in Section 4.3.
    """

    def __init__(self, predict_fn, background_data: np.ndarray,
                 modality_names: list[str], n_background: int = 50):
        """
        Parameters
        ----------
        predict_fn      : callable (n, 4) -> (n,), the MLP forward pass as numpy fn
        background_data : np.ndarray (n_background, 4) — per-modality risk scores
        modality_names  : list of 4 strings
        """
        import shap
        self.modality_names = modality_names
        self.predict_fn     = predict_fn

        # KernelExplainer uses a background summary (k-means for efficiency)
        bg = shap.kmeans(background_data, min(n_background, len(background_data)))
        self.explainer = shap.KernelExplainer(predict_fn, bg)
        print("[SHAP-Fusion] KernelExplainer initialized.")

    def compute_shap_values(self, X: np.ndarray,
                             n_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for fusion input X.

        Parameters
        ----------
        X        : np.ndarray (n_samples, 4) — per-modality risk scores
        n_samples: number of coalitions sampled per explanation

        Returns
        -------
        shap_values : np.ndarray (n_samples, 4)
        """
        sv = self.explainer.shap_values(X, nsamples=n_samples, silent=True)
        return sv

    def plot_modality_importance(self, shap_values: np.ndarray,
                                  save_path: str = None):
        """
        Bar chart of mean |SHAP value| per modality — Section 4.3 key figure.
        """
        import matplotlib.pyplot as plt

        mean_abs = np.abs(shap_values).mean(axis=0)
        idx = np.argsort(mean_abs)
        names  = [self.modality_names[i] for i in idx]
        values = mean_abs[idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        ax.barh(names, values, color=[colors[i] for i in idx], alpha=0.85)
        ax.set_xlabel("Mean |SHAP Value| (Modality Contribution)")
        ax.set_title("Fusion-Level Explainability: Per-Modality SHAP Contributions")
        for i, v in enumerate(values):
            ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=10)
        ax.set_xlim(0, max(values) * 1.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"[SHAP-Fusion] Importance plot saved to {save_path}")
        return fig

    def plot_decision(self, X_single: np.ndarray, save_path: str = None):
        """
        Waterfall-style decision plot for a single patient's fusion explanation.
        Shows: base value + each modality's delta -> final risk score.
        """
        import matplotlib.pyplot as plt

        sv_single = self.compute_shap_values(X_single.reshape(1, -1), n_samples=200)[0]
        base_val  = self.explainer.expected_value
        final_val = base_val + sv_single.sum()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(self.modality_names, sv_single,
                color=["#C44E52" if v > 0 else "#4C72B0" for v in sv_single],
                alpha=0.85)
        ax.axvline(0, color="gray", lw=1)
        ax.set_xlabel("SHAP Value (Impact on Risk Score)")
        ax.set_title(f"Individual Fusion Explanation\n"
                     f"Base: {base_val:.3f}  +  SHAP sum: {sv_single.sum():.3f}  "
                     f"= Final: {final_val:.3f}")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
        return fig


# ─────────────────────────────────────────────
# 3. SHAP UTILITIES (no dependency on shap library)
# ─────────────────────────────────────────────

def shapley_manual(model_fn, x: np.ndarray, background_mean: np.ndarray,
                    n_coalitions: int = 256) -> np.ndarray:
    """
    A simplified Monte Carlo Shapley estimator (for educational/fallback use).
    Approximates phi_i by sampling random permutations.

    Parameters
    ----------
    model_fn        : callable (n, p) -> (n,) — must return scalar prediction
    x               : (p,) — the instance to explain
    background_mean : (p,) — baseline (mean of training data)
    n_coalitions    : number of Monte Carlo samples per feature

    Returns
    -------
    phi : (p,) array of approximate Shapley values
    """
    p   = len(x)
    phi = np.zeros(p)

    for _ in range(n_coalitions):
        perm = np.random.permutation(p)
        x_with    = background_mean.copy()
        x_without = background_mean.copy()

        for i, feat_idx in enumerate(perm):
            # Add feature feat_idx to "with" side
            x_with[feat_idx] = x[feat_idx]
            # Marginal contribution
            v_with    = model_fn(x_with.reshape(1, -1))[0]
            v_without = model_fn(x_without.reshape(1, -1))[0]
            phi[feat_idx] += (v_with - v_without)
            x_without[feat_idx] = x[feat_idx]

    return phi / n_coalitions
