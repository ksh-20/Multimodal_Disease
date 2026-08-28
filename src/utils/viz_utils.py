"""
viz_utils.py
============
All plotting helpers for the Multimodal Disease Risk Framework.
Generates every figure required for Section 4 of the paper.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import VIZ, FIGURES_DIR

matplotlib.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi":    VIZ["dpi"],
})


def _save_fig(fig, name: str, subdir: str = ""):
    """Save figure to FIGURES_DIR / subdir / name.png and close."""
    out_dir = Path(FIGURES_DIR) / subdir if subdir else Path(FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{VIZ['save_format']}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved: {path}")
    return str(path)


# ─────────────────────────────────────────────
# 1. TRAINING CURVES
# ─────────────────────────────────────────────

def plot_training_curves(train_losses: list, val_losses: list,
                          train_accs: list, val_accs: list,
                          module_name: str, save: bool = True) -> plt.Figure:
    """
    Plot loss and accuracy curves over training epochs.
    Used in Section 4.1 / 4.3.
    """
    with plt.style.context(VIZ["style"]):
        fig, axes = plt.subplots(1, 2, figsize=VIZ["figsize_wide"])

        epochs = range(1, len(train_losses) + 1)

        # Loss
        axes[0].plot(epochs, train_losses, label="Train", linewidth=2, color="#4C72B0")
        axes[0].plot(epochs, val_losses,   label="Val",   linewidth=2,
                     color="#DD8452", linestyle="--")
        axes[0].set_title(f"{module_name} — Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Accuracy
        axes[1].plot(epochs, train_accs, label="Train", linewidth=2, color="#4C72B0")
        axes[1].plot(epochs, val_accs,   label="Val",   linewidth=2,
                     color="#DD8452", linestyle="--")
        axes[1].set_title(f"{module_name} — Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        fig.tight_layout()

    if save:
        _save_fig(fig, f"{module_name.lower().replace(' ', '_')}_training_curves")
    return fig


# ─────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names: list,
                           module_name: str, save: bool = True) -> plt.Figure:
    """Plot and optionally save a confusion matrix heatmap."""
    with plt.style.context(VIZ["style"]):
        fig, ax = plt.subplots(figsize=VIZ["figsize_square"])
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=class_names,
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"{module_name} — Confusion Matrix")
        fig.tight_layout()

    if save:
        _save_fig(fig, f"{module_name.lower().replace(' ', '_')}_confusion_matrix")
    return fig


# ─────────────────────────────────────────────
# 3. ROC CURVE
# ─────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, module_name: str,
                   save: bool = True) -> plt.Figure:
    """Plot a single ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    with plt.style.context(VIZ["style"]):
        fig, ax = plt.subplots(figsize=VIZ["figsize_square"])
        ax.plot(fpr, tpr, color="#4C72B0", lw=2,
                label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{module_name} — ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()

    if save:
        _save_fig(fig, f"{module_name.lower().replace(' ', '_')}_roc_curve")
    return fig


def plot_multi_roc(roc_data: list[dict], save: bool = True) -> plt.Figure:
    """
    Plot multiple ROC curves on one axes for comparison.

    Parameters
    ----------
    roc_data : list of dicts with keys: name, y_true, y_prob
    """
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    with plt.style.context(VIZ["style"]):
        fig, ax = plt.subplots(figsize=VIZ["figsize_square"])
        for i, rd in enumerate(roc_data):
            fpr, tpr, _ = roc_curve(rd["y_true"], rd["y_prob"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                    label=f"{rd['name']} (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Modules vs. Fusion")
        ax.legend(loc="lower right")
        fig.tight_layout()

    if save:
        _save_fig(fig, "all_modules_roc_comparison")
    return fig


# ─────────────────────────────────────────────
# 4. PER-MODULE METRICS BAR CHART
# ─────────────────────────────────────────────

def plot_module_metrics_bar(all_metrics: list[dict], save: bool = True) -> plt.Figure:
    """
    Grouped bar chart of Accuracy / Precision / Recall / F1 per module.
    Section 4.3 headline figure.
    """
    df = pd.DataFrame(all_metrics).set_index("module")
    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in df.columns]
    df_plot = df[metric_cols].astype(float)

    with plt.style.context(VIZ["style"]):
        fig, ax = plt.subplots(figsize=VIZ["figsize_wide"])
        x = np.arange(len(df_plot))
        width = 0.18
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for i, col in enumerate(metric_cols):
            bars = ax.bar(x + i * width, df_plot[col], width,
                          label=col.capitalize(), color=colors[i], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_plot.index, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Per-Module Classification Metrics")
        ax.legend()
        fig.tight_layout()

    if save:
        _save_fig(fig, "per_module_metrics_bar")
    return fig


# ─────────────────────────────────────────────
# 5. FUSION vs. SINGLE-MODALITY BAR CHART (HEADLINE)
# ─────────────────────────────────────────────

def plot_fusion_vs_single(all_metrics: list[dict], fusion_metric: dict,
                           metric: str = "f1", save: bool = True) -> plt.Figure:
    """
    Bar chart: each module's F1 vs. fusion F1 — the paper's headline result.
    """
    names  = [m["module"] for m in all_metrics] + [fusion_metric["module"]]
    values = [float(m.get(metric, 0)) for m in all_metrics] + [float(fusion_metric.get(metric, 0))]
    colors = ["#4C72B0"] * len(all_metrics) + ["#C44E52"]

    with plt.style.context(VIZ["style"]):
        fig, ax = plt.subplots(figsize=VIZ["figsize_wide"])
        bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Fusion vs. Single-Modality Baselines ({metric.upper()})")
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.axhline(y=float(fusion_metric.get(metric, 0)), color="#C44E52",
                   linestyle="--", alpha=0.6, label="Fusion level")
        ax.legend()
        fig.tight_layout()

    if save:
        _save_fig(fig, "fusion_vs_single_modality")
    return fig


# ─────────────────────────────────────────────
# 6. ABLATION STUDY PLOT
# ─────────────────────────────────────────────

def plot_ablation(ablation_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Horizontal bar chart showing F1 drop when each modality is removed.
    Section 4.5.
    """
    df = ablation_df.copy()
    baseline_f1 = float(df[df["dropped_modality"] == "None (full model)"]["f1"].values[0])
    df_ablated = df[df["dropped_modality"] != "None (full model)"].copy()
    df_ablated["f1_drop"] = baseline_f1 - df_ablated["f1"].astype(float)
    df_ablated = df_ablated.sort_values("f1_drop", ascending=True)

    with plt.style.context(VIZ["style"]):
        fig, ax = plt.subplots(figsize=VIZ["figsize_default"])
        colors = ["#C44E52" if v > 0.02 else "#4C72B0" for v in df_ablated["f1_drop"]]
        ax.barh(df_ablated["dropped_modality"], df_ablated["f1_drop"],
                color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel("F1 Score Drop (Full Model - Ablated)")
        ax.set_title("Ablation Study — Modality Contribution")
        ax.axvline(0, color="gray", lw=1)
        for i, (_, row) in enumerate(df_ablated.iterrows()):
            ax.text(row["f1_drop"] + 0.001, i,
                    f"{row['f1_drop']:.4f}", va="center", fontsize=10)
        fig.tight_layout()

    if save:
        _save_fig(fig, "ablation_study")
    return fig


# ─────────────────────────────────────────────
# 7. SHAP FEATURE IMPORTANCE (M4 / M5)
# ─────────────────────────────────────────────

def plot_shap_bar(shap_values: np.ndarray, feature_names: list,
                  title: str, save: bool = True, filename: str = "shap_bar") -> plt.Figure:
    """
    Horizontal bar chart of mean absolute SHAP values (fallback if shap not installed).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)
    sorted_names  = [feature_names[i] for i in idx]
    sorted_values = mean_abs[idx]

    with plt.style.context(VIZ["style"]):
        fig, ax = plt.subplots(figsize=VIZ["figsize_default"])
        ax.barh(sorted_names, sorted_values, color="#4C72B0", alpha=0.85)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(title)
        fig.tight_layout()

    if save:
        _save_fig(fig, filename)
    return fig


# ─────────────────────────────────────────────
# 8. GRAD-CAM OVERLAY
# ─────────────────────────────────────────────

def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray,
                     title: str = "Grad-CAM", alpha: float = 0.4,
                     save: bool = True, filename: str = "gradcam") -> plt.Figure:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img : np.ndarray, shape (H, W, 3), values in [0, 255]
    heatmap      : np.ndarray, shape (H, W), values in [0, 1]
    """
    import cv2

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original
    if heatmap_color.shape[:2] != original_img.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (original_img.shape[1], original_img.shape[0]))

    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * original_img)

    with plt.style.context(VIZ["style"]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_img);    axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
        axes[2].imshow(overlay);         axes[2].set_title("Overlay"); axes[2].axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

    if save:
        _save_fig(fig, filename)
    return fig


# ─────────────────────────────────────────────
# 9. COMPOSITE RISK GAUGE CHART
# ─────────────────────────────────────────────

def plot_risk_gauge(risk_score: float, module_scores: dict,
                    save: bool = True, filename: str = "risk_gauge") -> plt.Figure:
    """
    Display the composite risk score as a gauge + per-modality contributions.
    Used in Streamlit demo and paper illustration.
    """
    # Color per risk level
    if risk_score < 0.33:
        color = "#55A868"
        level = "LOW"
    elif risk_score < 0.66:
        color = "#DD8452"
        level = "MODERATE"
    else:
        color = "#C44E52"
        level = "HIGH"

    with plt.style.context(VIZ["style"]):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                  gridspec_kw={"width_ratios": [1, 1.5]})

        # Left: Gauge (semi-circle)
        ax_gauge = axes[0]
        theta = np.linspace(np.pi, 0, 300)
        ax_gauge.fill_between(np.cos(theta[:100]), np.sin(theta[:100]), alpha=0.2, color="#55A868")
        ax_gauge.fill_between(np.cos(theta[100:200]), np.sin(theta[100:200]), alpha=0.2, color="#DD8452")
        ax_gauge.fill_between(np.cos(theta[200:]), np.sin(theta[200:]), alpha=0.2, color="#C44E52")

        needle_angle = np.pi - risk_score * np.pi
        ax_gauge.annotate("", xy=(0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle)),
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
        ax_gauge.set_xlim(-1.1, 1.1)
        ax_gauge.set_ylim(-0.2, 1.1)
        ax_gauge.axis("off")
        ax_gauge.text(0, -0.1, f"Risk Score: {risk_score:.3f}", ha="center",
                       fontsize=14, fontweight="bold", color=color)
        ax_gauge.set_title(f"Composite Risk: {level}", color=color, fontweight="bold")

        # Right: modality breakdown
        ax_bar = axes[1]
        names  = list(module_scores.keys())
        values = list(module_scores.values())
        bar_colors = ["#C44E52" if v > 0.66 else "#DD8452" if v > 0.33 else "#55A868"
                      for v in values]
        bars = ax_bar.barh(names, values, color=bar_colors, alpha=0.85)
        ax_bar.set_xlim(0, 1.1)
        ax_bar.set_xlabel("Modality Risk Score")
        ax_bar.set_title("Per-Modality Contributions")
        for bar, val in zip(bars, values):
            ax_bar.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=10)
        ax_bar.axvline(0.33, color="gray", lw=1, linestyle=":")
        ax_bar.axvline(0.66, color="gray", lw=1, linestyle=":")

        fig.suptitle("Explainable Multimodal Metabolic Risk Assessment",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

    if save:
        _save_fig(fig, filename)
    return fig
