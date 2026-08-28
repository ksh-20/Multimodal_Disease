"""
m5_fusion.py
============
Module 5: Late-Fusion Composite Risk Scoring with Per-Modality XAI
Input   : 4 risk scores from M1–M4 (each in [0, 1])
Model   : Shallow 2-layer MLP (Fusion MLP)
XAI     : SHAP KernelExplainer over the fusion layer

Fusion formulation (paper Section 3.5):
  Let r = [r_M1, r_M2, r_M3, r_M4] be the per-modality risk vector.

  Weighted late-fusion (baseline):
    r_final = SUM_{i=1}^{4} w_i * r_Mi,  SUM w_i = 1             (Eq. M5.1)

  Learned fusion MLP (proposed):
    h_1     = ReLU(W_1 * r + b_1)                                  (Eq. M5.2)
    h_2     = ReLU(W_2 * h_1 + b_2)                                (Eq. M5.3)
    r_final = sigmoid(W_3 * h_2 + b_3)   in [0, 1]                (Eq. M5.4)

  Risk category:
    category = {LOW      if r_final < 0.33,
                MODERATE if 0.33 <= r_final < 0.66,
                HIGH     if r_final >= 0.66}                        (Eq. M5.5)

  SHAP attribution over r: phi_i = SHAP value of modality i's score r_Mi
    -> phi_i quantifies how much modality i pushed the final score up/down

Paper Algorithm 2 implementation is in the run_fusion_pipeline() function below.
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import M5, SEED


# ─────────────────────────────────────────────
# 1. FUSION MLP
# ─────────────────────────────────────────────

class FusionMLP(nn.Module):
    """
    Shallow MLP for late fusion of per-modality risk scores.

    Input  : (batch, 4) — [r_M1, r_M2, r_M3, r_M4]
    Output : (batch, 1) — sigmoid-activated composite risk score

    Architecture:
      Linear(4, 16) → ReLU → Dropout
      Linear(16, 8) → ReLU → Dropout
      Linear(8, 1)  → Sigmoid
    """

    def __init__(self,
                 input_dim:   int = M5["input_dim"],
                 hidden_dims: list = M5["hidden_dims"],
                 dropout:     float = M5["dropout"]):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit (pre-sigmoid). Use sigmoid(output) for probability."""
        return self.net(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns composite risk probability in [0, 1]."""
        return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────
# 2. WEIGHTED BASELINE (Eq. M5.1)
# ─────────────────────────────────────────────

class WeightedFusion:
    """
    Simple weighted average of per-modality scores (Eq. M5.1).
    Used as an interpretable baseline to compare against the learned MLP.
    """

    def __init__(self, weights: np.ndarray = None):
        """
        weights : (4,) array summing to 1.
                  Default = uniform (0.25 each).
        """
        if weights is None:
            weights = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights must sum to 1"
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, 4) -> (N,) composite score."""
        return X @ self.weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Optional: solve for weights by OLS (for paper comparison)."""
        from numpy.linalg import lstsq
        self.weights, _, _, _ = lstsq(X, y, rcond=None)
        self.weights = np.clip(self.weights, 0, None)
        self.weights /= self.weights.sum()
        return self


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

class FusionDataset(torch.utils.data.Dataset):
    """Simple dataset wrapping (risk_scores, labels) numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_fusion_model(X_train: np.ndarray, y_train: np.ndarray,
                        X_val:   np.ndarray, y_val:   np.ndarray,
                        device: torch.device,
                        epochs:       int   = M5["epochs"],
                        lr:           float = M5["lr"],
                        weight_decay: float = M5["weight_decay"],
                        batch_size:   int   = M5["batch_size"],
                        save_path:    str   = M5["model_path"]) -> Tuple[FusionMLP, dict]:
    """
    Train the Fusion MLP (Eqs. M5.2–M5.4).

    Parameters
    ----------
    X_train/X_val : (N, 4) — stacked per-modality risk scores
    y_train/y_val : (N,)   — binary diabetes labels (1=diabetic)

    Returns
    -------
    model, history
    """
    torch.manual_seed(SEED)

    train_ds = FusionDataset(X_train, y_train)
    val_ds   = FusionDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    model     = FusionMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {k: [] for k in ["train_losses", "val_losses", "train_accs", "val_accs"]}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ─── Train ───
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * X_batch.size(0)
            preds    = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y_batch.long()).sum().item()
            total   += X_batch.size(0)
        scheduler.step()

        # ─── Val ───
        model.eval()
        vl_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                vl_loss += criterion(logits, y_batch).item() * X_batch.size(0)
                preds    = (torch.sigmoid(logits) >= 0.5).long()
                v_correct += (preds == y_batch.long()).sum().item()
                v_total   += X_batch.size(0)

        tr_acc = correct / total
        vl_acc = v_correct / v_total
        history["train_losses"].append(tr_loss / total)
        history["val_losses"].append(vl_loss / v_total)
        history["train_accs"].append(tr_acc)
        history["val_accs"].append(vl_acc)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"Train Loss: {tr_loss/total:.4f}  Acc: {tr_acc:.4f} | "
                  f"Val Loss: {vl_loss/v_total:.4f}  Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"[M5] Best Val Acc: {best_val_acc:.4f}")
    return model, history


# ─────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_fusion_scores(model: FusionMLP, X: np.ndarray,
                           device: torch.device) -> np.ndarray:
    """
    Run the Fusion MLP on X (N, 4) -> risk probabilities (N,).
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    return model.predict_proba(X_t).cpu().numpy()


def classify_risk(risk_score: float,
                  thresholds: dict = M5["risk_thresholds"]) -> str:
    """Categorize a composite score: 'LOW', 'MODERATE', or 'HIGH'."""
    if risk_score < thresholds["low"]:
        return "LOW"
    elif risk_score < thresholds["moderate"]:
        return "MODERATE"
    else:
        return "HIGH"


def load_trained_model(model_path: str = M5["model_path"],
                        device: torch.device = None) -> FusionMLP:
    """Load saved Fusion MLP."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionMLP()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


# ─────────────────────────────────────────────
# 5. FULL PIPELINE (Algorithm 2 implementation)
# ─────────────────────────────────────────────

def run_fusion_pipeline(m1_scores: np.ndarray,
                         m2_scores: np.ndarray,
                         m3_scores: np.ndarray,
                         m4_scores: np.ndarray,
                         y_true:    np.ndarray,
                         device:    torch.device,
                         train_indices: np.ndarray = None) -> dict:
    """
    Algorithm 2: Late-Fusion Composite Risk Scoring
    ================================================
    Input : per-modality risk scores r_M1..r_M4 for N samples
            y_true : ground truth binary labels
            train_indices : indices of training samples (rest = test)
    Output: fusion_probs, y_pred, metrics

    Paper pseudocode:
      INPUT: r_M1, r_M2, r_M3, r_M4, y_true
      1: Stack scores: X <- [r_M1 | r_M2 | r_M3 | r_M4] (N x 4)
      2: Split X, y into train / val / test
      3: Train FusionMLP on (X_train, y_train) with BCE loss
      4: Load best checkpoint
      5: FOR each sample x_i in X_test:
           5a: r_final <- sigmoid(MLP(x_i))
           5b: category <- classify_risk(r_final)
           5c: phi <- SHAP(MLP, x_i)  // per-modality contribution
      6: Return predictions, risk scores, SHAP explanations
    """
    from sklearn.model_selection import train_test_split

    # 1. Stack modality scores
    X = np.stack([m1_scores, m2_scores, m3_scores, m4_scores], axis=1)

    # 2. Split
    if train_indices is not None:
        test_mask  = np.ones(len(y_true), dtype=bool)
        test_mask[train_indices] = False
        X_train_full = X[train_indices]
        y_train_full = y_true[train_indices]
        X_test = X[test_mask]
        y_test = y_true[test_mask]
    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y_true, test_size=0.20, stratify=y_true, random_state=SEED
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15,
        stratify=y_train_full, random_state=SEED
    )

    # 3-4. Train and load best MLP
    model, history = train_fusion_model(X_train, y_train, X_val, y_val, device)

    # 5. Inference on test set
    fusion_probs = extract_fusion_scores(model, X_test, device)
    y_pred       = (fusion_probs >= 0.5).astype(int)
    categories   = [classify_risk(p) for p in fusion_probs]

    return {
        "model":         model,
        "history":       history,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_test":        y_test,
        "fusion_probs":  fusion_probs,
        "y_pred":        y_pred,
        "categories":    categories,
    }
