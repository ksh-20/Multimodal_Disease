"""
m3_stress.py
============
Module 3: Psychological Stress Estimation via Facial Expression Recognition
Dataset : FER2013 (Kaggle: msambare/fer2013)
Model   : Custom lightweight CNN for 7-class emotion recognition (48x48 grayscale)
XAI     : Grad-CAM on last conv block (conv4)

Stress score derivation:
  Emotions: angry(0), disgust(1), fear(2), happy(3), sad(4), surprise(5), neutral(6)
  Stress-relevant emotions: angry, disgust, fear, sad (→ correlate with HPA-axis activation)

  r_M3 = SUM_i (s_i * p_i)    [stress_weight s_i from config.M3.stress_weights]

Physiological basis: Fear → highest weight (0.95), reflects acute cortisol response.
  Angry (0.90), Sad (0.75), Disgust (0.70) follow. Happy/Neutral/Surprise ~ 0.

Reference: Goodfellow et al., "Challenges in Representation Learning: A report on
three machine learning contests," ICANN 2013. (FER2013 dataset paper)

CNN Architecture equations in paper:
  - Conv: (I * K)[i,j] = SUM_m SUM_n I[i+m, j+n] * K[m,n]
  - ReLU: f(x) = max(0, x)
  - Batch Normalization: x_hat = (x - mu) / sqrt(sigma^2 + eps)
  - Global Average Pooling: gap(A) = (1/HW) * SUM_i SUM_j A[i,j]
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import M3, SEED


# ─────────────────────────────────────────────
# 1. CNN ARCHITECTURE
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class StressCNN(nn.Module):
    """
    Lightweight CNN for 48x48 grayscale facial expression recognition.

    Architecture (paper Section 3.3):
      Input: (B, 1, 48, 48)
      conv1: 1 → 32 channels,  3x3, BN, ReLU, Pool → (B, 32, 24, 24)
      conv2: 32 → 64 channels, 3x3, BN, ReLU, Pool → (B, 64, 12, 12)
      conv3: 64 → 128 channels,3x3, BN, ReLU, Pool → (B, 128, 6, 6)
      conv4: 128 → 256 channels,3x3, BN, ReLU       → (B, 256, 6, 6)  [Grad-CAM target]
      GlobalAvgPool                                  → (B, 256)
      Dropout(p) → Linear(256, 7)
    """

    def __init__(self, num_classes: int = M3["num_classes"],
                 dropout: float = M3["dropout"]):
        super().__init__()
        self.conv1 = ConvBlock(1,   32,  pool=True)
        self.conv2 = ConvBlock(32,  64,  pool=True)
        self.conv3 = ConvBlock(64,  128, pool=True)
        self.conv4 = ConvBlock(128, 256, pool=False)   # No pool — Grad-CAM target
        self.gap   = nn.AdaptiveAvgPool2d(1)            # Global Average Pool
        self.drop  = nn.Dropout(p=dropout)
        self.fc    = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)      # (B, 256, 6, 6) — activations for Grad-CAM
        x = self.gap(x)        # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.drop(x)
        return self.fc(x)      # (B, 7)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Return the last conv block for Grad-CAM targeting."""
        return self.conv4.block  # last sequential block before GAP


# ─────────────────────────────────────────────
# 2. STRESS SCORE COMPUTATION
# ─────────────────────────────────────────────

def compute_stress_score(logits: torch.Tensor,
                          stress_weights: dict = M3["stress_weights"]) -> torch.Tensor:
    """
    Derive a continuous stress score r_M3 in [0, 1] from 7-class emotion probabilities.

    r_M3 = SUM_{i=0}^{6} s_i * p_i

    where s_i are per-emotion stress weights (see config.M3.stress_weights)
    and p_i = softmax(logit_i).

    Parameters
    ----------
    logits : (batch, 7)

    Returns
    -------
    stress_scores : (batch,) in [0, 1]
    """
    probs   = torch.softmax(logits, dim=1)
    weights = torch.tensor(
        [stress_weights[i] for i in range(len(stress_weights))],
        dtype=torch.float32, device=logits.device
    )
    return (probs * weights).sum(dim=1)


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

def train_stress_model(train_loader, val_loader,
                        device: torch.device,
                        epochs: int = M3["epochs"],
                        lr: float = M3["lr"],
                        weight_decay: float = M3["weight_decay"],
                        step_size: int = M3["step_size"],
                        gamma: float = M3["gamma"],
                        save_path: str = M3["model_path"]) -> Tuple[StressCNN, dict]:
    """Full training pipeline for M3."""
    torch.manual_seed(SEED)
    model     = StressCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = {k: [] for k in ["train_losses", "val_losses", "train_accs", "val_accs"]}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ─── Train ───
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  Stress Ep{epoch}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss  += loss.item() * imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        scheduler.step()

        # ─── Val ───
        model.eval()
        vl_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                vl_loss += criterion(logits, labels).item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        tr_acc = correct / total
        vl_acc = v_correct / v_total

        history["train_losses"].append(tr_loss / total)
        history["val_losses"].append(vl_loss / v_total)
        history["train_accs"].append(tr_acc)
        history["val_accs"].append(vl_acc)

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {tr_loss/total:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss/v_total:.4f}  Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best saved (val_acc={vl_acc:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


# ─────────────────────────────────────────────
# 4. INFERENCE & RISK SCORES
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_risk_scores(model: StressCNN, loader,
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      stress_scores : (N,) in [0, 1] — used as r_M3 for fusion
      emotion_preds : (N,) predicted emotion class index
      ground_truth  : (N,) true emotion labels (NOT same as diabetes label)
    """
    model.eval()
    all_stress, all_preds, all_labels = [], [], []

    for imgs, labels in tqdm(loader, desc="  Stress Inference"):
        imgs   = imgs.to(device)
        logits = model(imgs)
        stress = compute_stress_score(logits)
        all_stress.append(stress.cpu().numpy())
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(labels.numpy())

    return (np.concatenate(all_stress),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


def load_trained_model(model_path: str = M3["model_path"],
                        device: torch.device = None) -> StressCNN:
    """Load saved StressCNN."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StressCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model
