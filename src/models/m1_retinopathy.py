"""
m1_retinopathy.py
=================
Module 1: Diabetic Retinopathy Severity Detection
Dataset : APTOS 2019 Blindness Detection (Kaggle)
Model   : ResNet18 fine-tuned for 5-class classification (severity 0–4)
XAI     : Grad-CAM on layer4

Paper equations implemented here:
  - Convolution: (f * g)[n] = SUM_k f[k] * g[n - k]         (Eq. 3)
  - Softmax: sigma(z_i) = exp(z_i) / SUM_j exp(z_j)         (Eq. 4)
  - Cross-entropy loss: L = -SUM_i y_i * log(p_i)            (Eq. 5)
  - Risk score mapping: r_M1 = SUM_j (w_j * p_j), w in {0, 0.25, 0.5, 0.75, 1.0}

Reference: He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import models
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import M1, SEED


# ─────────────────────────────────────────────
# 1. MODEL DEFINITION
# ─────────────────────────────────────────────

class RetinopathyModel(nn.Module):
    """
    ResNet18 with a custom classification head for 5-class DR severity.

    Architecture:
      ResNet18 backbone (ImageNet pretrained)
      -> AdaptiveAvgPool2d (built-in)
      -> Dropout(p)
      -> Linear(512, num_classes)
    """

    def __init__(self, num_classes: int = M1["num_classes"],
                 pretrained: bool = M1["pretrained"],
                 dropout: float = M1["dropout"],
                 freeze_backbone: bool = M1["freeze_backbone"]):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully-connected layer
        in_features = self.backbone.fc.in_features   # 512 for ResNet18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def layer4(self):
        """Expose layer4 for Grad-CAM targeting."""
        return self.backbone.layer4


# ─────────────────────────────────────────────
# 2. RISK SCORE COMPUTATION
# ─────────────────────────────────────────────

def compute_retinopathy_risk(logits: torch.Tensor,
                              severity_weights: list = M1["severity_weights"]) -> torch.Tensor:
    """
    Map class probabilities to a continuous risk score in [0, 1].

    r_M1 = SUM_j (w_j * p_j)
    where w = [0.0, 0.25, 0.50, 0.75, 1.0] for severity classes 0-4.

    Parameters
    ----------
    logits : (batch, 5) — raw model outputs (pre-softmax)

    Returns
    -------
    risk_scores : (batch,) float tensor in [0, 1]
    """
    probs   = torch.softmax(logits, dim=1)
    weights = torch.tensor(severity_weights, dtype=torch.float32, device=logits.device)
    return (probs * weights).sum(dim=1)


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def train_retinopathy_model(train_loader, val_loader,
                             device: torch.device,
                             epochs: int = M1["epochs"],
                             lr: float = M1["lr"],
                             weight_decay: float = M1["weight_decay"],
                             scheduler_type: str = M1["scheduler"],
                             save_path: str = M1["model_path"]) -> Tuple[RetinopathyModel, dict]:
    """
    Full training pipeline for M1.

    Returns
    -------
    model   : best RetinopathyModel (loaded from checkpoint)
    history : dict with train_losses, val_losses, train_accs, val_accs
    """
    torch.manual_seed(SEED)
    model     = RetinopathyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    history = {k: [] for k in ["train_losses", "val_losses", "train_accs", "val_accs"]}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_losses"].append(tr_loss)
        history["val_losses"].append(vl_loss)
        history["train_accs"].append(tr_acc)
        history["val_accs"].append(vl_acc)

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (val_acc={vl_acc:.4f})")

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


# ─────────────────────────────────────────────
# 4. INFERENCE & RISK SCORE EXTRACTION
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_risk_scores(model: RetinopathyModel, loader,
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over a DataLoader and extract:
      - risk_scores  : (N,) continuous risk in [0, 1]
      - predictions  : (N,) predicted severity class
      - ground_truth : (N,) true labels
    """
    model.eval()
    all_risk, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="  Inference"):
        images = images.to(device)
        logits = model(images)
        risks  = compute_retinopathy_risk(logits)

        all_risk.append(risks.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    return (np.concatenate(all_risk),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


def load_trained_model(model_path: str = M1["model_path"],
                        device: torch.device = None) -> RetinopathyModel:
    """Load a saved RetinopathyModel checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinopathyModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────
# 5. BINARY CONVERSION (for fusion input)
# ─────────────────────────────────────────────

def severity_to_binary(severity_preds: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Convert 5-class severity to binary (diabetic / no diabetic) for evaluation.
    DR present: severity >= threshold (default=1: mild or worse).
    """
    return (severity_preds >= threshold).astype(int)
