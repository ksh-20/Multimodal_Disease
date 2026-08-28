"""
m2_acanthosis.py
================
Module 2: Acanthosis Nigricans (AN) Detection — Skin-based Insulin Resistance Marker
Dataset : HAM10000 (proxy: 'akiec' class as AN-like lesion)
Model   : ResNet18 — two-stage:
            Stage 1) Pretrain on HAM10000 7-class classification
            Stage 2) Fine-tune as binary (AN-proxy / normal) with frozen backbone
XAI     : Grad-CAM on layer4

Paper discussion (Limitation section):
  "No standardized public benchmark exists for acanthosis nigricans classification.
   We employ HAM10000 (Tschandl et al., 2018) with the actinic keratosis (akiec)
   class as a morphological proxy, given its shared keratinocyte-proliferation
   pathophysiology with AN. A dedicated AN dataset is proposed as a future
   dataset contribution of this work."

Reference: Tschandl et al., "The HAM10000 dataset, a large collection of
multi-source dermatoscopic images of common pigmented skin lesions,"
Scientific Data, 2018.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import M2, SEED


# ─────────────────────────────────────────────
# 1. STAGE 1 — HAM10000 PRETRAIN (7-class)
# ─────────────────────────────────────────────

class HAMPretainModel(nn.Module):
    """
    ResNet18 for HAM10000 7-class skin lesion classification.
    Used as Stage 1 pretraining to learn general dermatology features
    before binary AN fine-tuning.
    """

    def __init__(self, num_classes: int = M2["ham_classes"],
                 pretrained: bool = M2["pretrained"],
                 dropout: float = M2["dropout"]):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features   = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    @property
    def layer4(self):
        return self.backbone.layer4


# ─────────────────────────────────────────────
# 2. STAGE 2 — BINARY AN CLASSIFIER (fine-tune)
# ─────────────────────────────────────────────

class AcanthosisModel(nn.Module):
    """
    Binary classifier for AN-proxy detection.
    Loads pretrained HAM weights for the backbone, freezes early layers,
    replaces head with a binary classifier.

    Output: sigmoid probability of AN-like lesion presence
    Risk score: p(class=1) directly used as r_M2 in fusion
    """

    def __init__(self, ham_weights_path: str = None,
                 freeze_backbone: bool = M2["freeze_backbone"],
                 dropout: float = M2["dropout"]):
        super().__init__()

        # Start from ImageNet weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Load HAM pretrained feature extractor if available
        if ham_weights_path and Path(ham_weights_path).exists():
            ham_model = HAMPretainModel()
            ham_model.load_state_dict(
                torch.load(ham_weights_path, map_location="cpu")
            )
            # Transfer everything except the final FC layer
            self.backbone.load_state_dict(
                {k: v for k, v in ham_model.backbone.state_dict().items()
                 if not k.startswith("fc")},
                strict=False
            )
            print(f"[M2] Loaded HAM pretrained weights from {ham_weights_path}")

        # Freeze backbone (conv layers) — only train the new head
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc") and not name.startswith("layer4"):
                    param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),   # Binary logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit (scalar per sample)."""
        return self.backbone(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probability of AN-positive class."""
        return torch.sigmoid(self.forward(x))

    @property
    def layer4(self):
        return self.backbone.layer4


# ─────────────────────────────────────────────
# 3. TRAINING — STAGE 1 (HAM pretraining)
# ─────────────────────────────────────────────

def train_ham_pretrain(train_loader, val_loader,
                        device: torch.device,
                        epochs: int = 15,
                        lr: float = 1e-4,
                        save_path: str = M2["ham_model_path"]) -> HAMPretainModel:
    """Train Stage 1: 7-class HAM10000 classifier."""
    torch.manual_seed(SEED)
    model     = HAMPretainModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for imgs, labels in tqdm(train_loader, desc=f"  HAM Ep{epoch}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct  += (model(imgs).detach().argmax(1) == labels).sum().item()
            total    += imgs.size(0)
        scheduler.step()

        # Val
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)

        val_acc = v_correct / v_total
        print(f"[HAM Stage1] Epoch {epoch}/{epochs} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# ─────────────────────────────────────────────
# 4. TRAINING — STAGE 2 (Binary AN fine-tune)
# ─────────────────────────────────────────────

def train_acanthosis_model(train_loader, val_loader,
                            device: torch.device,
                            epochs: int = M2["epochs"],
                            lr: float = M2["lr"],
                            weight_decay: float = M2["weight_decay"],
                            ham_weights_path: str = M2["ham_model_path"],
                            save_path: str = M2["model_path"]) -> Tuple[AcanthosisModel, dict]:
    """Train Stage 2: Binary AN-proxy classifier."""
    torch.manual_seed(SEED)
    model     = AcanthosisModel(ham_weights_path=ham_weights_path).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {k: [] for k in ["train_losses", "val_losses", "train_accs", "val_accs"]}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  AN Ep{epoch}", leave=False):
            imgs   = imgs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss  += loss.item() * imgs.size(0)
            preds     = (torch.sigmoid(logits) >= 0.5).long()
            correct  += (preds == labels.long()).sum().item()
            total    += imgs.size(0)
        scheduler.step()

        tr_acc = correct / total

        # Val
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().to(device)
                logits = model(imgs)
                v_loss += criterion(logits, labels).item() * imgs.size(0)
                preds   = (torch.sigmoid(logits) >= 0.5).long()
                v_correct += (preds == labels.long()).sum().item()
                v_total   += imgs.size(0)

        vl_acc  = v_correct / v_total
        vl_loss = v_loss / v_total

        history["train_losses"].append(tr_loss / total)
        history["val_losses"].append(vl_loss)
        history["train_accs"].append(tr_acc)
        history["val_accs"].append(vl_acc)

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {tr_loss/total:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


# ─────────────────────────────────────────────
# 5. INFERENCE & RISK SCORES
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_risk_scores(model: AcanthosisModel, loader,
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (risk_scores, binary_preds, ground_truth)."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels in tqdm(loader, desc="  AN Inference"):
        imgs  = imgs.to(device)
        probs = model.predict_proba(imgs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


def load_trained_model(model_path: str = M2["model_path"],
                        device: torch.device = None) -> AcanthosisModel:
    """Load saved AcanthosisModel."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AcanthosisModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model
