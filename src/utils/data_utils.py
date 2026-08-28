"""
data_utils.py
=============
Dataset loaders, preprocessors, and augmentation pipelines for all 5 modules.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Allow running from notebook root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    APTOS_DIR, HAM_DIR, FER_DIR, NHANES_DIR, PIMA_DIR,
    M1, M2, M3, M4, SEED
)

# ─────────────────────────────────────────────
# 1. SHARED HELPERS
# ─────────────────────────────────────────────

def set_seed(seed: int = SEED):
    """Fix all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 2. MODULE 1 — APTOS 2019 RETINOPATHY DATASET
# ─────────────────────────────────────────────

class APTOSDataset(Dataset):
    """
    APTOS 2019 Diabetic Retinopathy Detection dataset.
    Expects:
      <aptos_dir>/train_images/   (fundus .png files)
      <aptos_dir>/train.csv       (id_code, diagnosis)
    """

    TRAIN_TRANSFORMS = T.Compose([
        T.Resize((M1["img_size"], M1["img_size"])),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = T.Compose([
        T.Resize((M1["img_size"], M1["img_size"])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['id_code']}.png"
        image = Image.open(img_path).convert("RGB")
        label = int(row["diagnosis"])
        if self.transform:
            image = self.transform(image)
        return image, label


def load_aptos_dataloaders(aptos_dir=APTOS_DIR, batch_size=M1["batch_size"],
                            val_split=0.15, num_workers=2):
    """
    Returns train_loader, val_loader, test_loader for APTOS 2019.
    Performs a 70/15/15 stratified split on the training CSV.
    """
    csv_path = Path(aptos_dir) / "train.csv"
    img_dir  = Path(aptos_dir) / "train_images"

    df = pd.read_csv(csv_path)

    # Stratified split: 70 train, 15 val, 15 test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["diagnosis"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["diagnosis"], random_state=SEED
    )

    train_ds = APTOSDataset(train_df, img_dir, APTOSDataset.TRAIN_TRANSFORMS)
    val_ds   = APTOSDataset(val_df,   img_dir, APTOSDataset.VAL_TRANSFORMS)
    test_ds  = APTOSDataset(test_df,  img_dir, APTOSDataset.VAL_TRANSFORMS)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"[APTOS] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, test_df


# ─────────────────────────────────────────────
# 3. MODULE 2 — HAM10000 SKIN LESION DATASET
# ─────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    """
    HAM10000 Skin Lesion Analysis dataset.
    Expects:
      <ham_dir>/images/          (ISIC_*.jpg files)
      <ham_dir>/HAM10000_metadata.csv
    """

    TRAIN_TRANSFORMS = T.Compose([
        T.Resize((M2["img_size"], M2["img_size"])),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORMS = T.Compose([
        T.Resize((M2["img_size"], M2["img_size"])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    LABEL_MAP = {
        "akiec": 0, "bcc": 1, "bkl": 2,
        "df": 3,    "mel": 4, "nv": 5, "vasc": 6,
    }

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        label = self.LABEL_MAP[row["dx"]]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_ham_dataloaders(ham_dir=HAM_DIR, batch_size=M2["batch_size"],
                          val_split=0.15, num_workers=2):
    """Returns train/val/test DataLoaders for HAM10000 (M2 pretraining)."""
    csv_path = Path(ham_dir) / "HAM10000_metadata.csv"
    img_dir  = Path(ham_dir) / "images"

    df = pd.read_csv(csv_path).drop_duplicates("image_id")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["dx"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["dx"], random_state=SEED
    )

    train_ds = HAM10000Dataset(train_df, img_dir, HAM10000Dataset.TRAIN_TRANSFORMS)
    val_ds   = HAM10000Dataset(val_df,   img_dir, HAM10000Dataset.VAL_TRANSFORMS)
    test_ds  = HAM10000Dataset(test_df,  img_dir, HAM10000Dataset.VAL_TRANSFORMS)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"[HAM10000] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def make_binary_acanthosis_dataset(ham_dir=HAM_DIR, batch_size=M2["batch_size"],
                                    num_workers=2):
    """
    Converts HAM10000 into a binary proxy dataset for Acanthosis Nigricans:
      - Positive (label=1): 'akiec' (Actinic Keratosis) — keratinocyte proliferation,
                             metabolically linked, used as morphological proxy for AN
      - Negative (label=0): everything else (sampled to balance)

    PAPER NOTE: This proxy approach is an acknowledged limitation; a true AN-labeled
    dataset is proposed as future contribution. Cite: HAM10000 (Tschandl et al., 2018).
    """
    csv_path = Path(ham_dir) / "HAM10000_metadata.csv"
    img_dir  = Path(ham_dir) / "images"

    df = pd.read_csv(csv_path).drop_duplicates("image_id")
    df["binary_label"] = (df["dx"] == "akiec").astype(int)

    # Balance classes via undersampling majority
    pos = df[df["binary_label"] == 1]
    neg = df[df["binary_label"] == 0].sample(len(pos) * 3, random_state=SEED)
    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)

    train_df, temp_df = train_test_split(
        balanced, test_size=0.30, stratify=balanced["binary_label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["binary_label"], random_state=SEED
    )

    class BinaryHAMDataset(Dataset):
        TRAIN_TF = HAM10000Dataset.TRAIN_TRANSFORMS
        VAL_TF   = HAM10000Dataset.VAL_TRANSFORMS

        def __init__(self, df, img_dir, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = Path(img_dir)
            self.transform = transform

        def __len__(self): return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(self.img_dir / f"{row['image_id']}.jpg").convert("RGB")
            if self.transform: img = self.transform(img)
            return img, int(row["binary_label"])

    train_ds = BinaryHAMDataset(train_df, img_dir, HAM10000Dataset.TRAIN_TRANSFORMS)
    val_ds   = BinaryHAMDataset(val_df,   img_dir, HAM10000Dataset.VAL_TRANSFORMS)
    test_ds  = BinaryHAMDataset(test_df,  img_dir, HAM10000Dataset.VAL_TRANSFORMS)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    print(f"[AN-Proxy] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return (DataLoader(train_ds, shuffle=True, **loader_kwargs),
            DataLoader(val_ds,   shuffle=False, **loader_kwargs),
            DataLoader(test_ds,  shuffle=False, **loader_kwargs))


# ─────────────────────────────────────────────
# 4. MODULE 3 — FER2013 FACIAL EXPRESSION
# ─────────────────────────────────────────────

class FERDataset(Dataset):
    """
    FER2013 Facial Expression Recognition dataset.
    Expects:
      <fer_dir>/train/  and  <fer_dir>/test/
      Each subdir has class-named folders: angry, disgust, fear, happy, sad, surprise, neutral
    """

    TRAIN_TRANSFORMS = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((M3["img_size"], M3["img_size"])),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    VAL_TRANSFORMS = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((M3["img_size"], M3["img_size"])),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self, root_dir: Path, split: str = "train"):
        from torchvision.datasets import ImageFolder
        transform = self.TRAIN_TRANSFORMS if split == "train" else self.VAL_TRANSFORMS
        self._ds = ImageFolder(str(Path(root_dir) / split), transform=transform)

    def __len__(self): return len(self._ds)
    def __getitem__(self, idx): return self._ds[idx]
    @property
    def classes(self): return self._ds.classes


def load_fer_dataloaders(fer_dir=FER_DIR, batch_size=M3["batch_size"], num_workers=2):
    """Returns train_loader, val_loader (from train split), test_loader."""
    train_full = FERDataset(fer_dir, "train")

    # Split training set into train/val (85/15)
    n_val = int(0.15 * len(train_full._ds))
    n_train = len(train_full._ds) - n_val
    train_ds, val_ds = random_split(
        train_full._ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_full = FERDataset(fer_dir, "test")
    test_ds   = test_full._ds

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"[FER2013] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 5. MODULE 4 — NHANES / PIMA TABULAR DATA
# ─────────────────────────────────────────────

NHANES_FEATURE_RENAME = {
    # Maps NHANES variable names -> our standard feature names
    "RIDAGEYR": "age",
    "BMXBMI":   "bmi",
    "BMXWAIST": "waist_circumference",
    "BPXSY1":   "systolic_bp",
    "BPXDI1":   "diastolic_bp",
    "LBXGLU":   "fasting_glucose",
    "LBXGH":    "hba1c",
    "LBXTC":    "total_cholesterol",
    "LBDHDD":   "hdl",
    "LBXTR":    "triglycerides",
    "PAD680":   "physical_activity",
    "SMQ020":   "smoking_status",
    "ALQ130":   "alcohol_use",
    "DIQ175A":  "family_history_diabetes",
    "DMDEDUC2": "education_level",
}


def load_nhanes(nhanes_dir=NHANES_DIR):
    """
    Load and preprocess NHANES data.
    Tries the nhanes PyPI package first; falls back to CSVs saved in nhanes_dir.
    Returns X (DataFrame), y (Series).
    """
    try:
        import nhanes
        from nhanes.load import load_NHANES_data
        df = load_NHANES_data(year="2017-2018",
                              NHANES_subset="demographics",
                              NHANES_dataset="DEMO_J")
        print("[NHANES] Loaded via nhanes PyPI package")
    except Exception:
        # Fallback: load pre-downloaded CSVs
        csv_path = Path(nhanes_dir) / "nhanes_diabetes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"NHANES CSV not found at {csv_path}. "
                "Run Notebook 1 to download it."
            )
        df = pd.read_csv(csv_path)
        print(f"[NHANES] Loaded from {csv_path}")

    # Rename columns if original NHANES names present
    df = df.rename(columns=NHANES_FEATURE_RENAME)

    feature_cols = M4["nhanes_features"]
    target_col   = M4["target_col"]

    # Keep only relevant columns that exist
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[NHANES] Missing columns (will be excluded): {missing}")

    df = df[available + [target_col]].dropna()
    X = df[available]
    y = df[target_col].astype(int)

    print(f"[NHANES] Shape: {X.shape} | Positive rate: {y.mean():.3f}")
    return X, y


def load_pima(pima_dir=PIMA_DIR):
    """Load and preprocess Pima Indians Diabetes dataset. Returns X, y."""
    csv_path = Path(pima_dir) / "diabetes.csv"
    df = pd.read_csv(csv_path)

    # Replace biologically implausible zeros with NaN then impute with median
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

    X = df[M4["pima_features"]]
    y = df["Outcome"].astype(int)
    print(f"[Pima] Shape: {X.shape} | Positive rate: {y.mean():.3f}")
    return X, y


def prepare_tabular_splits(X: pd.DataFrame, y: pd.Series,
                            test_size=M4["test_size"],
                            val_size=M4["val_size"]):
    """
    Stratified train/val/test split + StandardScaler.
    Returns: X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, stratify=y_train, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    print(f"[Tabular] Train: {len(X_train_sc)} | Val: {len(X_val_sc)} | Test: {len(X_test_sc)}")
    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler
