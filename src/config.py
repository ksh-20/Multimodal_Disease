"""
config.py
=========
Central configuration for the Explainable Multimodal Diabetes/Metabolic Risk Framework.
All notebooks and src modules import from here to stay in sync.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# 1. ROOT PATHS
# ─────────────────────────────────────────────
# Colab: '/content/drive/MyDrive/MultimodalDisease'
# Local: repository root
BASE_DIR = Path(os.environ.get("MMDISEASE_BASE", Path(__file__).resolve().parent.parent))

DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = BASE_DIR / "saved_models"
OUTPUTS_DIR   = BASE_DIR / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
SCORES_DIR    = OUTPUTS_DIR / "scores"
LOGS_DIR      = OUTPUTS_DIR / "logs"

# Create dirs if they don't exist (safe to call multiple times)
for _d in [DATA_DIR, MODELS_DIR, FIGURES_DIR, SCORES_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATASET PATHS
# ─────────────────────────────────────────────
APTOS_DIR    = DATA_DIR / "aptos2019"       # Kaggle: aptos2019-blindness-detection
HAM_DIR      = DATA_DIR / "ham10000"        # Kaggle: kmader/skin-lesion-analysis-toward-melanoma-detection
FER_DIR      = DATA_DIR / "fer2013"         # Kaggle: msambare/fer2013
NHANES_DIR   = DATA_DIR / "nhanes"          # via nhanes PyPI package or CDC
PIMA_DIR     = DATA_DIR / "pima"            # Kaggle: uciml/pima-indians-diabetes-database

# ─────────────────────────────────────────────
# 3. GLOBAL TRAINING DEFAULTS
# ─────────────────────────────────────────────
SEED         = 42
DEVICE       = "cuda"          # overridden to "cpu" if CUDA unavailable
NUM_WORKERS  = 2               # DataLoader workers (set 0 on Windows)
PIN_MEMORY   = True

# ─────────────────────────────────────────────
# 4. MODULE 1 — RETINOPATHY (M1)
# ─────────────────────────────────────────────
M1 = dict(
    backbone        = "resnet18",       # torchvision model name
    num_classes     = 5,                # APTOS severity: 0-4
    img_size        = 224,
    batch_size      = 32,
    epochs          = 20,
    lr              = 1e-4,
    weight_decay    = 1e-5,
    scheduler       = "cosine",         # 'cosine' | 'step'
    pretrained      = True,
    freeze_backbone = False,
    dropout         = 0.4,
    # Grad-CAM target layer (last conv block of ResNet18)
    gradcam_layer   = "layer4",
    # Score mapping: severity -> continuous risk [0,1]
    # 0=no DR, 1=mild, 2=moderate, 3=severe, 4=proliferative
    severity_weights = [0.0, 0.25, 0.50, 0.75, 1.0],
    model_path      = str(MODELS_DIR / "m1_retinopathy.pth"),
    scores_path     = str(SCORES_DIR / "m1_risk_scores.npy"),
)

# ─────────────────────────────────────────────
# 5. MODULE 2 — ACANTHOSIS NIGRICANS (M2)
# ─────────────────────────────────────────────
M2 = dict(
    backbone        = "resnet18",       # Transfer from HAM10000 pretraining
    num_classes     = 2,                # Binary: AN-like lesion present / absent (proxy)
    img_size        = 224,
    batch_size      = 32,
    epochs          = 15,
    lr              = 5e-5,
    weight_decay    = 1e-5,
    scheduler       = "cosine",
    pretrained      = True,
    freeze_backbone = True,             # Freeze early layers; only fine-tune head
    dropout         = 0.5,
    gradcam_layer   = "layer4",
    model_path      = str(MODELS_DIR / "m2_acanthosis.pth"),
    scores_path     = str(SCORES_DIR / "m2_risk_scores.npy"),
    # HAM10000 classes for pretraining
    ham_classes     = 7,
    ham_model_path  = str(MODELS_DIR / "m2_ham_pretrained.pth"),
)

# ─────────────────────────────────────────────
# 6. MODULE 3 — STRESS ESTIMATION (M3)
# ─────────────────────────────────────────────
M3 = dict(
    backbone        = "custom_cnn",     # Lightweight CNN for 48x48 grayscale
    num_classes     = 7,                # FER2013 emotions
    img_size        = 48,
    batch_size      = 64,
    epochs          = 30,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    scheduler       = "step",
    step_size       = 10,
    gamma           = 0.5,
    pretrained      = False,
    dropout         = 0.3,
    gradcam_layer   = "conv4",          # Last conv block in custom CNN
    model_path      = str(MODELS_DIR / "m3_stress.pth"),
    scores_path     = str(SCORES_DIR / "m3_risk_scores.npy"),
    # FER2013 class indices -> emotion names
    emotions        = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    # High-stress emotions and their contribution weights
    stress_weights  = {0: 0.9, 1: 0.7, 2: 0.95, 3: 0.0, 4: 0.75, 5: 0.1, 6: 0.0},
)

# ─────────────────────────────────────────────
# 7. MODULE 4 — TABULAR DIABETES RISK (M4)
# ─────────────────────────────────────────────
M4 = dict(
    model_type      = "xgboost",        # 'xgboost' | 'random_forest'
    xgb = dict(
        n_estimators        = 300,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        use_label_encoder   = False,
        eval_metric         = "logloss",
        random_state        = 42,
        n_jobs              = -1,
    ),
    rf = dict(
        n_estimators    = 200,
        max_depth       = 10,
        min_samples_leaf= 5,
        random_state    = 42,
        n_jobs          = -1,
    ),
    nhanes_features = [
        "age", "bmi", "waist_circumference", "systolic_bp", "diastolic_bp",
        "fasting_glucose", "hba1c", "total_cholesterol", "hdl", "triglycerides",
        "physical_activity", "smoking_status", "alcohol_use",
        "family_history_diabetes", "education_level",
    ],
    pima_features   = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ],
    target_col      = "diabetes",
    test_size       = 0.20,
    val_size        = 0.10,
    model_path      = str(MODELS_DIR / "m4_tabular.pkl"),
    scores_path     = str(SCORES_DIR / "m4_risk_scores.npy"),
    shap_plots_dir  = str(FIGURES_DIR / "m4_shap"),
)

# ─────────────────────────────────────────────
# 8. MODULE 5 — FUSION (M5)
# ─────────────────────────────────────────────
M5 = dict(
    input_dim       = 4,                # One score per modality
    hidden_dims     = [16, 8],          # MLP hidden layers
    output_dim      = 1,                # Final risk probability
    dropout         = 0.2,
    batch_size      = 32,
    epochs          = 50,
    lr              = 5e-4,
    weight_decay    = 1e-4,
    modalities      = ["Retinopathy (M1)", "Acanthosis Nigricans (M2)",
                       "Stress (M3)", "Tabular Risk (M4)"],
    risk_thresholds = dict(low=0.33, moderate=0.66),
    model_path      = str(MODELS_DIR / "m5_fusion.pth"),
    scores_path     = str(SCORES_DIR / "m5_fusion_scores.npy"),
    shap_plots_dir  = str(FIGURES_DIR / "m5_shap"),
    ablation_path   = str(OUTPUTS_DIR / "ablation_results.csv"),
)

# ─────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────
EVAL = dict(
    metrics         = ["accuracy", "precision", "recall", "f1", "roc_auc"],
    average         = "weighted",
    results_csv     = str(OUTPUTS_DIR / "all_module_results.csv"),
    comparison_csv  = str(OUTPUTS_DIR / "baseline_comparison.csv"),
)

# ─────────────────────────────────────────────
# 10. VISUALIZATION
# ─────────────────────────────────────────────
VIZ = dict(
    figsize_default = (10, 6),
    figsize_wide    = (14, 6),
    figsize_square  = (8, 8),
    dpi             = 150,
    style           = "seaborn-v0_8-whitegrid",
    palette         = "husl",
    cmap_gradcam    = "jet",
    save_format     = "png",
)
