# Explainable Multimodal Diabetes/Metabolic Risk Framework

> **An Explainable Multimodal Intelligence Framework for Personalized Diabetes and Metabolic Risk Prediction**

A complete implementation of a 5-module late-fusion framework combining fundus imaging (retinopathy), skin imaging (acanthosis nigricans proxy), facial expression (stress), and tabular clinical data under a per-modality explainability regime.

---

## 📁 Project Structure

```
Multimodal_Disease/
├── src/
│   ├── config.py               # All hyperparameters and paths
│   ├── models/
│   │   ├── m1_retinopathy.py   # ResNet18 + Grad-CAM (APTOS 2019)
│   │   ├── m2_acanthosis.py    # 2-stage HAM10000 transfer + binary (Grad-CAM)
│   │   ├── m3_stress.py        # StressCNN + stress score (FER2013)
│   │   ├── m4_tabular.py       # XGBoost + SHAP (NHANES)
│   │   └── m5_fusion.py        # Fusion MLP + SHAP (Eqs M5.1–M5.5)
│   ├── utils/
│   │   ├── data_utils.py       # All dataset loaders
│   │   ├── eval_utils.py       # Metrics, ablation, hyperparameter table
│   │   └── viz_utils.py        # All Section 4 figures
│   └── xai/
│       ├── gradcam.py          # Grad-CAM (Selvaraju et al. 2017)
│       └── shap_explainer.py   # SHAP wrappers (Lundberg & Lee 2017)
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_M1_M2_Vision.ipynb
│   ├── 03_M3_Stress.ipynb
│   ├── 04_M4_Tabular.ipynb
│   └── 05_M5_Fusion.ipynb
├── demo/
│   ├── app.py                  # Streamlit dashboard
│   └── static/style.css
└── requirements.txt
```

---

## 🚀 Quick Start

### Google Colab (recommended)

1. Upload this repo to Google Drive as `MultimodalDisease/`
2. Open notebooks in order: 01 → 02 → 03 → 04 → 05
3. Each notebook starts with Drive mount + pip installs — **run in sequence**

```python
# At the top of every notebook:
import os, sys
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = '/content/drive/MyDrive/MultimodalDisease'
REPO_DIR = '/content/Multimodal_Disease'   # where you cloned the repo
os.environ['MMDISEASE_BASE'] = BASE_DIR
sys.path.insert(0, REPO_DIR)
```

### Local (Streamlit demo)

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

---

## 📦 Datasets

| Module | Dataset | How to get |
|--------|---------|-----------|
| M1 | **APTOS 2019** | `kaggle competitions download -c aptos2019-blindness-detection` |
| M2 | **HAM10000** | `kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection` |
| M3 | **FER2013** | `kaggle datasets download -d msambare/fer2013` |
| M4 | **NHANES 2017-18** | Via `nhanes` PyPI package (auto-downloaded in Notebook 1) |
| M4 | **Pima Indians** | `kaggle datasets download -d uciml/pima-indians-diabetes-database` |

You need a [Kaggle API key](https://kaggle.com/settings/account) (`kaggle.json`).  
Notebook 1 walks you through setting it up.

---

## 🧮 Framework Architecture

```
Fundus Image  ──→  [M1: ResNet18]  ──→  r_M1 (Grad-CAM) ─┐
Skin Image    ──→  [M2: ResNet18]  ──→  r_M2 (Grad-CAM) ─┤
Facial Image  ──→  [M3: StressCNN] ──→  r_M3 (Grad-CAM) ─┼──→ [M5: Fusion MLP] ──→ r_final (SHAP)
Tabular Data  ──→  [M4: XGBoost]   ──→  r_M4 (SHAP)     ─┘
```

### Late Fusion (Eq. M5.4):
```
h₁ = ReLU(W₁ · [r_M1, r_M2, r_M3, r_M4] + b₁)
h₂ = ReLU(W₂ · h₁ + b₂)
r_final = sigmoid(W₃ · h₂ + b₃)
```

---

## 📊 Explainability Methods

| Module | XAI Method | Citation |
|--------|-----------|---------|
| M1, M2, M3 | **Grad-CAM** | Selvaraju et al., ICCV 2017 |
| M4 | **TreeSHAP** | Lundberg & Lee, NeurIPS 2017 |
| M5 | **KernelSHAP** | Lundberg & Lee, NeurIPS 2017 |

### Grad-CAM (Eq. 1-2):
```
α_k^c = (1/Z) ΣᵢΣⱼ [∂y^c / ∂A^k_{ij}]     (neuron importance weights)
L^c = ReLU( Σ_k α_k^c · A^k )              (class activation map)
```

### SHAP Shapley Value (Eq. M4.1):
```
φᵢ = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)! / |F|!] · [f(S∪{i}) - f(S)]
```

---

## 📈 Expected Results (Section 4.3)

| Module | Accuracy | F1 | AUC |
|--------|----------|----|-----|
| M1 Retinopathy (binary) | ~0.84 | ~0.83 | ~0.91 |
| M2 Acanthosis (proxy) | ~0.79 | ~0.78 | ~0.86 |
| M3 Stress (7-class) | ~0.62 | ~0.60 | — |
| M4 Tabular (XGB) | ~0.88 | ~0.87 | ~0.94 |
| **M5 Fusion (MLP)** | **~0.91** | **~0.90** | **~0.96** |

*Actual results depend on training run and dataset. Run Notebook 5 for your numbers.*

---

## ⚠️ Limitations (State in paper Section V)

1. **No standard AN dataset**: HAM10000 `akiec` is a morphological proxy. Propose dedicated AN dataset as contribution.
2. **Cross-modality alignment**: Fusion uses same-sized subsets of independent datasets. Joint patient cohort would be ideal.
3. **FER2013 quality**: Lab-collected, compressed images; may not generalize to real-world facial inputs.
4. **NHANES missing features**: Some NHANES variables may need manual download from CDC portal.

---

## 📚 Key References

1. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
2. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
3. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
4. He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
5. Tschandl et al. (2018). The HAM10000 dataset. *Scientific Data*.
6. Goodfellow et al. (2013). Challenges in Representation Learning (FER2013). *ICANN*.
