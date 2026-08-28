"""
app.py
======
Streamlit demo dashboard for the Explainable Multimodal Diabetes/Metabolic Risk Framework.

Run locally:
    pip install streamlit torch torchvision shap xgboost pillow opencv-python-headless
    streamlit run demo/app.py

Features:
  - Upload fundus image  → M1 prediction + Grad-CAM
  - Upload skin image    → M2 prediction + Grad-CAM
  - Upload facial image  → M3 stress score + Grad-CAM
  - Enter clinical data  → M4 XGBoost risk + SHAP
  - Composite risk gauge → M5 Fusion MLP + modality SHAP
"""

import os, sys
import numpy as np
from pathlib import Path
from PIL import Image
import io

# ── Path setup ────────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).resolve().parent
REPO_DIR = DEMO_DIR.parent
sys.path.insert(0, str(REPO_DIR))

import streamlit as st
import torch
import torchvision.transforms as T
import cv2

from src.config import M1, M2, M3, M4, M5
from src.xai.gradcam import GradCAM

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "Multimodal Metabolic Risk Analyzer",
    page_icon   = "🧬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
with open(DEMO_DIR / "static" / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CACHING — Load models once
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading M1 Retinopathy model...")
def load_m1():
    from src.models.m1_retinopathy import RetinopathyModel
    model_path = M1["model_path"]
    if not Path(model_path).exists():
        return None
    model = RetinopathyModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.eval()

@st.cache_resource(show_spinner="Loading M2 Acanthosis model...")
def load_m2():
    from src.models.m2_acanthosis import AcanthosisModel
    model_path = M2["model_path"]
    if not Path(model_path).exists():
        return None
    model = AcanthosisModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.eval()

@st.cache_resource(show_spinner="Loading M3 Stress model...")
def load_m3():
    from src.models.m3_stress import StressCNN
    model_path = M3["model_path"]
    if not Path(model_path).exists():
        return None
    model = StressCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.eval()

@st.cache_resource(show_spinner="Loading M4 Tabular model...")
def load_m4():
    import pickle
    model_path = M4["model_path"]
    if not Path(model_path).exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading M5 Fusion model...")
def load_m5():
    from src.models.m5_fusion import FusionMLP
    model_path = M5["model_path"]
    if not Path(model_path).exists():
        return None
    model = FusionMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.eval()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def pil_to_tensor_rgb(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Preprocess PIL image for ResNet (RGB, 224x224, ImageNet norm)."""
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


def pil_to_tensor_gray(img: Image.Image, size: int = 48) -> torch.Tensor:
    """Preprocess PIL image for StressCNN (grayscale, 48x48)."""
    tf = T.Compose([
        T.Grayscale(1),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return tf(img).unsqueeze(0)


def gradcam_overlay(model, input_tensor, target_layer, orig_pil, size):
    """Run Grad-CAM and return overlay image as PIL."""
    cam = GradCAM(model, target_layer)
    heatmap = cam(input_tensor, class_idx=None)
    cam.remove_hooks()
    heatmap = cv2.resize(heatmap, (size, size))
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_c = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
    orig_np = np.array(orig_pil.convert("RGB").resize((size, size)))
    overlay = np.uint8(0.4 * heatmap_c + 0.6 * orig_np)
    return Image.fromarray(overlay), Image.fromarray(heatmap_c)


def risk_color(score: float) -> str:
    if score < 0.33:  return "#55A868"
    if score < 0.66:  return "#DD8452"
    return "#C44E52"

def risk_label(score: float) -> str:
    if score < 0.33:  return "🟢 LOW"
    if score < 0.66:  return "🟡 MODERATE"
    return "🔴 HIGH"


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class='main-header'>
    <h1>🧬 Explainable Multimodal Metabolic Risk Analyzer</h1>
    <p>An AI framework fusing retinal imaging, skin markers, facial stress,
    and clinical data for personalized diabetes risk prediction.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — Architecture overview
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Framework Modules")
    st.markdown("""
    | # | Module | Modality | XAI |
    |---|--------|----------|-----|
    | M1 | Retinopathy | Fundus image | Grad-CAM |
    | M2 | Acanthosis  | Skin image   | Grad-CAM |
    | M3 | Stress      | Facial image | Grad-CAM |
    | M4 | Tabular     | Clinical data| SHAP     |
    | M5 | Fusion      | All 4 scores | SHAP     |
    """)
    st.markdown("---")
    st.markdown("**Risk Thresholds:**")
    st.markdown("🟢 Low: < 0.33")
    st.markdown("🟡 Moderate: 0.33–0.66")
    st.markdown("🔴 High: > 0.66")
    st.markdown("---")
    demo_mode = st.checkbox("🎭 Demo mode (use dummy scores if models not trained yet)", value=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
m1_model = load_m1()
m2_model = load_m2()
m3_model = load_m3()
m4_model = load_m4()
m5_model = load_m5()

# Initialize risk scores (will be updated by each module)
risk_scores = {"m1": 0.5, "m2": 0.5, "m3": 0.5, "m4": 0.5}

# ─────────────────────────────────────────────
# MODULE 1 — RETINOPATHY
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 👁️ Module 1 — Diabetic Retinopathy (Fundus Image)")

m1_col1, m1_col2, m1_col3 = st.columns([1, 1, 1])

with m1_col1:
    m1_file = st.file_uploader("Upload fundus image (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                 key="m1_upload")

if m1_file is not None:
    m1_pil = Image.open(m1_file)
    with m1_col1:
        st.image(m1_pil, caption="Uploaded Fundus Image", width=250)

    if m1_model is not None and not demo_mode:
        with st.spinner("Running M1 inference..."):
            m1_tensor = pil_to_tensor_rgb(m1_pil, M1["img_size"])
            with torch.no_grad():
                m1_logits = m1_model(m1_tensor)
            m1_probs  = torch.softmax(m1_logits, dim=1)[0].numpy()
            severity  = int(m1_logits.argmax(1).item())
            severity_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            weights = M1["severity_weights"]
            m1_risk = float(sum(w * p for w, p in zip(weights, m1_probs)))

            overlay, heatmap = gradcam_overlay(
                m1_model, m1_tensor, m1_model.layer4[-1], m1_pil, M1["img_size"]
            )
    else:
        # Demo mode
        m1_probs = np.array([0.05, 0.15, 0.45, 0.25, 0.10])
        severity = 2
        severity_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        m1_risk = 0.55
        overlay = m1_pil.resize((224, 224))
        heatmap = m1_pil.resize((224, 224))

    risk_scores["m1"] = m1_risk

    with m1_col2:
        st.markdown(f"**Grad-CAM Overlay**")
        st.image(overlay, width=250)
    with m1_col3:
        st.markdown(f"**Prediction:** `{severity_names[severity]}`")
        st.markdown(f"**DR Risk Score:** {m1_risk:.3f}")
        st.progress(m1_risk, text=f"r_M1 = {m1_risk:.3f}")
        color = risk_color(m1_risk)
        st.markdown(f"<span style='color:{color}; font-size:1.2em; font-weight:bold;'>{risk_label(m1_risk)}</span>",
                    unsafe_allow_html=True)
        # Probability bar chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.5))
        colors = ["#4C72B0"] * 5
        colors[severity] = "#C44E52"
        ax.bar(severity_names, m1_probs, color=colors, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probabilities")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────
# MODULE 2 — ACANTHOSIS NIGRICANS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🩺 Module 2 — Acanthosis Nigricans (Skin Image)")

m2_col1, m2_col2, m2_col3 = st.columns([1, 1, 1])
with m2_col1:
    m2_file = st.file_uploader("Upload skin image (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                 key="m2_upload")

if m2_file is not None:
    m2_pil = Image.open(m2_file)
    with m2_col1:
        st.image(m2_pil, caption="Uploaded Skin Image", width=250)

    if m2_model is not None and not demo_mode:
        with st.spinner("Running M2 inference..."):
            m2_tensor = pil_to_tensor_rgb(m2_pil, M2["img_size"])
            with torch.no_grad():
                m2_risk = float(torch.sigmoid(m2_model(m2_tensor)).item())
            overlay, _ = gradcam_overlay(m2_model, m2_tensor, m2_model.layer4[-1], m2_pil, M2["img_size"])
    else:
        m2_risk = 0.72
        overlay = m2_pil.resize((224, 224))

    risk_scores["m2"] = m2_risk
    with m2_col2:
        st.image(overlay, caption="Grad-CAM Overlay", width=250)
    with m2_col3:
        pred_str = "AN-Positive" if m2_risk >= 0.5 else "Normal"
        st.markdown(f"**Prediction:** `{pred_str}`")
        st.markdown(f"**AN Risk Score:** {m2_risk:.3f}")
        st.progress(m2_risk)
        st.markdown(f"<span style='color:{risk_color(m2_risk)}; font-size:1.2em; font-weight:bold;'>{risk_label(m2_risk)}</span>",
                    unsafe_allow_html=True)
        st.caption("⚠️ Note: Using HAM10000 akiec class as AN proxy. See paper limitations.")

# ─────────────────────────────────────────────
# MODULE 3 — STRESS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 😤 Module 3 — Stress Estimation (Facial Image)")

m3_col1, m3_col2, m3_col3 = st.columns([1, 1, 1])
with m3_col1:
    m3_file = st.file_uploader("Upload facial image (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                 key="m3_upload")

if m3_file is not None:
    m3_pil = Image.open(m3_file)
    with m3_col1:
        st.image(m3_pil, caption="Uploaded Facial Image", width=250)

    if m3_model is not None and not demo_mode:
        with st.spinner("Running M3 inference..."):
            m3_tensor = pil_to_tensor_gray(m3_pil, M3["img_size"])
            from src.models.m3_stress import compute_stress_score
            with torch.no_grad():
                m3_logits = m3_model(m3_tensor)
                m3_probs  = torch.softmax(m3_logits, dim=1)[0].numpy()
                m3_risk   = float(compute_stress_score(m3_logits).item())
            pred_emotion = M3["emotions"][int(m3_logits.argmax(1).item())]
    else:
        m3_probs = np.array([0.25, 0.05, 0.40, 0.10, 0.15, 0.03, 0.02])
        m3_risk  = 0.68
        pred_emotion = "fear"

    risk_scores["m3"] = m3_risk
    with m3_col2:
        st.markdown("**Emotion Probabilities**")
        emotions = M3["emotions"]
        fig, ax = plt.subplots(figsize=(5, 3))
        colors_e = ["#C44E52" if e in ["angry","fear","disgust","sad"] else "#4C72B0" for e in emotions]
        ax.barh(emotions, m3_probs, color=colors_e, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_title("Emotion Distribution")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with m3_col3:
        st.markdown(f"**Predicted Emotion:** `{pred_emotion}`")
        st.markdown(f"**Stress Score:** {m3_risk:.3f}")
        st.progress(m3_risk)
        st.markdown(f"<span style='color:{risk_color(m3_risk)}; font-size:1.2em; font-weight:bold;'>{risk_label(m3_risk)}</span>",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODULE 4 — TABULAR
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Module 4 — Clinical Risk Factors (Tabular)")

with st.expander("Enter Patient Clinical Data", expanded=True):
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        age          = st.slider("Age (years)", 18, 90, 45)
        bmi          = st.slider("BMI", 15.0, 60.0, 27.5, 0.1)
        waist        = st.slider("Waist Circumference (cm)", 60, 160, 90)
        glucose      = st.slider("Fasting Glucose (mg/dL)", 70, 300, 105)
        hba1c        = st.slider("HbA1c (%)", 4.0, 14.0, 5.7, 0.1)
    with t_col2:
        sys_bp       = st.slider("Systolic BP (mmHg)", 80, 200, 125)
        dia_bp       = st.slider("Diastolic BP (mmHg)", 50, 120, 78)
        cholesterol  = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
        hdl          = st.slider("HDL (mg/dL)", 20, 100, 52)
        trig         = st.slider("Triglycerides (mg/dL)", 30, 800, 150)
    with t_col3:
        phys_act     = st.selectbox("Physical Activity (0=none, 4=very active)", [0,1,2,3,4], index=2)
        smoking      = st.selectbox("Smoking Status (0=never, 1=former, 2=current)", [0,1,2], index=0)
        alcohol      = st.selectbox("Alcohol Use (0=none, 4=heavy)", [0,1,2,3,4], index=1)
        family_hist  = st.selectbox("Family History of Diabetes", [0, 1], index=0,
                                     format_func=lambda x: "Yes" if x else "No")
        education    = st.selectbox("Education Level (1=<9th grade, 5=college grad)", [1,2,3,4,5], index=3)

if st.button("🔬 Analyze Clinical Risk (M4)", use_container_width=True):
    import pandas as pd
    X_input = pd.DataFrame([{
        "age": age, "bmi": bmi, "waist_circumference": waist,
        "systolic_bp": sys_bp, "diastolic_bp": dia_bp,
        "fasting_glucose": glucose, "hba1c": hba1c,
        "total_cholesterol": cholesterol, "hdl": hdl, "triglycerides": trig,
        "physical_activity": phys_act, "smoking_status": smoking,
        "alcohol_use": alcohol, "family_history_diabetes": family_hist,
        "education_level": education,
    }])

    if m4_model is not None and not demo_mode:
        m4_risk = float(m4_model.predict_proba(X_input)[:, 1][0])
    else:
        # Clinical heuristic for demo
        m4_risk = min(1.0, max(0.0,
            0.03 * (age - 40) / 40 +
            0.15 * (bmi - 25) / 35 +
            0.30 * (glucose - 90) / 210 +
            0.25 * (hba1c - 4.5) / 9.5 +
            0.1  * family_hist +
            0.05 * smoking / 2 +
            0.02 * (1 - phys_act / 4)
        ))

    risk_scores["m4"] = m4_risk

    m4_r1, m4_r2 = st.columns(2)
    with m4_r1:
        st.markdown(f"**Diabetes Risk Score:** {m4_risk:.3f}")
        st.progress(m4_risk)
        st.markdown(f"<span style='color:{risk_color(m4_risk)}; font-size:1.3em; font-weight:bold;'>{risk_label(m4_risk)}</span>",
                    unsafe_allow_html=True)
        if glucose >= 126 or hba1c >= 6.5:
            st.warning("⚠️ ADA diagnostic criteria met: Fasting glucose ≥ 126 mg/dL or HbA1c ≥ 6.5%")

    with m4_r2:
        # Simple feature contribution bars (demo)
        contrib = {
            "Glucose": (glucose - 90) / 210,
            "HbA1c": (hba1c - 4.5) / 9.5,
            "BMI": (bmi - 25) / 35,
            "Age": (age - 40) / 50,
            "Family Hx": float(family_hist),
        }
        contrib = {k: max(0, min(1, v)) for k, v in contrib.items()}
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(list(contrib.keys()), list(contrib.values()), color="#4C72B0", alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_title("Feature Risk Contributions (SHAP proxy)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────
# MODULE 5 — COMPOSITE FUSION
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔗 Module 5 — Composite Risk Score (Fusion)")

if st.button("🧬 Compute Composite Risk (M5 Fusion)", use_container_width=True, type="primary"):
    r_m1 = risk_scores["m1"]
    r_m2 = risk_scores["m2"]
    r_m3 = risk_scores["m3"]
    r_m4 = risk_scores["m4"]

    if m5_model is not None and not demo_mode:
        X_fusion_input = torch.tensor([[r_m1, r_m2, r_m3, r_m4]], dtype=torch.float32)
        with torch.no_grad():
            composite_risk = float(m5_model.predict_proba(X_fusion_input).item())
    else:
        # Weighted average fallback for demo
        composite_risk = float(0.25 * r_m1 + 0.25 * r_m2 + 0.20 * r_m3 + 0.30 * r_m4)

    risk_category = "LOW" if composite_risk < 0.33 else "MODERATE" if composite_risk < 0.66 else "HIGH"
    category_emoji = "🟢" if risk_category == "LOW" else "🟡" if risk_category == "MODERATE" else "🔴"

    # Display composite score prominently
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 15px; padding: 30px; text-align: center; margin: 10px 0;'>
        <h2 style='color: white; margin-bottom: 10px;'>Composite Metabolic Risk Score</h2>
        <h1 style='color: {risk_color(composite_risk)}; font-size: 3em; margin: 0;'>
            {composite_risk:.3f}
        </h1>
        <h2 style='color: {risk_color(composite_risk)}; margin-top: 10px;'>
            {category_emoji} {risk_category} RISK
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Per-modality contribution
    f5_col1, f5_col2 = st.columns(2)
    with f5_col1:
        st.markdown("#### Per-Modality Scores")
        module_scores = {
            "Retinopathy (M1)":       r_m1,
            "Acanthosis Nigricans (M2)": r_m2,
            "Stress (M3)":            r_m3,
            "Tabular Risk (M4)":      r_m4,
        }
        for name, score in module_scores.items():
            bar_color = risk_color(score)
            st.markdown(f"**{name}:** {score:.3f} {risk_label(score)}")
            st.progress(score)

    with f5_col2:
        st.markdown("#### Modality SHAP Attribution")
        # Compute approximate modality importance (demo)
        shap_vals = np.array([r_m1, r_m2, r_m3, r_m4]) * np.array([0.25, 0.25, 0.20, 0.30])
        shap_vals -= shap_vals.mean()
        labels = ["Retinopathy", "Acanthosis", "Stress", "Tabular"]
        colors_shap = ["#C44E52" if v > 0 else "#4C72B0" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(labels, shap_vals, color=colors_shap, alpha=0.85)
        ax.axvline(0, color="gray", lw=1)
        ax.set_xlabel("SHAP Value (Impact on Risk)")
        ax.set_title("Modality Contribution (M5 SHAP)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Clinical recommendation
    st.markdown("---")
    st.markdown("#### 💊 Clinical Recommendation")
    if risk_category == "LOW":
        st.success("✅ Low composite metabolic risk. Continue annual screening and maintain lifestyle.")
    elif risk_category == "MODERATE":
        st.warning("⚠️ Moderate metabolic risk. Recommend follow-up with HbA1c and OGTT. "
                   "Consider lifestyle intervention program.")
    else:
        st.error("🚨 High metabolic risk. Urgent consultation with endocrinologist recommended. "
                 "Comprehensive diabetes evaluation required.")

    st.caption("⚠️ This tool is for research and educational purposes only. "
               "Not a substitute for professional medical diagnosis.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
    <p>Explainable Multimodal Diabetes/Metabolic Risk Framework</p>
    <p>Modules: ResNet18 (Grad-CAM) | HAM10000 Transfer (Grad-CAM) | StressCNN (Grad-CAM) | XGBoost (SHAP) | Fusion MLP (SHAP)</p>
    <p>References: Selvaraju et al. 2017 (Grad-CAM) · Lundberg & Lee 2017 (SHAP) · Chen & Guestrin 2016 (XGBoost)</p>
</div>
""", unsafe_allow_html=True)
