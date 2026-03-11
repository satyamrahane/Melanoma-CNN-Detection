import streamlit as st
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.model    import load_model
from backend.metrics  import get_threshold, load_metrics, load_risk_analysis
from frontend.components.sidebar    import render_sidebar
from frontend.components.topbar     import render_topbar
from frontend.components.diagnosis  import render_diagnosis
from frontend.components.evaluation import render_evaluation
from frontend.components.history    import render_history

st.set_page_config(
    page_title="MelanomaAI · Diagnostic Portal",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css(*paths):
    css = ""
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                css += f.read() + "\n"
    return f"<style>{css}</style>"

st.markdown(load_css(
    "frontend/styles/main.css",
    "frontend/styles/components.css"
), unsafe_allow_html=True)

if "history"         not in st.session_state: st.session_state.history = []
if "show_explainer"  not in st.session_state: st.session_state.show_explainer = False

@st.cache_resource
def get_model():
    return load_model()

model, model_path = get_model()
metrics   = load_metrics()
threshold = get_threshold()

mname   = os.path.basename(model_path) if model_path else "No model"
gpu_str = "GPU" if torch.cuda.is_available() else "CPU"
acc_str = f"{metrics.get('accuracy',0)*100:.1f}%" if metrics else "—"
auc_str = f"{metrics.get('auc_roc',0):.4f}"      if metrics else "—"

st.markdown(render_sidebar(), unsafe_allow_html=True)
st.markdown(render_topbar(mname, gpu_str, acc_str, auc_str), unsafe_allow_html=True)

if metrics:
    sens = metrics.get("sensitivity", 0)
    spec = metrics.get("specificity", 0)
    f1   = metrics.get("f1_score", 0)
    fp_c = metrics.get("risk_analysis", {}).get("fp_caught", 0)
    fp_r = metrics.get("risk_analysis", {}).get("fp_catch_rate", 0)
    st.markdown(f"""
    <div class="stitch-kpi-grid">
        <div class="stitch-kpi-card">
            <p class="stitch-kpi-label">Accuracy</p>
            <p class="stitch-kpi-val">{metrics.get('accuracy',0)*100:.1f}%</p>
        </div>
        <div class="stitch-kpi-card">
            <p class="stitch-kpi-label">AUC-ROC</p>
            <p class="stitch-kpi-val">{metrics.get('auc_roc',0):.3f}</p>
        </div>
        <div class="stitch-kpi-card">
            <p class="stitch-kpi-label">Sensitivity</p>
            <p class="stitch-kpi-val">{sens*100:.1f}%</p>
        </div>
        <div class="stitch-kpi-card">
            <p class="stitch-kpi-label">Specificity</p>
            <p class="stitch-kpi-val">{spec*100:.1f}%</p>
        </div>
        <div class="stitch-kpi-card">
            <p class="stitch-kpi-label">F1 Score</p>
            <p class="stitch-kpi-val">{f1:.3f}</p>
        </div>
        <div class="stitch-kpi-card">
            <p class="stitch-kpi-label">FP Caught</p>
            <p class="stitch-kpi-val">{fp_c}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬  Diagnosis", "📊  Model Evaluation", "📋  Session History"])

with tab1: render_diagnosis(model, threshold, metrics)
with tab2: render_evaluation(metrics, threshold)
with tab3: render_history(st.session_state.history)

st.markdown(f"""
</div>
<div style="border-top:1px solid #EEF2EF;padding:12px 28px 12px 100px;
    display:flex;justify-content:space-between;align-items:center;
    background:white;margin-top:24px">
    <div style="display:flex;align-items:center;gap:12px;font-size:.68rem;color:#8AA49F">
        <span>🔬 MelanomaAI v3.0</span>
        <span style="color:#EEF2EF">|</span>
        <span>{mname}</span>
        <span style="color:#EEF2EF">|</span>
        <span>Threshold {threshold:.2f}</span>
        <span style="color:#EEF2EF">|</span>
        <span>{gpu_str} · RTX 3050</span>
    </div>
    <div style="font-size:.65rem;color:#C8D5CF">
        Clinical decision support only · Not a substitute for professional diagnosis
    </div>
</div>
""", unsafe_allow_html=True)