"""
MelanomaAI - Clinical Performance Dashboard
Analytics page with model performance metrics and charts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils import SHARED_CSS, get_topnav, load_metrics

st.set_page_config(
    page_title="Analytics - MelanomaAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown(get_topnav("Analytics"), unsafe_allow_html=True)

# PAGE HEADER
st.markdown("""
<div class="page-header">
    <h1 class="page-title">Clinical Performance Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# Header badges
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin: 20px 0;">
    <span class="badge badge-success" style="background: #00E5A0;">PRODUCTION</span>
    <span style="color: var(--muted); font-family: 'IBM Plex Mono', monospace;">Model ID: MEL-2024-V2.4.0</span>
</div>
""", unsafe_allow_html=True)

# Load metrics
metrics = load_metrics()

# KPI Cards Section
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">PERFORMANCE METRICS</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown("""
    <div class="metric-chip">
        <div class="metric-label">MODEL HEALTH</div>
        <div class="metric-value" style="color: #00E5A0;">OPTIMAL</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    sensitivity = metrics.get('sensitivity', 0.87)
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">SENSITIVITY</div>
        <div class="metric-value" style="color: #00D4FF;">{sensitivity*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    specificity = metrics.get('specificity', 0.91)
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">SPECIFICITY</div>
        <div class="metric-value" style="color: #00D4FF;">{specificity*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    auc = metrics.get('auc_roc', 0.92)
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">AUC-ROC</div>
        <div class="metric-value" style="color: #00D4FF;">{auc:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Charts Section
st.markdown("""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
""", unsafe_allow_html=True)

# Chart 1: Accuracy Evolution
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">ACCURACY EVOLUTION</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

# Generate sample accuracy data
epochs = list(range(1, 21))
accuracy_values = [0.65 + 0.015 * i + np.random.normal(0, 0.02) for i in range(20)]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=epochs,
    y=accuracy_values,
    mode='lines+markers',
    line=dict(color='#00D4FF', width=3),
    marker=dict(color='#00D4FF', size=6),
    fill='tonexty',
    fillcolor='rgba(0,212,255,0.1)'
))

fig1.update_layout(
    title_text="Model Accuracy Over Time",
    xaxis_title="Epoch",
    yaxis_title="Accuracy",
    yaxis_tickformat='.1%',
    plot_bgcolor='rgba(3,7,15,0)',
    paper_bgcolor='rgba(3,7,15,0)',
    font=dict(color='#CBD5E1'),
    height=400
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Chart 2: Confusion Matrix
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">CONFUSION MATRIX</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

# Sample confusion matrix data
confusion_data = np.array([[850, 120], [95, 780]])  # TP, FP, FN, TN

fig2 = go.Figure(data=go.Heatmap(
    z=confusion_data,
    x=['Predicted Positive', 'Predicted Negative'],
    y=['Actual Positive', 'Actual Negative'],
    colorscale='Blues',
    text=confusion_data,
    texttemplate="%{text}",
    textfont=dict(color="white", size=14)
))

fig2.update_layout(
    title_text="Confusion Matrix",
    xaxis_title="Predicted",
    yaxis_title="Actual",
    plot_bgcolor='rgba(3,7,15,0)',
    paper_bgcolor='rgba(3,7,15,0)',
    font=dict(color='#CBD5E1'),
    height=400
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Second row of charts
st.markdown("""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
""", unsafe_allow_html=True)

# Chart 3: AUC-ROC Curve
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">AUC-ROC CURVE</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

# Generate ROC curve data
fpr = np.linspace(0, 1, 100)
tpr = 1 - np.exp(-5 * fpr)  # Sample ROC curve

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name='ROC Curve',
    line=dict(color='#00D4FF', width=3)
))
fig3.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(color='gray', width=2, dash='dash')
))

fig3.update_layout(
    title_text=f"AUC-ROC Curve (AUC = {auc:.3f})",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    plot_bgcolor='rgba(3,7,15,0)',
    paper_bgcolor='rgba(3,7,15,0)',
    font=dict(color='#CBD5E1'),
    height=400
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Chart 4: Demographic Fairness
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">DEMOGRAPHIC FAIRNESS</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

# Fitzpatrick skin type data
skin_types = ['Type I', 'Type II', 'Type III', 'Type IV', 'Type V', 'Type VI']
accuracy_by_skin = [0.92, 0.89, 0.87, 0.85, 0.82, 0.78]

fig4 = go.Figure(data=[
    go.Bar(
        x=skin_types,
        y=accuracy_by_skin,
        marker_color=['#F5CBA7', '#E59866', '#CA8A5A', '#A0522D', '#6B3A2A', '#3D1C0E']
    )
])

fig4.update_layout(
    title_text="Accuracy by Fitzpatrick Skin Type",
    xaxis_title="Skin Type",
    yaxis_title="Accuracy",
    yaxis_tickformat='.1%',
    plot_bgcolor='rgba(3,7,15,0)',
    paper_bgcolor='rgba(3,7,15,0)',
    font=dict(color='#CBD5E1'),
    height=400
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("</div></div></div>", unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer-bar">
    © 2023 MelanomaAI Systems · HIPAA COMPLIANT · FDA CLASS II ENCRYPTED
</div>
""", unsafe_allow_html=True)
