"""
MelanomaAI - Settings page
System configuration and API settings
"""

import streamlit as st
from utils import SHARED_CSS, get_topnav, load_model

st.set_page_config(
    page_title="Settings - MelanomaAI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown(get_topnav("Settings"), unsafe_allow_html=True)

# PAGE HEADER
st.markdown("""
<div class="page-header">
    <h1 class="page-title">System Settings</h1>
</div>
""", unsafe_allow_html=True)

# Main layout: Sidebar + Content
settings_col1, settings_col2 = st.columns([1, 3], gap="large")

with settings_col1:
    # Configuration Menu
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-title">CONFIGURATION MENU</span>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)
    
    menu_options = [
        "General", "Security", "AI Model", "API Settings", "User Management"
    ]
    
    for option in menu_options:
        is_active = option == "API Settings"
        active_style = "background: rgba(0,212,255,0.1); color: var(--primary);" if is_active else ""
        st.markdown(f"""
        <div style="padding: 12px 16px; border-radius: 8px; cursor: pointer; margin-bottom: 8px; {active_style}">
            <div style="font-weight: 600;">{option}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

with settings_col2:
    # API Configuration
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-title">API CONFIGURATION</span>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)
    
    # API Keys Section
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h4 style="color: white; margin-bottom: 15px;">Production API Keys</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col_api1, col_api2 = st.columns(2)
    
    with col_api1:
        st.text_input("Primary API Key", value="••••••••••••••••••••••••••••••••", disabled=True, key="primary_api")
    
    with col_api2:
        st.text_input("Secondary API Key", value="••••••••••••••••••••••••••••••••", disabled=True, key="secondary_api")
    
    # Usage Counter
    st.markdown(f"""
    <div style="margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;">
        <div style="font-family: 'IBM Plex Mono', monospace; color: var(--muted); font-size: 0.8rem;">24h USAGE COUNTER</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: var(--primary); margin-top: 5px;">1,247 / 5,000</div>
        <div style="background: var(--card); height: 4px; border-radius: 2px; margin-top: 10px;">
            <div style="background: var(--primary); width: 25%; height: 100%; border-radius: 2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Endpoint Configuration
    st.markdown("""
    <h4 style="color: white; margin: 30px 0 15px;">Endpoint Configuration</h4>
    """, unsafe_allow_html=True)
    
    st.text_input("Endpoint URL", value="https://api.melanomai.ai/v2", key="endpoint_url")
    st.text_input("Webhook Callback", value="https://clinic.melanomai.ai/webhook", key="webhook_url")
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # AI Model Settings
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-title">AI MODEL SETTINGS</span>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)
    
    # Model Selection
    model, model_path = load_model()
    current_model = model_path.split('/')[-1] if model_path else "No model loaded"
    
    st.selectbox(
        "Active Model Version",
        ["melanoma_final.keras", "melanoma_model_improved.keras", "best_phase1.keras"],
        index=0 if model_path else None,
        key="model_version"
    )
    
    # Confidence Threshold
    st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        key="confidence_threshold"
    )
    
    # Diagnostics Logic
    st.markdown("""
    <h4 style="color: white; margin: 20px 0 15px;">Diagnostics Logic</h4>
    """, unsafe_allow_html=True)
    
    st.toggle("Auto-Flag Malignancy", value=True, key="auto_flag")
    st.toggle("Detailed Feature Mapping", value=True, key="feature_mapping")
    st.toggle("Anonymize Patient Data", value=True, key="anonymize_data")
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # System Logs & Health
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-title">SYSTEM LOGS & HEALTH</span>
        </div>
        <div class="card-body">
    """, unsafe_allow_html=True)
    
    # Activity Stream
    st.markdown("""
    <h4 style="color: white; margin-bottom: 15px;">Real-time Activity Stream</h4>
    """, unsafe_allow_html=True)
    
    # Sample log entries
    log_entries = [
        {"time": "14:32:15", "event": "Prediction completed", "patient": "PT-001", "status": "SUCCESS"},
        {"time": "14:28:42", "event": "Model loaded", "patient": "-", "status": "INFO"},
        {"time": "14:25:18", "event": "API request processed", "patient": "PT-002", "status": "SUCCESS"},
        {"time": "14:22:33", "event": "System health check", "patient": "-", "status": "OK"},
        {"time": "14:18:27", "event": "Prediction completed", "patient": "PT-003", "status": "SUCCESS"},
    ]
    
    for log in log_entries:
        status_color = "#00E5A0" if log["status"] == "SUCCESS" else "#00D4FF" if log["status"] == "INFO" else "#FF3B5C"
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;">
            <span style="color: var(--muted);">{log['time']}</span>
            <span>{log['event']}</span>
            <span>{log['patient']}</span>
            <span style="color: {status_color};">{log['status']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # System Status
    st.markdown("""
    <div style="margin-top: 20px; padding: 15px; background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.2); border-radius: 8px; text-align: center;">
        <div style="color: #00E5A0; font-weight: 600; margin-bottom: 5px;">SYSTEM STATUS</div>
        <div style="color: white;">● All systems operational</div>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer-bar">
    © 2023 MelanomaAI Systems · HIPAA COMPLIANT · FDA CLASS II ENCRYPTED
</div>
""", unsafe_allow_html=True)
