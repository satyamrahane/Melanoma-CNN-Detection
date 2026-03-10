"""
MelanomaAI - Diagnostic Archive page
Historical case analysis with filtering and statistics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils import SHARED_CSS, get_topnav

st.set_page_config(
    page_title="Archive - MelanomaAI",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown(get_topnav("Archive"), unsafe_allow_html=True)

# PAGE HEADER
st.markdown("""
<div class="page-header">
    <h1 class="page-title">DIAGNOSTIC ARCHIVE</h1>
</div>
""", unsafe_allow_html=True)

# Initialize sample archive data
if 'archive_data' not in st.session_state:
    st.session_state.archive_data = [
        {
            'analysis_date': '2024-03-06 14:30',
            'patient_id': 'PT-001',
            'name': 'John Smith',
            'diagnosis': 'MALIGNANT',
            'confidence': 0.92,
            'risk_level': 'HIGH'
        },
        {
            'analysis_date': '2024-03-05 10:15',
            'patient_id': 'PT-002',
            'name': 'Sarah Johnson',
            'diagnosis': 'BENIGN',
            'confidence': 0.23,
            'risk_level': 'LOW'
        },
        {
            'analysis_date': '2024-03-04 16:45',
            'patient_id': 'PT-003',
            'name': 'Michael Chen',
            'diagnosis': 'MALIGNANT',
            'confidence': 0.87,
            'risk_level': 'HIGH'
        },
        {
            'analysis_date': '2024-03-03 09:20',
            'patient_id': 'PT-004',
            'name': 'Emily Davis',
            'diagnosis': 'BENIGN',
            'confidence': 0.15,
            'risk_level': 'LOW'
        },
        {
            'analysis_date': '2024-03-02 13:10',
            'patient_id': 'PT-005',
            'name': 'Robert Wilson',
            'diagnosis': 'MALIGNANT',
            'confidence': 0.78,
            'risk_level': 'MEDIUM'
        }
    ]

# Stats section
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">ARCHIVE STATISTICS</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">TOTAL CASES</div>
        <div class="metric-value" style="color: #00D4FF;">{len(st.session_state.archive_data)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    malignancy_rate = len([c for c in st.session_state.archive_data if c['diagnosis'] == 'MALIGNANT']) / len(st.session_state.archive_data) * 100
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">MALIGNANCY %</div>
        <div class="metric-value" style="color: #FF3B5C;">{malignancy_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_confidence = sum(c['confidence'] for c in st.session_state.archive_data) / len(st.session_state.archive_data)
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">AVG CONFIDENCE</div>
        <div class="metric-value" style="color: #00D4FF;">{avg_confidence:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Search and filters section
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">SEARCH & FILTERS</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

with filter_col1:
    search_term = st.text_input("Search archive", placeholder="Search by patient or case...", key="archive_search")

with filter_col2:
    date_range = st.date_input("Date Range", value=[datetime(2024, 3, 1), datetime(2024, 3, 6)], key="date_range")

with filter_col3:
    if st.button("🔍 APPLY FILTERS", type="primary"):
        st.success("Filters applied successfully")

st.markdown("</div></div>", unsafe_allow_html=True)

# Quick filters
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">QUICK FILTERS</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

quick_col1, quick_col2, quick_col3 = st.columns(3)

with quick_col1:
    outcome_filter = st.selectbox("Outcome", ["All", "MALIGNANT", "BENIGN"], key="outcome_filter")

with quick_col2:
    risk_filter = st.selectbox("Risk", ["All", "High Risk", "Medium Risk", "Low Risk"], key="risk_filter")

with quick_col3:
    clinic_filter = st.selectbox("Clinic", ["All", "Main Campus", "North Branch", "South Branch"], key="clinic_filter")

st.markdown("</div></div>", unsafe_allow_html=True)

# Archive table
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">CASE HISTORY</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

# Filter data based on search and filters
filtered_data = st.session_state.archive_data
if search_term:
    filtered_data = [
        c for c in filtered_data 
        if search_term.lower() in c['name'].lower() or search_term.lower() in c['patient_id'].lower()
    ]

if outcome_filter != "All":
    filtered_data = [c for c in filtered_data if c['diagnosis'] == outcome_filter]

if risk_filter != "All":
    filtered_data = [c for c in filtered_data if c['risk_level'] == risk_filter]

# Create table
for case in filtered_data:
    diagnosis_class = "badge-danger" if case['diagnosis'] == 'MALIGNANT' else "badge-success"
    risk_color = "#FF3B5C" if case['risk_level'] == 'HIGH' else "#FFB800" if case['risk_level'] == 'MEDIUM' else "#00E5A0"
    
    st.markdown(f"""
    <div style="border-bottom: 1px solid var(--border); padding: 12px 0;">
        <table class="data-table">
            <tr>
                <td style="width: 20%; font-weight: 600;">{case['analysis_date']}</td>
                <td style="width: 15%;">{case['patient_id']}</td>
                <td style="width: 20%;">{case['name']}</td>
                <td style="width: 15%;">
                    <span class="badge {diagnosis_class}">{case['diagnosis']}</span>
                </td>
                <td style="width: 15%;">
                    <div class="progress-bar" style="width: 80px;">
                        <div class="progress-fill" style="width: {case['confidence']*100}%; background: {risk_color};"></div>
                    </div>
                    <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; margin-left: 8px;">{case['confidence']:.2f}</span>
                </td>
                <td style="width: 15%;">
                    <span class="badge badge-danger" style="background: {risk_color};">{case['risk_level']}</span>
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Bottom analysis panels
st.markdown("""
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 20px 0;">
""", unsafe_allow_html=True)

# Panel 1: Case Type Distribution
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">CASE TYPE DISTRIBUTION</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

malignant_count = len([c for c in st.session_state.archive_data if c['diagnosis'] == 'MALIGNANT'])
benign_count = len([c for c in st.session_state.archive_data if c['diagnosis'] == 'BENIGN'])

fig1 = go.Figure(data=[
    go.Bar(name='MALIGNANT', x=['Cases'], y=[malignant_count], marker_color='#FF3B5C'),
    go.Bar(name='BENIGN', x=['Cases'], y=[benign_count], marker_color='#00E5A0')
])
fig1.update_layout(
    title_text="Case Distribution",
    showlegend=True,
    plot_bgcolor='rgba(3,7,15,0)',
    paper_bgcolor='rgba(3,7,15,0)',
    font=dict(color='#CBD5E1')
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Panel 2: Risk Score Trends
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">RISK SCORE TRENDS</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

# Create trend data
dates = [datetime(2024, 3, i) for i in range(1, 7)]
risk_scores = [0.78, 0.65, 0.82, 0.45, 0.91, 0.38, 0.73]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=dates,
    y=risk_scores,
    mode='lines+markers',
    line=dict(color='#00D4FF', width=3),
    marker=dict(color='#00D4FF', size=8),
    fill='tonexty',
    fillcolor='rgba(0,212,255,0.1)'
))

fig2.update_layout(
    title_text="Risk Trends (7 Days)",
    xaxis_title="Date",
    yaxis_title="Risk Score",
    plot_bgcolor='rgba(3,7,15,0)',
    paper_bgcolor='rgba(3,7,15,0)',
    font=dict(color='#CBD5E1')
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Panel 3: System Health
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">SYSTEM HEALTH</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

health_col1, health_col2 = st.columns(2)

with health_col1:
    st.markdown("""
    <div class="metric-chip">
        <div class="metric-label">MODEL STATUS</div>
        <div class="metric-value" style="color: #00E5A0;">OPTIMAL</div>
    </div>
    """, unsafe_allow_html=True)

with health_col2:
    st.markdown("""
    <div class="metric-chip">
        <div class="metric-label">UPTIME</div>
        <div class="metric-value" style="color: #00D4FF;">99.9%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div></div>", unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer-bar">
    © 2023 MelanomaAI Systems · HIPAA COMPLIANT · FDA CLASS II ENCRYPTED
</div>
""", unsafe_allow_html=True)
