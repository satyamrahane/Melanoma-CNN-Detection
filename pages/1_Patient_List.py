"""
MelanomaAI - Patient List page
Patient directory with search, filters, and table view
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils import SHARED_CSS, get_topnav

st.set_page_config(
    page_title="Patient List - MelanomaAI",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(SHARED_CSS, unsafe_allow_html=True)
st.markdown(get_topnav("Patient List"), unsafe_allow_html=True)

# PAGE HEADER
st.markdown("""
<div class="page-header">
    <h1 class="page-title">Patient Directory</h1>
</div>
""", unsafe_allow_html=True)

# Initialize sample patient data
if 'patients' not in st.session_state:
    st.session_state.patients = [
        {
            'name': 'John Smith',
            'id': 'PT-001',
            'last_analysis': '2024-03-06',
            'verdict': 'MALIGNANT',
            'risk_score': 0.78,
            'actions': 'View Details'
        },
        {
            'name': 'Sarah Johnson',
            'id': 'PT-002', 
            'last_analysis': '2024-03-05',
            'verdict': 'BENIGN',
            'risk_score': 0.23,
            'actions': 'View Details'
        },
        {
            'name': 'Michael Chen',
            'id': 'PT-003',
            'last_analysis': '2024-03-04',
            'verdict': 'MALIGNANT',
            'risk_score': 0.92,
            'actions': 'View Details'
        },
        {
            'name': 'Emily Davis',
            'id': 'PT-004',
            'last_analysis': '2024-03-03',
            'verdict': 'BENIGN',
            'risk_score': 0.15,
            'actions': 'View Details'
        },
        {
            'name': 'Robert Wilson',
            'id': 'PT-005',
            'last_analysis': '2024-03-02',
            'verdict': 'MALIGNANT',
            'risk_score': 0.85,
            'actions': 'View Details'
        }
    ]

# Stats section
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">PATIENT OVERVIEW</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-chip">
        <div class="metric-label">TOTAL PATIENTS</div>
        <div class="metric-value" style="color: #00D4FF;">""" + str(len(st.session_state.patients)) + """</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    critical_alerts = len([p for p in st.session_state.patients if p['verdict'] == 'MALIGNANT'])
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">CRITICAL ALERTS</div>
        <div class="metric-value" style="color: #FF3B5C;">{critical_alerts}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_risk = sum(p['risk_score'] for p in st.session_state.patients) / len(st.session_state.patients)
    st.markdown(f"""
    <div class="metric-chip">
        <div class="metric-label">AVERAGE RISK</div>
        <div class="metric-value" style="color: #FFB800;">{avg_risk:.2f}</div>
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

search_col1, search_col2, search_col3 = st.columns([2, 1, 1])

with search_col1:
    search_term = st.text_input("Search patients", placeholder="Search by name or ID...", key="patient_search")

with search_col2:
    status_filter = st.selectbox("Filter by Status", ["All", "MALIGNANT", "BENIGN"], key="status_filter")

with search_col3:
    if st.button("➕ ADD PATIENT", type="primary"):
        st.info("Add patient functionality would be implemented here")

st.markdown("</div></div>", unsafe_allow_html=True)

# Filter patients based on search
filtered_patients = st.session_state.patients
if search_term:
    filtered_patients = [
        p for p in filtered_patients 
        if search_term.lower() in p['name'].lower() or search_term.lower() in p['id'].lower()
    ]

if status_filter != "All":
    filtered_patients = [
        p for p in filtered_patients 
        if p['verdict'] == status_filter
    ]

# Patient table
st.markdown("""
<div class="card">
    <div class="card-header">
        <span class="card-title">PATIENT RECORDS</span>
        <span style="color: var(--muted); font-size: 0.8rem;">Showing 1-""" + str(len(filtered_patients)) + """ of """ + str(len(st.session_state.patients)) + """ records</span>
    </div>
    <div class="card-body">
""", unsafe_allow_html=True)

# Create table with custom styling
for i, patient in enumerate(filtered_patients[:5]):  # Show first 5 for pagination demo
    verdict_class = "badge-danger" if patient['verdict'] == 'MALIGNANT' else "badge-success"
    risk_color = "#FF3B5C" if patient['risk_score'] > 0.7 else "#FFB800" if patient['risk_score'] > 0.4 else "#00E5A0"
    
    st.markdown(f"""
    <div style="border-bottom: 1px solid var(--border); padding: 12px 0;">
        <table class="data-table">
            <tr>
                <td style="width: 25%; font-weight: 600;">{patient['name']}</td>
                <td style="width: 15%; color: var(--muted);">{patient['id']}</td>
                <td style="width: 20%; color: var(--muted);">{patient['last_analysis']}</td>
                <td style="width: 15%;">
                    <span class="badge {verdict_class}">{patient['verdict']}</span>
                </td>
                <td style="width: 25%;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div class="progress-bar" style="width: 100px;">
                            <div class="progress-fill" style="width: {patient['risk_score']*100}%; background: {risk_color};"></div>
                        </div>
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;">{patient['risk_score']:.2f}</span>
                    </div>
                </td>
                <td style="width: 10%;">
                    <button class="btn-cyan" style="padding: 6px 12px; font-size: 0.8rem;" onclick="window.location.href='/'">
                        Dashboard
                    </button>
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# Pagination
if len(filtered_patients) > 5:
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: var(--muted);">
        <div style="font-family: 'IBM Plex Mono', monospace;">« 1 2 3 4 5 »</div>
        <div style="font-size: 0.8rem; margin-top: 5px;">Page 1 of """ + str((len(filtered_patients) + 4) // 5) + """</div>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer-bar">
    © 2023 MelanomaAI Systems · HIPAA COMPLIANT · FDA CLASS II ENCRYPTED
</div>
""", unsafe_allow_html=True)
