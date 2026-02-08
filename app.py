import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# ========================================
# PERFECT CONFIGURATION - FULL PAGE LAYOUT
# ========================================
st.set_page_config(
    page_title="Telco Churn AI", 
    layout="wide", 
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# ========================================
# PERFECT CSS - DARK MODE TEXT FIXED + FULL PAGE
# ========================================
st.markdown("""
<style>
    /* ========================================
       #13293D PERFECT THEME - DARK MODE TEXT PERFECTLY FIXED
    ======================================== */
    
    /* FULL PAGE - NO MARGINS, MAX WIDTH */
    .stApp {
        background: linear-gradient(135deg, #E8F1F2 0%, #F0F9FF 100%);
        max-width: 100vw !important;
        margin: 0 !important;
        padding: 0.5rem !important;
    }
    
    /* REMOVE ALL CENTER ALIGNMENT - FULL WIDTH */
    .main > div {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: none !important;
    }
    
    /* Sidebar Navy Blue */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13293D 0%, #1E3A8A 100%) !important;
        padding: 1rem 1rem !important;
    }
    
    /* SIDEBAR TEXT - Always WHITE */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }
    
    /* DROPDOWN - Perfect contrast */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.95) !important;
        border: 2px solid #60A5FA !important;
        border-radius: 10px !important;
        margin: 0.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] div {
        color: #13293D !important;
        font-weight: 600 !important;
    }
    
    /* 🎯 METRICS - Light Mode (White cards, Navy text) */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 16px !important;
        border: 2px solid #E2E8F0 !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        height: 130px !important;
        padding: 1rem !important;
    }
    
    [data-testid="metric-container"] .stMetricLabel {
        color: #1F2937 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    [data-testid="metric-container"] .stMetricValue {
        color: #13293D !important;
        font-size: 32px !important;
        font-weight: 800 !important;
    }
    
    /* 🎯 DARK MODE FIX - ALL TEXT BLACK + PERFECT VISIBILITY */
    [data-testid="theme-root"][data-testid="theme-dark"],
    [data-testid="theme-root"].dark-theme {
        color: #1F2937 !important;
    }
    
    /* DARK MODE - FORCE ALL TEXT BLACK */
    [data-testid="theme-root"][data-testid="theme-dark"] *,
    [data-testid="theme-root"].dark-theme * {
        color: #1F2937 !important;
    }
    
    /* DARK MODE - Headers, Subheaders, Paragraphs */
    [data-testid="theme-root"][data-testid="theme-dark"] h1,
    [data-testid="theme-root"][data-testid="theme-dark"] h2,
    [data-testid="theme-root"][data-testid="theme-dark"] h3,
    [data-testid="theme-root"][data-testid="theme-dark"] p,
    [data-testid="theme-root"][data-testid="theme-dark"] div,
    [data-testid="theme-root"][data-testid="theme-dark"] span,
    [data-testid="theme-root"][data-testid="theme-dark"] .stMarkdown {
        color: #1F2937 !important;
    }
    
    /* DARK MODE - INFO/SUCCESS/WARNING/ERROR BOXES */
    [data-testid="theme-root"][data-testid="theme-dark"] div[role="alert"],
    [data-testid="theme-root"][data-testid="theme-dark"] .stAlert,
    [data-testid="theme-root"][data-testid="theme-dark"] div.element-container {
        background: rgba(248, 250, 252, 0.98) !important;
        color: #1F2937 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
    }
    
    /* DARK MODE - METRICS (White text on dark cards) */
    [data-testid="theme-root"][data-testid="theme-dark"] [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.95) !important;
        border: 2px solid #475569 !important;
    }
    
    [data-testid="theme-root"][data-testid="theme-dark"] [data-testid="metric-container"] .stMetricLabel,
    [data-testid="theme-root"][data-testid="theme-dark"] [data-testid="metric-container"] .stMetricValue {
        color: #F8FAFC !important;
    }
    
    /* BUTTONS - Shiny Navy */
    .stButton > button {
        background: linear-gradient(135deg, #13293D 0%, #1E40AF 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 3.5em !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        box-shadow: 0 8px 25px rgba(19,41,61,0.4) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
        box-shadow: 0 12px 35px rgba(59,130,246,0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    /* TITLE - Gradient */
    h1 {
        background: linear-gradient(135deg, #13293D, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3rem !important;
        text-align: left !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #60A5FA !important;
        border-radius: 16px !important;
        background: rgba(96,165,250,0.08) !important;
        padding: 2rem !important;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------- 
# GLOBAL CONSTANTS
# ---------------------------------------------------------
TARGET_CLASSES = ['No Churn', 'Churn'] 

if 'test_data' not in st.session_state:
    st.session_state['test_data'] = None
if 'test_targets' not in st.session_state:
    st.session_state['test_targets'] = None

# --------------------------------------------------------- 
# UTILITY FUNCTIONS
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        return preprocessor
    except FileNotFoundError:
        st.error("⚠️ 'preprocessor.pkl' not found. Please run the training notebook first.")
        return None

@st.cache_data
def load_sample_data_from_github():
    """Load test_data_combined.csv from GitHub"""
    GITHUB_RAW_URL = "https://github.com/VashisthaMeenakshi15/Telecom-Churn/blob/main/test_data_combined.csv"
    try:
        return pd.read_csv(GITHUB_RAW_URL)
    except:
        return None
        
def get_trained_model(selection):
    model_map = {
        "Logistic Regression": "log
