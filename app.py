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

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# st.set_page_config(page_title="Telco Churn AI", layout="wide", page_icon="📡")

# st.markdown("""
#     <style>
#     .main { background-color: #F8F9FA; font-family: 'Segoe UI', sans-serif; }
#     .stButton>button { 
#         width: 100%; border-radius: 6px; height: 3em; 
#         background-color: #007BFF; color: white; font-weight: 600; 
#     }
#     .stButton>button:hover { background-color: #0056b3; }
#     </style>
#     """, unsafe_allow_html=True)
# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
# st.set_page_config(page_title="Telco Churn AI", layout="wide", page_icon="📡")

# # CHANGE BACKGROUND COLOR HERE
# st.markdown("""
# <style>
#     /* Main Background */
#     .stApp {
#         background-color: #E8F1F2;
#     }
 
#     /* Sidebar Background */
#     [data-testid="stSidebar"] {
#         background-color: #13293D;
#     }
 
#     /* TEXT COLOR FIX: Forces all sidebar text to be white with NO background */
#     [data-testid="stSidebar"] * {
#         color: white !important;
#         background-color: transparent !important; /* Removes the pink block */
#     }
 
#     /* DROPDOWN FIX: Reset the input box to be white with black text */
#     /* This targets the specific box where "Logistic Regression" is written */
#     [data-testid="stSidebar"] div[data-baseweb="select"] > div,
#     [data-testid="stSidebar"] div[data-baseweb="select"] span {
#         color: black !important;
#         background-color: white !important;
#     }
 
#     /* Dropdown Arrow Icon */
#     [data-testid="stSidebar"] svg {
#         fill: black !important;
#     }
 
#     /* Button Styling */
#     .stButton>button { 
#         width: 100%; border-radius: 6px; height: 3em; 
#         background-color: #000000; color: white; font-weight: 600; 
#         border: none;
#     }
# </style>
#     """, unsafe_allow_html=True)
st.markdown("""
<style>
    /* ========================================
       THEME-AWARE CUSTOMIZATION (#13293D Navy Blue)
       Works PERFECTLY in Light + Dark mode!
    ======================================== */
    
    /* Main App Background - Light/Dark aware */
    .stApp {
        background: linear-gradient(135deg, #E8F1F2 0%, #F8FAFC 100%);
    }
    
    /* Sidebar - Navy Blue (#13293D) */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13293D 0%, #1E3A5F 100%);
    }
    
    /* ========================================
       LIGHT THEME (Default)
    ======================================== */
    /* Light theme sidebar text */
    [data-testid="stSidebar"] section[data-testid="stSidebar"] > div > div > div {
        color: white !important;
    }
    
    /* Light theme dropdown fix */
    [data-testid="stSidebar"] div[role="combobox"] {
        background-color: white;
        color: #13293D !important;
        border: 2px solid #13293D;
        border-radius: 8px;
    }
    
    [data-testid="stSidebar"] div[role="combobox"]::placeholder {
        color: #64748B;
    }
    
    /* ========================================
       DARK THEME (Auto-detects)
    ======================================== */
    /* Dark theme sidebar text stays white */
    [data-testid="theme-root"] [data-testid="theme-dark"] [data-testid="stSidebar"] * {
        color: white !important;
        background-color: transparent !important;
    }
    
    /* Dark theme dropdown (white bg, navy text) */
    [data-testid="theme-root"] [data-testid="theme-dark"] [data-testid="stSidebar"] div[role="combobox"] {
        background-color: #1E293B !important;
        color: white !important;
        border: 2px solid #60A5FA;
    }
    
    /* Button - Navy Blue (#13293D) - Both themes */
    .stButton > button {
        background: linear-gradient(135deg, #13293D 0%, #1E40AF 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 3.2em !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 12px rgba(19, 41, 61, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Button Hover (Gold accent) */
    .stButton > button:hover {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%) !important;
        box-shadow: 0 6px 20px rgba(19, 41, 61, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, white 0%, #F8FAFC 100%);
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Dark theme metrics */
    [data-testid="theme-root"] [data-testid="theme-dark"] [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border: 1px solid #475569;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Title styling */
    h1 {
        color: #13293D !important;
        font-family: 'Segoe UI', -apple-system, sans-serif !important;
        text-shadow: 0 2px 4px rgba(19,41,61,0.1);
    }
</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# 2. GLOBAL CONSTANTS
# ---------------------------------------------------------
TARGET_CLASSES = ['No Churn', 'Churn'] 

if 'test_data' not in st.session_state:
    st.session_state['test_data'] = None
if 'test_targets' not in st.session_state:
    st.session_state['test_targets'] = None

# ---------------------------------------------------------
# 3. UTILITY FUNCTIONS
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        return preprocessor
    except FileNotFoundError:
        st.error("⚠️ 'preprocessor.pkl' not found. Please run the training notebook first.")
        return None

def get_trained_model(selection):
    model_map = {
        "Logistic Regression": "log_reg.pkl",
        "Decision Tree": "dt_clf.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Naive Bayes (Gaussian)": "gnb.pkl",
        "Random Forest": "rf_clf.pkl",
        "XGBoost": "xgb.pkl"
    }
    f_path = os.path.join('saved_models', model_map.get(selection, ""))
    if os.path.exists(f_path):
        return joblib.load(f_path)
    return None

def compute_metrics(clf, x_data, y_data):
    # Ensure y_data contains no NaNs before calculating
    if np.isnan(y_data).any():
        return None
        
    preds = clf.predict(x_data)
    auc_score = 0.0
    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x_data)[:, 1]
            auc_score = roc_auc_score(y_data, probs)
    except:
        pass
        
    return {
        "Accuracy": accuracy_score(y_data, preds),
        "AUC": auc_score,
        "Precision": precision_score(y_data, preds, zero_division=0),
        "Recall": recall_score(y_data, preds, zero_division=0),
        "F1": f1_score(y_data, preds, zero_division=0),
        "MCC": matthews_corrcoef(y_data, preds),
        "predictions": preds
    }

# ---------------------------------------------------------
# 4. SIDEBAR SETUP
# ---------------------------------------------------------
st.sidebar.title("📡 Menu")
app_mode = st.sidebar.radio("Navigate:", ["Batch Prediction Tool", "Model Insights"])

if app_mode == "Batch Prediction Tool":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    selected_algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors", 
         "Naive Bayes (Gaussian)", "Random Forest", "XGBoost")
    )

# ---------------------------------------------------------
# 5. PAGE: PREDICTION TOOL
# ---------------------------------------------------------
if app_mode == "Batch Prediction Tool":
    st.title("📡 Telco Customer Churn Prediction")
    st.markdown("Upload a CSV of customers to identify who is at risk of leaving.")
    st.markdown("---")

    left_col, right_col = st.columns([2, 4])

    with left_col:
        st.subheader("1. Upload Data")
        st.info("Upload a CSV file. It must contain the standard Telco columns.")
        
        data_file = st.file_uploader("Drop CSV Here", type=["csv"])

        if data_file:
            try:
                # Read CSV
                raw_df = pd.read_csv(data_file)
                
                # Cleanup: Force TotalCharges to numeric
                if 'TotalCharges' in raw_df.columns:
                    raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce').fillna(0)

                # --- FIX: ROBUST TARGET MAPPING ---
                if 'Churn' in raw_df.columns:
                    # 1. Convert to string, strip whitespace, make lowercase
                    clean_churn = raw_df['Churn'].astype(str).str.strip().str.lower()
                    
                    # 2. Map 'yes'->1, 'no'->0. 
                    # 3. Fill any failures (NaN) with 0 so the app doesn't crash
                    st.session_state['test_targets'] = clean_churn.map({'yes': 1, 'no': 0}).fillna(0).values
                    
                    features_df = raw_df.drop(columns=['Churn', 'customerID'], errors='ignore')
                else:
                    st.session_state['test_targets'] = None
                    features_df = raw_df.drop(columns=['customerID'], errors='ignore')
                
                # Save features for display
                st.session_state['display_data'] = features_df.copy()
                
                # PREPROCESSING
                preprocessor = load_assets()
                if preprocessor:
                    st.session_state['test_data'] = preprocessor.transform(features_df)
                    st.success(f"✅ Loaded & Processed {len(raw_df)} customers.")
                
            except Exception as err:
                st.error(f"Error processing file: {err}")

    with right_col:
        if st.session_state['test_data'] is not None:
            st.subheader("2. Prediction Results")
            active_model = get_trained_model(selected_algorithm)
            
            if active_model:
                X_in = st.session_state['test_data']
                y_true = st.session_state['test_targets']
                
                try:
                    # Generate Predictions
                    pred_indices = active_model.predict(X_in)
                    pred_names = [TARGET_CLASSES[i] for i in pred_indices]
                    
                    # Display Data Table with Predictions
                    display_df = st.session_state['display_data'].copy()
                    display_df.insert(0, "⚠️ Risk Prediction", pred_names)
                    
                    def highlight_churn(val):
                        return 'background-color: #ffcccc' if val == 'Churn' else ''
                    
                    st.dataframe(display_df.style.applymap(highlight_churn, subset=['⚠️ Risk Prediction']), height=300)
                    
                    # --- METRICS SECTION ---
                    if y_true is not None:
                        st.markdown("### Performance Metrics")
                        scores = compute_metrics(active_model, X_in, y_true)
                        
                        if scores:
                            # Metrics Row
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("Accuracy", f"{scores['Accuracy']:.1%}")
                            c2.metric("AUC", f"{scores['AUC']:.3f}")
                            c3.metric("F1-Score", f"{scores['F1']:.3f}")
                            c4.metric("Precision", f"{scores['Precision']:.3f}")
                            c5.metric("Recall", f"{scores['Recall']:.3f}")
                            
                            # Confusion Matrix
                            col_cm1, col_cm2 = st.columns([1, 2])
                            with col_cm1:
                                 st.write("#### Confusion Matrix")
                                 fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                                 cm_data = confusion_matrix(y_true, scores['predictions'])
                                 sns.heatmap(
                                    cm_data, annot=True, fmt='d', cmap='Reds', 
                                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes']
                                 )
                                 plt.xlabel('Predicted')
                                 plt.ylabel('Actual')
                                 st.pyplot(fig_cm)
                        else:
                            st.warning("⚠️ Could not calculate metrics. Target column contained errors.")
                            
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            else:
                st.error(f"Model file for '{selected_algorithm}' not found.")

# ---------------------------------------------------------
# 6. PAGE: INSIGHTS
# ---------------------------------------------------------
else:
    st.title("📊 Model Benchmark & Insights")
    CSV_PATH = 'model_performance.csv'
    if os.path.exists(CSV_PATH):
        df_results = pd.read_csv(CSV_PATH)
        st.dataframe(df_results.sort_values(by="F1-Score", ascending=False), use_container_width=True)
    else:
        st.warning("⚠️ Benchmark file 'model_performance.csv' not found.")
