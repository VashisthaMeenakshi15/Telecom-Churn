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
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/VashisthaMeenakshi15/Telecom-Churn/main/test_data_combined.csv"
    try:
        return pd.read_csv(GITHUB_RAW_URL)
    except:
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
# SIDEBAR SETUP
# ---------------------------------------------------------
st.sidebar.title("📡 Menu")
app_mode = st.sidebar.radio("Navigate:", ["Batch Prediction Tool", "Model Insights"])

if app_mode == "Batch Prediction Tool":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Selection")
    selected_algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors", 
         "Naive Bayes (Gaussian)", "Random Forest", "XGBoost")
    )

# --------------------------------------------------------- 
# PAGE: PREDICTION TOOL
# ---------------------------------------------------------
if app_mode == "Batch Prediction Tool":
    # FULL WIDTH TITLE
    st.title("📡 Telco Customer Churn Prediction")
    st.markdown("**Upload customer data to predict churn risk with 6 ML models**")
    st.markdown("---")

    left_col, right_col = st.columns([2, 4])

    with left_col:
        st.subheader("1. 📤 Upload Data")
        st.info("📋 CSV must contain standard Telco columns (Churn optional)")

        # Download Sample Data from GitHub
        # 🎯 STYLISH DOWNLOAD SECTION - Perfect placement + colors
        st.markdown("---")
        col_sample1, col_sample2 = st.columns([3, 1])
        with col_sample1:
            st.markdown("**💾 Test with sample data**")
        with col_sample2:
            # Note: This empty button seems redundant if using download_button below, 
            # but kept for UI consistency if intended as a label
            if st.button("📥 Download", 
                         help="Download test_data_combined.csv from GitHub",
                         use_container_width=True):
                pass 
                
        try:
            df_sample = load_sample_data_from_github()
            if df_sample is not None:
                csv_sample = df_sample.to_csv(index=False).encode('utf-8')
                    
                st.download_button(
                    label=f"📥 test_data_combined.csv ({len(df_sample):,} rows)",
                    data=csv_sample,
                    file_name="test_data_combined.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="secondary"
                )
                st.caption("*Powered by your GitHub repo*")
            else:
                st.error("❌ Cannot fetch from GitHub")
                st.markdown("[🔗 View on GitHub](https://github.com/VashisthaMeenakshi15/Telecom-Churn/blob/main/test_data_combined.csv)")
        except:
            st.error("❌ Network error")
            st.markdown("[🔗 GitHub Link](https://github.com/VashisthaMeenakshi15/Telecom-Churn/blob/main/test_data_combined.csv)")

        data_file = st.file_uploader("Drop CSV Here", type=["csv"])
        
        if data_file:
            try:
                raw_df = pd.read_csv(data_file)
                
                if 'TotalCharges' in raw_df.columns:
                    raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce').fillna(0)

                if 'Churn' in raw_df.columns:
                    clean_churn = raw_df['Churn'].astype(str).str.strip().str.lower()
                    st.session_state['test_targets'] = clean_churn.map({'yes': 1, 'no': 0}).fillna(0).values
                    features_df = raw_df.drop(columns=['Churn', 'customerID'], errors='ignore')
                else:
                    st.session_state['test_targets'] = None
                    features_df = raw_df.drop(columns=['customerID'], errors='ignore')
                
                st.session_state['display_data'] = features_df.copy()
                
                preprocessor = load_assets()
                if preprocessor:
                    st.session_state['test_data'] = preprocessor.transform(features_df)
                    st.success(f"✅ Processed {len(raw_df)} customers")
                
            except Exception as err:
                st.error(f"❌ Error: {err}")

    with right_col:
        if st.session_state['test_data'] is not None:
            st.subheader("2. 🎯 Prediction Results")
            active_model = get_trained_model(selected_algorithm)
            
            if active_model:
                X_in = st.session_state['test_data']
                y_true = st.session_state['test_targets']
                
                try:
                    pred_indices = active_model.predict(X_in)
                    pred_names = [TARGET_CLASSES[i] for i in pred_indices]
                    
                    display_df = st.session_state['display_data'].copy()
                    display_df.insert(0, "⚠️ Risk Prediction", pred_names)
                    
                    def highlight_churn(val):
                        return 'background-color: #ffcccc' if val == 'Churn' else ''
                    
                    st.dataframe(
                        display_df.style.applymap(highlight_churn, subset=['⚠️ Risk Prediction']), 
                        height=350, 
                        use_container_width=True
                    )
                    
                    if y_true is not None:
                        st.markdown("### 📊 Performance Metrics")
                        scores = compute_metrics(active_model, X_in, y_true)
                        
                        if scores:
                            # 6 PERFECT METRICS
                            col1, col2, col3, col4, col5, col6 = st.columns(6)
                            col1.metric("Accuracy", f"{scores['Accuracy']:.1%}")
                            col2.metric("AUC", f"{scores['AUC']:.3f}")
                            col3.metric("F1 Score", f"{scores['F1']:.3f}")
                            col4.metric("Precision", f"{scores['Precision']:.3f}")
                            col5.metric("Recall", f"{scores['Recall']:.3f}")
                            col6.metric("MCC", f"{scores['MCC']:.3f}")
                            
                            # 🎯 HUGE CONFUSION MATRIX (12x10)
                            st.markdown("### 📈 Confusion Matrix")
                            cm_data = confusion_matrix(y_true, scores['predictions'])
                            
                            fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
                            sns.set(font_scale=1.6)
                            sns.heatmap(
                                cm_data, 
                                annot=True, 
                                fmt='d', 
                                cmap='Blues',
                                xticklabels=['No Churn', 'Churn'],
                                yticklabels=['No Churn', 'Churn'],
                                ax=ax_cm,
                                cbar_kws={'shrink': 0.75},
                                annot_kws={'size': 24, 'weight': 'bold', 'color': 'white'}
                            )
                            ax_cm.set_xlabel("Predicted", fontsize=22, fontweight='bold', color='#13293D')
                            ax_cm.set_ylabel("Actual", fontsize=22, fontweight='bold', color='#13293D')
                            ax_cm.set_title("Telco Churn Confusion Matrix", 
                                            fontsize=26, fontweight='bold', pad=25, color='#13293D')
                            plt.tight_layout()
                            st.pyplot(fig_cm)
                            sns.set(font_scale=1.0)
                            
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
            else:
                st.error(f"❌ Model '{selected_algorithm}' not found in saved_models/")

# --------------------------------------------------------- 
# PAGE: INSIGHTS
# ---------------------------------------------------------
else:
    st.title("📊 Model Benchmark & Insights")
    
    CSV_PATH = 'model_performance.csv'
    if os.path.exists(CSV_PATH):
        df_results = pd.read_csv(CSV_PATH)
        
        # 1. Prepare Metrics Table (Sorted)
        df_display = df_results.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
        
        st.subheader("1. 🏆 Performance Metrics")
        st.dataframe(
            df_display, 
            use_container_width=True,
            hide_index=True,               # <--- FIX: Hides the 0,1,2 index column
            height=(len(df_display) + 1) * 35 + 3  # <--- FIX: Removes extra empty white rows
        )
        
        # 2. Add Key Insights Table
        st.subheader("2. 🧠 Model Observations")
        
        # Defining insights based on your screenshot metrics
        insights_data = [
            {"Algorithm": "Logistic Regression", "Observation": "🏆 Best Performer: Highest Accuracy (82%) and AUC (0.86). Excellent baseline."},
            {"Algorithm": "Gaussian NB",         "Observation": "⚠️ High Sensitivity: Top Recall (89%) captures most churners, but high False Alarms (Low Precision)."},
            {"Algorithm": "XGBoost",             "Observation": "⚖️ Balanced: Strong trade-off between Precision and Recall. Robust for complex data."},
            {"Algorithm": "Random Forest",       "Observation": "🌲 Stable: Performance is very close to XGBoost. Good handling of non-linear features."},
            {"Algorithm": "K-Nearest Neighbors", "Observation": "📍 Moderate: Decent accuracy (77%), but struggled to distinguish boundary cases."},
            {"Algorithm": "Decision Tree",       "Observation": "📉 Overfitting: Lowest AUC (0.64) and Accuracy. The model is likely too complex (needs pruning)."}
        ]
        
        df_insights = pd.DataFrame(insights_data)
        
        # Display as a static table (cleaner for text)
        st.table(df_insights)

    else:
        st.warning("⚠️ 'model_performance.csv' not found. Please run the training notebook to generate this file.")
