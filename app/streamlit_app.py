import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
import joblib

# --- CONFIGURATION & PATHS ---
# Deployment-safe path logic
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model_pipeline.joblib')
EXPLAINER_PATH = os.path.join(MODEL_DIR, 'shap_explainer.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'data/Telco-Customer-Churn.csv')

from src.preprocessing import clean_data
from src.predictor import ChurnPredictor

# --- UI CONFIG ---
st.set_page_config(
    page_title="ChurnAI | Enterprise XAI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #0c111d; border-right: 1px solid #1e293b; }
    .metric-card { background: rgba(30, 41, 59, 0.7); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; }
    .risk-score { font-size: 3rem; font-weight: 900; color: #f43f5e; text-shadow: 0 0 20px rgba(244, 63, 94, 0.4); }
    .warning-score { font-size: 3rem; font-weight: 900; color: #f59e0b; text-shadow: 0 0 20px rgba(245, 158, 11, 0.4); }
    .safe-score { font-size: 3rem; font-weight: 900; color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
    .stButton>button { background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); color: white; border-radius: 10px; font-weight: 700; border: none; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    """Initializes and caches the predictive core."""
    return ChurnPredictor()

def main():
    try:
        predictor = get_predictor()
        metrics = joblib.load(os.path.join(ROOT_DIR, 'models/metrics.joblib'))
    except Exception as e:
        st.error(f"Failed to initialize AI Engine: {e}. Check /models artifacts.")
        return

    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
        st.title("ChurnAI")
        st.markdown("---")
        page = st.radio("Navigation", ["Overview", "Single Analysis", "Batch Processing", "XAI deep-dive"])
        st.info("System Ready | Optimized XGBoost")

    # --- ROUTING ---
    if page == "Overview":
        st.title("📊 Intelligence Overview")
        st.write("Model performance benchmarks and health stats.")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
        m2.metric("Recall", f"{metrics['Recall']:.1%}")
        m3.metric("Precision", f"{metrics['Precision']:.3f}")
        m4.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
        
    elif page == "Single Analysis":
        st.title("👤 Individual Risk Forecast")
        with st.form("input_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", [0, 1])
                tenure = st.slider("Tenure (Months)", 1, 72, 24)
            with col2:
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            with col3:
                monthly = st.number_input("Monthly Charges ($)", 18.25, 118.75, 70.0)
                total = st.number_input("Total Charges ($)", 18.0, 9000.0, 2000.0)
            
            submit = st.form_submit_button("⚡ Run Forecase")
            
        if submit:
            raw_data = pd.read_csv(os.path.join(ROOT_DIR, 'data/Telco-Customer-Churn.csv'))
            X_cols = raw_data.drop(['customerID', 'Churn'], axis=1, errors='ignore').columns.tolist()
            
            sample = pd.DataFrame([{
                'gender': gender, 'SeniorCitizen': int(senior), 'Partner': "Yes", 'Dependents': "No",
                'tenure': int(tenure), 'PhoneService': "Yes", 'MultipleLines': "No", 'InternetService': internet,
                'OnlineSecurity': "No", 'OnlineBackup': "No", 'DeviceProtection': 'No', 'TechSupport': 'No',
                'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': "No",
                'PaymentMethod': payment, 'MonthlyCharges': float(monthly), 'TotalCharges': float(total)
            }])[X_cols]
            
            _, prob_arr = predictor.predict(sample)
            prob = prob_arr[0]
            
            # Risk Display
            if prob > 0.65: risk_lvl, risk_cls = "🔴 High", "risk-score"
            elif prob > 0.35: risk_lvl, risk_cls = "🟡 Medium", "warning-score"
            else: risk_lvl, risk_cls = "🟢 Low", "safe-score"
            
            st.markdown(f"#### Predicted Risk: <span class='{risk_cls}'>{prob:.1%}</span> ({risk_lvl})", unsafe_allow_html=True)
            
            # Local XAI
            st.divider()
            st.subheader("💡 Why this prediction?")
            sh_v = predictor.explain(sample)
            
            c1, c2 = st.columns(2)
            with c1:
                fig_w, _ = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(sh_v[0], max_display=10, show=False)
                st.pyplot(fig_w)
            with c2:
                fig_d, _ = plt.subplots(figsize=(8, 5))
                shap.plots.decision(sh_v.base_values[0], sh_v.values[0], feature_names=predictor.feature_names, max_display=10, show=False)
                st.pyplot(fig_d)

    elif page == "Batch Processing":
        st.title("📂 Enterprise Batch Pipeline")
        upload = st.file_uploader("Upload CSV", type="csv")
        if upload:
            df_up = pd.read_csv(upload)
            if st.button("🚀 Process Batch"):
                raw_data = pd.read_csv(os.path.join(ROOT_DIR, 'data/Telco-Customer-Churn.csv'))
                X_cols = raw_data.drop(['customerID', 'Churn'], axis=1, errors='ignore').columns.tolist()
                
                df_clean = clean_data(df_up)
                batch_in = df_clean[X_cols]
                
                _, probs = predictor.predict(batch_in)
                df_up['Risk_Prob'] = probs
                df_up['Risk_Level'] = df_up['Risk_Prob'].apply(lambda x: 'High' if x > 0.65 else ('Medium' if x > 0.35 else 'Low'))
                
                st.dataframe(df_up[['Risk_Prob', 'Risk_Level']].style.background_gradient(subset=['Risk_Prob'], cmap='YlOrRd'))
                st.download_button("📥 Result CSV", df_up.to_csv(index=False), "results.csv")

    elif page == "XAI deep-dive":
        st.title("🔎 Model Feature Repository")
        raw_data = pd.read_csv(os.path.join(ROOT_DIR, 'data/Telco-Customer-Churn.csv'))
        df_g = clean_data(raw_data).drop('Churn', axis=1, errors='ignore').head(100)
        
        sh_g = predictor.explain(df_g)
        fig_b, _ = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(sh_g, max_display=15, show=False)
        st.pyplot(fig_b)

if __name__ == "__main__":
    main()
