# 🛡️ ChurnAI: Enterprise Customer Retention Platform

A production-ready Customer Churn forecasting system with **Explainable AI (XAI)**. This platform uses XGBoost for predictions and SHAP for providing deep, actionable insights into customer behavior.

## 🚀 Cloud Deployment (Streamlit Cloud)
1. Fork this repository.
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io/).
3. Choose `app/streamlit_app.py` as the main file.
4. Add `PYTHONPATH=.` to your environment variables if necessary (though the app handles path injection).

## 📁 Local Setup & Development

### 1. Environment Configuration
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Model Training & Optimization
If you wish to retrain the model on the latest Telco data:
```bash
python3 src/train.py
```
This will automatically:
- Perform 5-fold Stratified Cross-Validation.
- Run Randomized Hyperparameter Search.
- Export optimized serialized artifacts to `/models`.

### 3. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

## 🛠️ Tech Stack
- **XGBoost**: Gradient Boosted Decision Trees for best-in-class tabular accuracy.
- **SHAP**: Game-theoretic approach to model explainability (Waterfall, Beeswarm, Decision Paths).
- **Streamlit**: High-fidelity reactive UI for data applications.
- **Scikit-Learn**: Robust preprocessing and cross-validation pipelines.

## 📈 Performance Summary
The current model handles class imbalance using `scale_pos_weight`, achieving:
- **Recall (Churners)**: ~80%
- **ROC-AUC**: ~0.85

---
*Developed by Antigravity - Senior ML & Frontend Engineer*
