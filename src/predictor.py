import os
import joblib
import pandas as pd
import shap

# Settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model_pipeline.joblib')
EXPLAINER_PATH = os.path.join(MODEL_DIR, 'shap_explainer.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.joblib')

class ChurnPredictor:
    """
    Handles model loading, inference, and SHAP explanations.
    Provides a standardized interface for both CLI and UI.
    """
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self._load_assets()

    def _load_assets(self):
        """Loads serialized model and XAI artifacts from /models."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.explainer = joblib.load(EXPLAINER_PATH)
            self.feature_names = joblib.load(FEATURES_PATH)
        else:
            raise FileNotFoundError(f"Missing model artifacts in {MODEL_DIR}")

    def predict(self, df):
        """Returns binary predictions and probability scores."""
        probs = self.model.predict_proba(df)[:, 1]
        preds = self.model.predict(df)
        return preds, probs

    def explain(self, df):
        """Generates SHAP values for a given input dataframe."""
        # 1. Transform data using the pipeline preprocessor
        X_tx = self.model.named_steps['preprocessor'].transform(df)
        # 2. Get explanations
        sh_vals = self.explainer(X_tx)
        
        # Ensure feature names are intact (Joblib sometimes drops them)
        if not hasattr(sh_vals, 'feature_names') or sh_vals.feature_names is None:
            sh_vals.feature_names = self.feature_names
            
        return sh_vals
