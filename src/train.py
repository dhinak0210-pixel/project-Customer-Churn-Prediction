import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score

# Relative imports from current directory or ROOT
try:
    from src.preprocessing import clean_data, encode_target
except ModuleNotFoundError:
    from preprocessing import clean_data, encode_target

# Configuration (Relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data/Telco-Customer-Churn.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

class ChurnTrainer:
    """
    Handles the end-to-end model development lifecycle:
    - Preprocessing
    - Pipeline Construction
    - RandomizedSearchCV Tuning
    - Evaluation & XAI Capture
    """
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.raw_df = pd.read_csv(self.data_path)
        self.df = None
        self.y = None
        self.X = None
        self.pipeline = None

    def prepare(self):
        """Clean and split data for modeling."""
        print("Cleaning and preparing dataset...")
        self.df = clean_data(self.raw_df)
        self.y = encode_target(self.raw_df)
        self.X = self.df.drop(columns=['Churn'], errors='ignore')
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

    def build_pipeline(self):
        """Construct the ML pipeline (Scaling -> Encoding -> XGBoost)."""
        cat_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        num_cols = self.X.select_dtypes(exclude=['object']).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ])
        
        # Handle Imbalance
        neg_count = (self.y == 0).sum()
        pos_count = (self.y == 1).sum()
        scale_weight = neg_count / pos_count
        
        model = xgb.XGBClassifier(
            random_state=42, use_label_encoder=False, 
            eval_metric='logloss', scale_pos_weight=scale_weight
        )
        
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)]), cat_cols, num_cols

    def train_and_tune(self, X_train, y_train):
        """Execute RandomizedSearchCV to find optimal hyperparameters."""
        print("Building and Tuning model...")
        pipeline, cat_cols, num_cols = self.build_pipeline()
        
        params = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'classifier__gamma': [0, 0.1, 0.5, 1]
        }
        
        search = RandomizedSearchCV(
            pipeline, param_distributions=params, n_iter=15, 
            scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, random_state=42
        )
        
        search.fit(X_train, y_train)
        self.pipeline = search.best_estimator_
        print(f"Best Params: {search.best_params_}")
        return self.pipeline

    def export_artifacts(self, X_test, y_test, cat_cols, num_cols):
        """Evaluate result and export artifacts to models/."""
        print("Evaluating result and exporting artifacts...")
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred = self.pipeline.predict(X_test)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
        # SHAP Artifact
        cat_feature_names = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(cat_feature_names)
        X_test_transformed = self.pipeline.named_steps['preprocessor'].transform(X_test)
        explainer = shap.Explainer(self.pipeline.named_steps['classifier'], X_test_transformed, feature_names=feature_names)
        
        # Save
        joblib.dump(self.pipeline, os.path.join(MODEL_DIR, 'churn_model_pipeline.joblib'))
        joblib.dump(explainer, os.path.join(MODEL_DIR, 'shap_explainer.joblib'))
        joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.joblib'))
        joblib.dump(metrics, os.path.join(MODEL_DIR, 'metrics.joblib'))
        print(f"Deployment artifacts saved in {MODEL_DIR}")

if __name__ == "__main__":
    trainer = ChurnTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare()
    trainer.train_and_tune(X_train, y_train)
    # Get cat/num columns again for names
    cat_cols = trainer.X.select_dtypes(include=['object']).columns.tolist()
    num_cols = trainer.X.select_dtypes(exclude=['object']).columns.tolist()
    trainer.export_artifacts(X_test, y_test, cat_cols, num_cols)
