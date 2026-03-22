import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Paths
DATA_PATH = '/home/dhina/Customer Churn Analysis/data/churn_data.csv'
MODEL_DIR = '/home/dhina/Customer Churn Analysis/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_churn_model():
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    # 2. Pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    clf.fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # 5. Explainer (SHAP)
    # Get preprocessed data to explain model
    X_test_processed = clf.named_steps['preprocessor'].transform(X_test)
    # SHAP expects dense arrays if data was transformed to sparse
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()
    
    # Get feature names after one-hot encoding
    cat_feature_names = clf.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)
    
    explainer = shap.Explainer(clf.named_steps['classifier'], X_test_processed, feature_names=feature_names)
    
    # 6. Save
    with open(os.path.join(MODEL_DIR, 'model_pipeline.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    
    with open(os.path.join(MODEL_DIR, 'explainer.pkl'), 'wb') as f:
        pickle.dump(explainer, f)
    
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    print("Model and Explainer saved successfully.")

if __name__ == "__main__":
    train_churn_model()
