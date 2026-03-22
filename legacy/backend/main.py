import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Paths
MODEL_PATH = '/home/dhina/Customer Churn Analysis/models/model_pipeline.pkl'
EXPLAINER_PATH = '/home/dhina/Customer Churn Analysis/models/explainer.pkl'
FEATURES_PATH = '/home/dhina/Customer Churn Analysis/models/feature_names.pkl'

app = FastAPI(title="Customer Churn XAI API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models
model = None
explainer = None
feature_names = None

@app.on_event("startup")
def load_assets():
    global model, explainer, feature_names
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(EXPLAINER_PATH, 'rb') as f:
        explainer = pickle.load(f)
        
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)

# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float

@app.get("/")
def read_root():
    return {"message": "Customer Churn XAI API is running."}

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    # Convert incoming features to DataFrame for pipeline preprocessing
    sample = pd.DataFrame([features.model_dump()])
    
    # Check probability
    prob = model.predict_proba(sample)[0][1]
    prediction = int(model.predict(sample)[0])
    
    # 2. Local Explanation
    # Need to preprocess sample first as model was trained in pipeline
    # Transformation expects a single sample
    preprocessor = model.named_steps['preprocessor']
    X_processed = preprocessor.transform(sample)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    
    # Explain prediction
    shap_vals = explainer(X_processed)
    
    # For a single sample, we take the 0th element
    explanation = {
        "churn_probability": float(prob),
        "prediction": "Yes" if prediction == 1 else "No",
        "base_value": float(shap_vals.base_values[0]),
        "shap_values": [float(v) for v in shap_vals.values[0]],
        "feature_names": feature_names
    }
    
    return explanation

@app.get("/global-importance")
def get_global_importance():
    # Summarize global importance by taking mean of absolute SHAP values for 100 random samples
    # For now, let's return a simple version or a placeholder if expensive
    # In a real app we'd cache this
    return {"message": "Global importance endpoint ready. Please call with samples if needed."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
