import pandas as pd
import numpy as np

def clean_data(df):
    """
    Standardize the Telco Customer Churn dataset.
    - Handles case-insensitive customerID removal.
    - Converts TotalCharges to numeric and imputes missing values.
    """
    df_clean = df.copy()
    
    # Handle TotalCharges strings (common in Telco dataset)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # Fill with median
        median_val = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val if not pd.isna(median_val) else 0)
    
    # Drop customerID (case-insensitive check)
    id_cols = [c for c in df_clean.columns if c.lower() == 'customerid']
    if id_cols:
        df_clean = df_clean.drop(columns=id_cols)
        
    return df_clean

def encode_target(df, target_col='Churn'):
    """
    Encode binary target variable ('Yes'/'No') to (1/0).
    """
    if target_col in df.columns:
        return df[target_col].apply(lambda x: 1 if x == 'Yes' else 0)
    return None
