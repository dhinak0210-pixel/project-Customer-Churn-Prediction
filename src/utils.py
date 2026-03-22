import pandas as pd

def clean_telco_data(df):
    """Standardized preprocessing for Telco Customer Churn dataset."""
    df_clean = df.copy()
    
    # Handle TotalCharges strings
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # Fill with median (must be calculated before drop or pass in a value)
        median_val = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val if not pd.isna(median_val) else 0)
    
    # Drop customerID (case insensitive check)
    id_cols = [c for c in df_clean.columns if c.lower() == 'customerid']
    if id_cols:
        df_clean = df_clean.drop(columns=id_cols)
        
    return df_clean
