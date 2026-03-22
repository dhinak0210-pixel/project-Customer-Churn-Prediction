import pandas as pd
import numpy as np
import os

def generate_churn_data(n_samples=1000, output_path='/home/dhina/Customer Churn Analysis/data/churn_data.csv'):
    np.random.seed(42)
    
    data = {
        'customerID': [f'{i:04d}-ABCDE' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['No phone service', 'No', 'Yes'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.around(np.random.uniform(18, 120, n_samples), 2),
    }

    # Simulate TotalCharges as tenure * MonthlyCharges with some noise
    data['TotalCharges'] = np.around(data['tenure'] * data['MonthlyCharges'] * np.random.uniform(0.95, 1.05, n_samples), 2)
    
    # Simulate Churn based on tenure, MonthlyCharges, and Contract
    # Lower tenure, higher monthly charges, and Month-to-month contract increases churn
    churn_prob = (
        (72 - np.array(data['tenure'])) / 72 * 0.4 +
        (np.array(data['MonthlyCharges']) - 18) / 102 * 0.3 +
        (np.where(np.array(data['Contract']) == 'Month-to-month', 1, 0)) * 0.3
    )
    # Add noise
    churn_prob += np.random.normal(0, 0.1, n_samples)
    data['Churn'] = np.where(churn_prob > 0.5, 'Yes', 'No')

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {n_samples} samples at {output_path}")

if __name__ == "__main__":
    generate_churn_data()
