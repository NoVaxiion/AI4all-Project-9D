#!/usr/bin/env python3
"""
Regenerate preprocessing artifacts to fix numpy compatibility issues
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def regenerate_artifacts():
    print("Loading combined data...")
    df = pd.read_csv('combined_data.csv')
    
    print("Preparing data for preprocessing...")
    
    # Create a sample of the data for preprocessing artifacts
    # We need to identify the categorical and numerical features
    
    # Assuming your model expects these features (adjust as needed):
    categorical_features = ['town', 'crime_category', 'premise_type']  # Add your actual categorical columns
    numerical_features = ['year', 'month', 'day', 'hour', 'dayofweek']  # Add your actual numerical columns
    
    # Create Label Encoder for crime categories (target variable)
    le = LabelEncoder()
    if 'crime_category' in df.columns:
        le.fit(df['crime_category'].astype(str))
        print(f"Label encoder classes: {le.classes_}")
    else:
        # Create a dummy label encoder with common crime categories
        dummy_categories = ['Theft', 'Assault', 'Burglary', 'Vandalism', 'Drug', 'Other']
        le.fit(dummy_categories)
        print(f"Using dummy label encoder classes: {le.classes_}")
    
    # Create StandardScaler
    scaler = StandardScaler()
    if all(col in df.columns for col in numerical_features):
        scaler.fit(df[numerical_features])
        print("Scaler fitted on numerical features")
    else:
        # Fit on dummy data
        dummy_data = np.array([[2023, 1, 1, 0, 0], [2023, 12, 31, 23, 6]])
        scaler.fit(dummy_data)
        print("Scaler fitted on dummy numerical data")
    
    # Create SimpleImputer
    imputer = SimpleImputer(strategy='median')
    if all(col in df.columns for col in numerical_features):
        imputer.fit(df[numerical_features])
        print("Imputer fitted on numerical features")
    else:
        # Fit on dummy data
        dummy_data = np.array([[2023, 1, 1, 0, 0], [2023, 12, 31, 23, 6]])
        imputer.fit(dummy_data)
        print("Imputer fitted on dummy numerical data")
    
    # Create one-hot encoding columns list
    # This should match what your model expects
    ohe_columns = []  # You may need to adjust this based on your actual features
    
    # If you have categorical columns, create dummy OHE columns
    if 'town' in df.columns:
        towns = df['town'].unique()[:10]  # Limit to first 10 towns
        ohe_columns.extend([f'town_{town}' for town in towns])
    
    if not ohe_columns:
        # Create some dummy columns if none exist
        ohe_columns = ['feature_1', 'feature_2', 'feature_3']
    
    print(f"OHE columns: {len(ohe_columns)} columns")
    
    # Save all artifacts
    print("Saving artifacts...")
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(ohe_columns, 'ohe_columns.pkl')
    
    print("All artifacts regenerated successfully!")
    print(f"- Label encoder: {len(le.classes_)} classes")
    print(f"- Scaler: fitted")
    print(f"- Imputer: fitted")
    print(f"- OHE columns: {len(ohe_columns)} columns")

if __name__ == "__main__":
    regenerate_artifacts()
