import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create target directories if they don't exist
os.makedirs("D:/customer_churn_ai_agent/data", exist_ok=True)
os.makedirs("D:/customer_churn_ai_agent/saved_models", exist_ok=True)

RAW_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = "D:/customer_churn_ai_agent/data"
RAW_CSV_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "processed_churn_data.csv")
PREPROCESSOR_PATH = "D:/customer_churn_ai_agent/saved_models/preprocessor.pkl"

def download_dataset():
    """Downloads the IBM Telco Churn dataset programmatically."""
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Downloading Telco Churn dataset from: {RAW_DATA_URL}")
        try:
            urllib.request.urlretrieve(RAW_DATA_URL, RAW_CSV_PATH)
            print("Dataset downloaded successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    else:
        print("Dataset already cached locally.")

def load_and_clean_data():
    """Loads the cached dataset and performs standard data cleaning."""
    download_dataset()
    df = pd.read_csv(RAW_CSV_PATH)
    
    # 1. Clean TotalCharges (has empty spaces for tenure=0)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    # Convert to numeric, and fill empty spaces with 0.0 (since they are new customers with 0 tenure)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # 2. Map Churn target variable to 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def engineer_features(df):
    """Performs domain-specific feature engineering to improve model performance."""
    # Create copy to prevent slice warning
    df = df.copy()
    
    # 1. Total services subscribed (helps assess customer engagement)
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Count how many services are subscribed (where service does not equal 'No' or 'No internet service')
    total_services = np.zeros(len(df))
    for col in service_cols:
        if col in df.columns:
            # Active service check: is not 'No' and is not 'No internet service' and is not 'No phone service'
            is_active = ~df[col].isin(['No', 'No internet service', 'No phone service'])
            total_services += is_active.astype(int)
            
    df['total_services'] = total_services
    
    # 2. Monthly charge per active service
    # We add 1 to denominator to avoid division by zero
    df['monthly_charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)
    
    # 3. Tenure grouping (useful for categorical analysis in EDA)
    def group_tenure(t):
        if t <= 12:
            return '0-12 Months'
        elif t <= 24:
            return '12-24 Months'
        elif t <= 48:
            return '24-48 Months'
        elif t <= 60:
            return '48-60 Months'
        else:
            return 'Over 60 Months'
            
    df['tenure_group'] = df['tenure'].apply(group_tenure)
    
    return df

def preprocess_and_split():
    """Applies preprocessing transformations, fits encoders, saves preprocessor, and splits dataset."""
    df = load_and_clean_data()
    df = engineer_features(df)
    
    # Save clean dataset for EDA page
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"Cleaned and engineered dataset saved to {CLEAN_CSV_PATH}")
    
    # Features and Target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    
    # Identify continuous and categorical columns
    # We don't include customerID. We include engineered numeric features.
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services', 'monthly_charges_per_service']
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Get column names after OHE transformation
    cat_encoder = preprocessor.named_transformers_['cat']
    # Get feature names for categorical columns
    cat_features = cat_encoder.get_feature_names_out(cat_cols)
    all_features = num_cols + list(cat_features)
    
    # Save the fitted preprocessor along with feature names
    preprocessor_bundle = {
        'preprocessor': preprocessor,
        'feature_names': all_features,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
    
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor_bundle, f)
    print(f"Preprocessing bundle saved to {PREPROCESSOR_PATH}")
    
    return X_train_proc, X_test_proc, y_train, y_test, all_features

if __name__ == "__main__":
    preprocess_and_split()
