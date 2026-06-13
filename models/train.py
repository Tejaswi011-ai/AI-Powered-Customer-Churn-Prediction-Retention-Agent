import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Add parent directory to path to allow absolute imports
sys.path.append("D:/customer_churn_ai_agent")

from data.data_preprocessing import preprocess_and_split

os.makedirs("D:/customer_churn_ai_agent/saved_models", exist_ok=True)
BEST_MODEL_PATH = "D:/customer_churn_ai_agent/saved_models/best_model.pkl"
TRAINING_METRICS_PATH = "D:/customer_churn_ai_agent/saved_models/training_metrics.pkl"

def train_and_compare():
    print("Starting Model Ingestion and Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split()
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=3.0,  # Handle class imbalance (roughly 3:1 negative to positive ratio)
            eval_metric='logloss',
            random_state=42
        )
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON compatibility
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": auc,
            "Confusion Matrix": cm
        }
        
        print(f"{name} Metrics -> Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    # Automatically select the best model based on F1 Score
    # F1 score is selected as the primary metric because customer churn is typically imbalanced.
    best_model_name = max(results, key=lambda k: results[k]["F1 Score"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\nWinner Model: {best_model_name}")
    print(f"Winner F1 Score: {best_metrics['F1 Score']:.4f}")
    
    # Feature Importances/Coefficients for the winner model
    feature_importances = {}
    if best_model_name == "Logistic Regression":
        importances = np.abs(best_model.coef_[0])
        raw_importances = best_model.coef_[0]
    elif best_model_name == "Random Forest":
        importances = best_model.feature_importances_
        raw_importances = importances
    else: # XGBoost
        importances = best_model.feature_importances_
        raw_importances = importances
        
    for feat, imp, raw_imp in zip(feature_names, importances, raw_importances):
        feature_importances[feat] = {
            "importance": float(imp),
            "raw_importance": float(raw_imp)
        }
        
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1]["importance"], reverse=True))
    
    # Save the winner model bundle
    best_model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "metrics": best_metrics,
        "feature_names": feature_names,
        "feature_importances": sorted_importances
    }
    
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model_bundle, f)
    print(f"Saved best model bundle to {BEST_MODEL_PATH}")
    
    # Save all results for display on the streamlit comparison page
    comparison_bundle = {
        "results": results,
        "winner": best_model_name,
        "feature_names": feature_names
    }
    with open(TRAINING_METRICS_PATH, "wb") as f:
        pickle.dump(comparison_bundle, f)
    print(f"Saved comparison metrics to {TRAINING_METRICS_PATH}")
    
    return best_model_name, best_metrics

if __name__ == "__main__":
    train_and_compare()
