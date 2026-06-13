import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging

# Add parent directory to path
sys.path.append("D:/customer_churn_ai_agent")

logger = logging.getLogger(__name__)

BEST_MODEL_PATH = "D:/customer_churn_ai_agent/saved_models/best_model.pkl"
PREPROCESSOR_PATH = "D:/customer_churn_ai_agent/saved_models/preprocessor.pkl"

def explain_prediction(customer_features_df, is_processed=False):
    """
    Generates local feature contributions (SHAP or fallback) for a single customer prediction.
    
    Parameters:
    - customer_features_df: Single row DataFrame of raw or processed features.
    - is_processed: If True, indicates that the input is already processed.
    
    Returns:
    - explanation_dict: Dictionary containing:
        - "base_value": Model intercept or baseline probability
        - "prediction_value": Predicted probability
        - "contributions": Dict of {feature_name: contribution_value} sorted by magnitude
        - "method": "SHAP" or "Fallback (Feature Attribution)"
    """
    # 1. Load best model bundle
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {BEST_MODEL_PATH}. Train models first.")
        
    with open(BEST_MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
        
    model = model_bundle["model"]
    model_name = model_bundle["model_name"]
    feature_names = model_bundle["feature_names"]
    
    # 2. Process customer features if raw
    if not is_processed:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}.")
        with open(PREPROCESSOR_PATH, "rb") as f:
            prep_bundle = pickle.load(f)
        preprocessor = prep_bundle["preprocessor"]
        processed_arr = preprocessor.transform(customer_features_df)
    else:
        processed_arr = customer_features_df
        
    if isinstance(processed_arr, pd.DataFrame):
        processed_arr = processed_arr.values
        
    # Get prediction probability
    pred_prob = float(model.predict_proba(processed_arr)[0, 1])
    
    # 3. Attempt SHAP explainability
    try:
        import shap
        
        # Select appropriate SHAP Explainer
        if model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(data=processed_arr))
            shap_values = explainer.shap_values(processed_arr)
            # Handle shape for binary classification
            if isinstance(shap_values, list):
                shap_val_row = shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
            elif len(shap_values.shape) > 1 and shap_values.shape[-1] == 2:
                shap_val_row = shap_values[0, :, 1]
            else:
                shap_val_row = shap_values[0]
                
            base_value = float(explainer.expected_value)
            
        elif model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_arr)
            
            # Handle output formats which vary by model and library version
            if isinstance(shap_values, list):
                # Random Forest binary class returns a list of length 2
                shap_val_row = shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
            elif len(shap_values.shape) > 2:
                # Multi-dimensional output
                shap_val_row = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
                # Shape (1, n_features)
                shap_val_row = shap_values[0]
            else:
                shap_val_row = shap_values
                
            # If shape is still 2D due to multiclass or binary (e.g. XGBoost logodds vs prob)
            if len(shap_val_row.shape) > 1 and shap_val_row.shape[1] == 2:
                shap_val_row = shap_val_row[:, 1]
                
            # Expected value
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = float(explainer.expected_value[1]) if len(explainer.expected_value) > 1 else float(explainer.expected_value[0])
            else:
                base_value = float(explainer.expected_value)
                
            # Map log-odds baseline to probability range if necessary
            # For simpler visualizations in gauge / waterfall, we'll keep logodds or map it
            
        else:
            raise ValueError("Unsupported model type for SHAP.")
            
        # Compile contributions
        contributions = {}
        for name, val in zip(feature_names, shap_val_row):
            contributions[name] = float(val)
            
        # Sort contributions by absolute value descending
        sorted_contribs = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
        
        return {
            "base_value": base_value,
            "prediction_value": pred_prob,
            "contributions": sorted_contribs,
            "method": "SHAP Explainability"
        }
        
    except Exception as e:
        logger.warning(f"SHAP explanation failed or not installed: {e}. Running custom Feature Attribution fallback.")
        
        # Fallback Calculation: Model-based Feature Attribution
        # Calculate feature attribution using model coefficients/feature importances multiplied by scaled feature value.
        contributions = {}
        row_values = processed_arr[0]
        
        # For fallback base value, we can use the average churn rate of the Telco dataset (~0.265)
        base_value = 0.265
        
        if model_name == "Logistic Regression":
            # For linear model, contribution is exactly coefficient * value
            coefs = model.coef_[0]
            raw_contribs = row_values * coefs
            # Normalize to sum up to roughly represent difference from base value
            diff = pred_prob - base_value
            total_abs = sum(abs(raw_contribs)) or 1
            scaled_contribs = [c * (diff / total_abs) for c in raw_contribs]
            
            for name, val in zip(feature_names, scaled_contribs):
                contributions[name] = float(val)
                
        else: # Random Forest or XGBoost
            # Use saved feature importances from model training
            saved_importances = model_bundle["feature_importances"]
            
            # Estimate direction based on customer's feature value compared to average
            # (Higher monthly charges increases churn risk, longer tenure decreases it, etc.)
            raw_contribs = []
            for name in feature_names:
                # Default direction
                direction = 1
                
                # Heuristics based on common churn indicators
                if "tenure" in name or "Contract_One year" in name or "Contract_Two year" in name or "OnlineSecurity_Yes" in name or "TechSupport_Yes" in name:
                    direction = -1 # Lowers churn risk
                elif "MonthlyCharges" in name or "PaymentMethod_Electronic check" in name or "InternetService_Fiber optic" in name:
                    direction = 1  # Raises churn risk
                    
                # Look up importance
                imp = saved_importances.get(name, {}).get("importance", 0.0)
                raw_contribs.append(imp * direction)
                
            # Scale contributions so their sum matches the difference between predictions and base value
            diff = pred_prob - base_value
            total_abs = sum(abs(c) for c in raw_contribs) or 1
            scaled_contribs = [c * (diff / total_abs) for c in raw_contribs]
            
            for name, val in zip(feature_names, scaled_contribs):
                contributions[name] = float(val)
                
        # Sort contributions by absolute value descending
        sorted_contribs = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
        
        return {
            "base_value": base_value,
            "prediction_value": pred_prob,
            "contributions": sorted_contribs,
            "method": "Feature Attribution Fallback"
        }

if __name__ == "__main__":
    # Test code with dummy input if model exists
    try:
        dummy_row = pd.DataFrame([{
            'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
            'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
            'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85, 'TotalCharges': 29.85,
            'total_services': 1, 'monthly_charges_per_service': 14.925, 'tenure_group': '0-12 Months'
        }])
        
        exp = explain_prediction(dummy_row)
        print("Success! Method:", exp["method"])
        print("Top 3 contributions:")
        for k, v in list(exp["contributions"].items())[:3]:
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print("Could not run test (models likely not trained yet):", e)
