import os
import sys
import pickle
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append("D:/customer_churn_ai_agent")

from models.explain import explain_prediction
from database.mongodb_client import db_client

BEST_MODEL_PATH = "D:/customer_churn_ai_agent/saved_models/best_model.pkl"
PREPROCESSOR_PATH = "D:/customer_churn_ai_agent/saved_models/preprocessor.pkl"

def show_predict():
    st.markdown(
        "<div style='margin-bottom: 25px;'>"
        "<h1>🔮 Churn Risk Predictor</h1>"
        "<p style='color: #94a3b8; font-size: 16px; margin-top: 5px;'>"
        "Configure customer demographic and operational attributes to calculate real-time churn risk and explore individual driver explanations."
        "</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # 1. Verify that trained model is loaded
    if st.session_state.best_model_bundle is None:
        st.warning("⚠️ No champion machine learning model is loaded.")
        st.info("💡 To perform live customer predictions, you must first train the classifiers. Please navigate to the ⚙️ 'Model Training & Comparison' page and run the training benchmark.")
        return
        
    model_bundle = st.session_state.best_model_bundle
    model_name = model_bundle["model_name"]
    
    st.info(f"✨ Production Model Active: **{model_name}**")
    
    # 2. Form Inputs with Tabs
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Enter Customer Metrics</h3>", unsafe_allow_html=True)
    
    tab_demo, tab_serv, tab_bill = st.tabs(["👤 Demographics", "📶 Service Profile", "💳 Invoicing & Billing"])
    
    with tab_demo:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen (Age >= 65)", ["No", "Yes"])
        with col_d2:
            partner = st.selectbox("Has Partner/Spouse", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            
    with tab_serv:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])
        with col_s2:
            online_security = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup Service", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection Service", ["No", "Yes", "No internet service"])
        with col_s3:
            tech_support = st.selectbox("Premium Technical Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV Services", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movie Services", ["No", "Yes", "No internet service"])
            
    with tab_bill:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            contract = st.selectbox("Contract Duration Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Invoicing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Gateway Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col_b2:
            tenure = st.slider("Tenure Length (Months Subscribed)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Subscription Charge ($)", min_value=15.0, max_value=150.0, value=65.0)
            
            # Simple automatic heuristic calculation for TotalCharges to be helpful to the user
            calculated_total = float(tenure * monthly_charges)
            total_charges = st.number_input("Total Historical Charges ($)", min_value=0.0, value=calculated_total)
            
    # Submit action
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Calculate Churn Risk Profile", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 3. Calculation & Outputs
    if predict_btn:
        with st.spinner("Analyzing profile vectors and running model inference..."):
            # 3.1 Feature Engineering calculations
            # Count active services (excluding 'No' / 'No internet service' / 'No phone service')
            active_services = 0
            services_checklist = [
                ('PhoneService', phone_service), 
                ('MultipleLines', multiple_lines), 
                ('InternetService', internet_service), 
                ('OnlineSecurity', online_security),
                ('OnlineBackup', online_backup), 
                ('DeviceProtection', device_protection), 
                ('TechSupport', tech_support), 
                ('StreamingTV', streaming_tv), 
                ('StreamingMovies', streaming_movies)
            ]
            for col, val in services_checklist:
                if val not in ['No', 'No internet service', 'No phone service']:
                    active_services += 1
                    
            monthly_per_service = monthly_charges / (active_services + 1)
            
            # Tenure Grouping
            if tenure <= 12:
                tenure_group = '0-12 Months'
            elif tenure <= 24:
                tenure_group = '12-24 Months'
            elif tenure <= 48:
                tenure_group = '24-48 Months'
            elif tenure <= 60:
                tenure_group = '48-60 Months'
            else:
                tenure_group = 'Over 60 Months'
                
            senior_val = 1 if senior_citizen == "Yes" else 0
            
            # Assemble raw DataFrame
            raw_input_dict = {
                'gender': gender,
                'SeniorCitizen': senior_val,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'total_services': active_services,
                'monthly_charges_per_service': monthly_per_service,
                'tenure_group': tenure_group,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            input_df = pd.DataFrame([raw_input_dict])
            
            # 3.2 Calculate SHAP and prediction
            try:
                explanation = explain_prediction(input_df, is_processed=False)
                risk_score = explanation["prediction_value"] * 100
                
                # Determine risk level
                if risk_score < 30.0:
                    risk_level = "Low Risk"
                    badge_class = "badge-low"
                elif risk_score < 70.0:
                    risk_level = "Medium Risk"
                    badge_class = "badge-medium"
                else:
                    risk_level = "High Risk"
                    badge_class = "badge-high"
                    
                # 3.3 Log to database
                db_id = db_client.save_prediction(raw_input_dict, risk_score, risk_level)
                
                # Cache in Session State for AI Retention Agent Page Redirection
                # Include database ID for tracking
                raw_input_dict["db_id"] = db_id
                st.session_state.last_prediction = {
                    "profile": raw_input_dict,
                    "risk_score": risk_score,
                    "risk_level": risk_level
                }
                
                # Render results
                st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns([2, 3])
                
                with col_res1:
                    st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>Risk Severity: <span class='{badge_class}'>{risk_level}</span></h3>", unsafe_allow_html=True)
                    
                    # Plotly gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        number = {'suffix': "%", 'font': {'size': 44, 'family': 'Outfit'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': "#a78bfa"},
                            'bgcolor': "rgba(30, 41, 59, 0.4)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(255, 255, 255, 0.1)",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                            ],
                            'threshold': {
                                'line': {'color': "#ef4444", 'width': 3},
                                'thickness': 0.75,
                                'value': 70.0
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#cbd5e1'),
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=250
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.markdown(f"<span style='font-size: 12px; color: #94a3b8;'>Prediction saved in DB (ID: {db_id})</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # High risk marketing trigger block
                    if risk_score >= 70.0:
                        st.markdown(
                            f"<div class='glass-card' style='border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.05);'>"
                            f"<h4>⚠️ High-Risk Action Plan Triggered</h4>"
                            f"<p style='font-size: 12px; color: #cbd5e1; margin-top: 5px;'>"
                            f"Because this customer exceeds our **70% churn risk threshold**, an automated customer loyalty campaign is required."
                            f"</p>"
                            f"<div style='font-size: 13px; color: #a78bfa; font-weight: 600; margin-top: 10px;'>"
                            f"👉 Click **'AI Retention Agent Workspace'** in the sidebar to generate personalized discount offers and email templates."
                            f"</div>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                        
                with col_res2:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h4>📊 Risk Driver Explanation ({explanation['method']})</h4>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 12px; color: #94a3b8; margin-bottom: 15px;'>Inspect which features contributed to the churn score. Red (positive) bars push the probability up, while Blue (negative) bars pull it down.</p>", unsafe_allow_html=True)
                    
                    # Process top 8 features for chart
                    contribs = explanation["contributions"]
                    top_feats = list(contribs.keys())[:8]
                    top_vals = [contribs[f] for f in top_feats]
                    
                    # Generate colors: Red for positive (raising churn), Blue for negative (preventing churn)
                    colors = ['rgba(239, 68, 68, 0.75)' if v >= 0 else 'rgba(59, 130, 246, 0.75)' for v in top_vals]
                    border_colors = ['rgba(239, 68, 68, 1)' if v >= 0 else 'rgba(59, 130, 246, 1)' for v in top_vals]
                    
                    fig_shap = go.Figure(go.Bar(
                        x=top_vals,
                        y=top_feats,
                        orientation='h',
                        marker=dict(color=colors, line=dict(color=border_colors, width=1))
                    ))
                    
                    fig_shap.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#cbd5e1'),
                        xaxis=dict(title='Risk Contribution', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(autorange="reversed", showgrid=False),
                        margin=dict(t=10, b=20, l=10, r=10),
                        height=280
                    )
                    
                    st.plotly_chart(fig_shap, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                
if __name__ == "__main__":
    show_predict()
