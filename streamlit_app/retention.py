import os
import sys
import streamlit as st
import pandas as pd

# Add parent directory to path
sys.path.append("D:/customer_churn_ai_agent")

from ai_agent.retention_agent import generate_retention_plan
from database.mongodb_client import db_client

def show_retention():
    st.markdown(
        "<div style='margin-bottom: 25px;'>"
        "<h1>🤖 AI Retention Agent Workspace</h1>"
        "<p style='color: #94a3b8; font-size: 16px; margin-top: 5px;'>"
        "Generate automated, data-driven churn reasonings, marketing rescue strategies, and high-converting personalized client correspondence using Google Gemini AI."
        "</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    db = st.session_state.db
    history = db.get_prediction_history()
    
    # 1. Select Customer Section
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>👤 Select Customer Cohort</h3>", unsafe_allow_html=True)
    
    # Check if there is a predicted customer in session state
    last_pred = st.session_state.last_prediction
    
    selector_options = []
    selected_customer = None
    
    if last_pred is not None:
        last_id = last_pred["profile"]["customer_id"]
        last_risk = last_pred["risk_score"]
        selector_options.append(f"⏱️ Current Prediction Target ({last_id} - Risk: {last_risk:.1f}%)")
        
    # Populate other records from MongoDB prediction history
    for item in history:
        cust_id = item["customer_id"]
        risk = item["risk_score"]
        # Avoid duplicate listing of last predicted customer
        if last_pred is not None and cust_id == last_id:
            continue
        # Standard format
        selector_options.append(f"📦 DB Log: {cust_id} (Risk: {risk:.1f}%)")
        
    # Fallback placeholder if database is completely empty
    if not selector_options:
        selector_options.append("📝 Load Prototype Customer Profile (Demo Mode)")
        
    selected_choice = st.selectbox("Choose a customer to analyze:", selector_options)
    
    # Extract selected customer profile details
    if last_pred is not None and "Current Prediction Target" in selected_choice:
        selected_customer = last_pred["profile"]
        selected_risk = last_pred["risk_score"]
    elif "DB Log:" in selected_choice:
        # Extract ID from string e.g. "DB Log: 1234-ABCD (Risk: 85.0%)"
        cust_id_extracted = selected_choice.split("DB Log: ")[1].split(" (")[0]
        # Find in history
        match = next((item for item in history if item["customer_id"] == cust_id_extracted), None)
        if match:
            selected_customer = match["customer_features"]
            selected_risk = match["risk_score"]
    else:
        # Fallback Mock Profile for demonstrations (92% risk month-to-month fiber)
        selected_customer = {
            'customer_id': 'DEMO_92104',
            'gender': 'Male',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'total_services': 3,
            'monthly_charges_per_service': 22.10,
            'tenure_group': '0-12 Months',
            'MonthlyCharges': 88.40,
            'TotalCharges': 176.80
        }
        selected_risk = 92.4
        
    # Display Customer Profile Card
    st.markdown("<br><b>Active Customer Features Summary:</b>", unsafe_allow_html=True)
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.write(f"**ID**: `{selected_customer.get('customer_id', 'N/A')}`")
        st.write(f"**Tenure**: `{selected_customer.get('tenure', 'N/A')} Months`")
    with col_c2:
        st.write(f"**Contract**: `{selected_customer.get('Contract', 'N/A')}`")
        st.write(f"**Charges**: `${selected_customer.get('MonthlyCharges', 'N/A')}/mo`")
    with col_c3:
        st.write(f"**Internet**: `{selected_customer.get('InternetService', 'N/A')}`")
        st.write(f"**Tech Support**: `{selected_customer.get('TechSupport', 'N/A')}`")
    with col_c4:
        st.write(f"**Risk Score**: `{selected_risk:.1f}%`")
        status_text = "🔴 HIGH RISK (>70%)" if selected_risk >= 70 else "🟡 MEDIUM/LOW"
        st.write(f"**Status**: **{status_text}**")
        
    st.markdown("<br>", unsafe_allow_html=True)
    generate_agent_btn = st.button("🤖 Launch AI Retention Agent Ingestion", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 2. Trigger AI Agent
    if generate_agent_btn:
        with st.spinner("Connecting to Gemini AI Engine... Formulating retention plan..."):
            plan = generate_retention_plan(selected_customer, selected_risk)
            
            st.session_state.active_retention_plan = {
                "customer_id": selected_customer.get("customer_id", "DEMO_ID"),
                "risk_score": selected_risk,
                "plan": plan
            }
            st.success(f"⚡ Retention Plan compiled using **{plan['method']}**!")
            
    # 3. Render Generated Plan
    if "active_retention_plan" in st.session_state:
        plan_bundle = st.session_state.active_retention_plan
        cust_id = plan_bundle["customer_id"]
        risk = plan_bundle["risk_score"]
        plan = plan_bundle["plan"]
        
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
        
        # Pre-format lists into HTML for clean f-string compatibility across all Python versions
        reasons_html = "".join([f"<li>{r}</li>" for r in plan.get('reasons', [])])
        strategies_html = "".join([f"<li>{s}</li>" for s in plan.get('strategies', [])])
        
        offers_list = []
        for o in plan.get('offers', []):
            if ":" in o:
                parts = o.split(":", 1)
                offers_list.append(f"<li><b>{parts[0].strip()}:</b>{parts[1]}</li>")
            else:
                offers_list.append(f"<li>{o}</li>")
        offers_html = "".join(offers_list)

        # Grid Dashboard for AI results
        col_res_l, col_res_r = st.columns([1, 1])
        
        with col_res_l:
            # Column 1: Churn Reasons
            st.markdown(
                f"<div class='glass-card' style='border-left: 4px solid #ef4444; min-height: 250px;'>"
                f"<h4>🚨 Risk Root-Cause Analysis</h4>"
                f"<hr style='border-color: rgba(255,255,255,0.05); margin-top: 5px; margin-bottom: 10px;'>"
                f"<ul>"
                f"  {reasons_html}"
                f"</ul>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            # Column 2: Recommended Strategies & Offers
            st.markdown(
                f"<div class='glass-card' style='border-left: 4px solid #3b82f6; min-height: 250px;'>"
                f"<h4>💡 Actionable Account Strategies</h4>"
                f"<hr style='border-color: rgba(255,255,255,0.05); margin-top: 5px; margin-bottom: 10px;'>"
                f"<ul>"
                f"  {strategies_html}"
                f"</ul>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"<div class='glass-card' style='border-left: 4px solid #10b981; min-height: 180px;'>"
                f"<h4>🎁 Authorized Loyalty Incentives</h4>"
                f"<hr style='border-color: rgba(255,255,255,0.05); margin-top: 5px; margin-bottom: 10px;'>"
                f"<ul>"
                f"  {offers_html}"
                f"</ul>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        with col_res_r:
            # Column 3: Generated Email (Interactive Editor)
            st.markdown("<div class='glass-card' style='min-height: 720px;'>", unsafe_allow_html=True)
            st.markdown("<h4>✉️ Generated Client Correspondence</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 11px; color: #94a3b8; margin-bottom: 10px;'>Edit the subject line or email body below before dispatching or logging the campaign.</p>", unsafe_allow_html=True)
            
            edited_subject = st.text_input("Email Subject Line:", value=plan.get("email_subject", "Exclusive Loyalty Rewards For You"))
            edited_body = st.text_area("Email Content Body:", value=plan.get("email_body", ""), height=450)
            
            # Interactive Save to MongoDB button
            save_plan_btn = st.button("💾 Deploy & Log Loyalty Campaign", type="primary", use_container_width=True)
            
            if save_plan_btn:
                try:
                    plan_id = db.save_retention_plan(
                        customer_id=cust_id,
                        risk_score=risk,
                        reasons=plan.get("reasons", []),
                        strategies=plan.get("strategies", []),
                        offers=plan.get("offers", []),
                        email_subject=edited_subject,
                        email_body=edited_body
                    )
                    st.success(f"💾 Campaign logged in database! Campaign ID: `{plan_id}`")
                except Exception as e:
                    st.error(f"Failed to log plan to MongoDB: {e}")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. View Deployment logs
    st.subheader("📋 Campaign Logs & Auditing Timeline")
    st.markdown("<p style='font-size: 13px; color: #94a3b8;'>Inspect past retention campaigns created for high-risk customer accounts.</p>", unsafe_allow_html=True)
    
    db_plans = db.get_retention_plans()
    
    if db_plans:
        for idx, item in enumerate(db_plans):
            with st.expander(f"Campaign Run: Customer `{item['customer_id']}` (Risk: {item['risk_score']:.1f}%) — {item['timestamp'][:16].replace('T', ' ')}"):
                col_e1, col_e2 = st.columns([1, 1])
                with col_e1:
                    st.write("**Identified Churn Drivers:**")
                    for r in item.get("reasons", []):
                        st.write(f"- {r}")
                    st.write("**Authorized Special Offers:**")
                    for o in item.get("offers", []):
                        st.write(f"- {o}")
                with col_e2:
                    st.write("**Deployed Loyalty Email:**")
                    st.markdown(f"<div class='email-container'><b>Subject: {item.get('email_subject','')}</b>\n\n{item.get('email_body','')}</div>", unsafe_allow_html=True)
    else:
        st.info("💡 No retention action plans found in database history logs. Complete the Gemini agent run above and click 'Deploy & Log Loyalty Campaign' to record your first audit!")

if __name__ == "__main__":
    show_retention()
