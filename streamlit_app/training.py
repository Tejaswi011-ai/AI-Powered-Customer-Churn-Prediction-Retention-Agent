import os
import sys
import pickle
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append("D:/customer_churn_ai_agent")

METRICS_PATH = "D:/customer_churn_ai_agent/saved_models/training_metrics.pkl"
BEST_MODEL_PATH = "D:/customer_churn_ai_agent/saved_models/best_model.pkl"

def show_training():
    st.markdown(
        "<div style='margin-bottom: 25px;'>"
        "<h1>⚙️ Model Training & Comparison</h1>"
        "<p style='color: #94a3b8; font-size: 16px; margin-top: 5px;'>"
        "Train, evaluate, and compare three high-performing machine learning classifiers to determine which model is selected to power the customer churn predictions."
        "</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Action panel
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>🎯 Execute Training Pipeline</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 13px; color: #94a3b8;'>Runs the ingestion script, performs cleaning, engineers features, fits the standard preprocessors, and executes comparative training across Logistic Regression, Random Forest, and XGBoost.</p>", unsafe_allow_html=True)
    
    train_button = st.button("🚀 Train & Benchmark Models", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 1. Check if we need to execute training or if we have cached results
    if train_button:
        with st.spinner("Executing Data Preprocessing & Model Benchmark Ingestion..."):
            try:
                # Import training routine directly
                from models.train import train_and_compare
                
                # Execute pipeline
                best_name, best_metrics = train_and_compare()
                
                # Reload model bundle and metrics into session state
                with open(BEST_MODEL_PATH, "rb") as f:
                    st.session_state.best_model_bundle = pickle.load(f)
                with open(METRICS_PATH, "rb") as f:
                    st.session_state.comparison_metrics = pickle.load(f)
                    
                st.success(f"🎉 Pipeline run completed successfully! Winner Model: {best_name} (F1 Score: {best_metrics['F1 Score']:.4f})")
            except Exception as e:
                st.error(f"❌ Training Pipeline failed: {e}")
                st.info("Check if D:\\ drive is writable or dependencies are correctly installed.")
                return
                
    # 2. Check if metrics are loaded in session state or disk
    if st.session_state.comparison_metrics is None:
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, "rb") as f:
                    st.session_state.comparison_metrics = pickle.load(f)
            except Exception:
                pass
                
    metrics_bundle = st.session_state.comparison_metrics
    
    if metrics_bundle is None:
        st.info("💡 No active trained models found in cache. Click the 'Train & Benchmark Models' button above to run the machine learning pipeline!")
        return
        
    results = metrics_bundle["results"]
    winner = metrics_bundle["winner"]
    
    # 3. Highlight Winner Model Banner
    st.markdown(
        f"<div class='glass-card' style='border-left: 6px solid #a78bfa; background: rgba(167, 139, 250, 0.08);'>"
        f"<h3 style='margin: 0;'>🏆 Selected Champion: <span style='color: #c084fc;'>{winner}</span></h3>"
        f"<p style='margin-bottom: 0px; margin-top: 5px; font-size: 14px; color: #cbd5e1;'>"
        f"This model was automatically selected for production predictions based on its superior **F1 Score** ({results[winner]['F1 Score']:.4f}) and robust generalization profile."
        f"</p>"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. Render comparative metric table / grid
    st.subheader("📊 Comparative Benchmarks")
    
    # Construct Pandas DataFrame for comparative view
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Classifier": model_name,
            "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
            "Precision": f"{metrics['Precision']*100:.2f}%",
            "Recall": f"{metrics['Recall']*100:.2f}%",
            "F1 Score": f"{metrics['F1 Score']*100:.2f}%",
            "ROC-AUC": f"{metrics['ROC-AUC']:.4f}"
        })
    df_metrics = pd.DataFrame(rows)
    
    # Display in a beautiful format
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 5. Side-by-Side: Interactive confusion matrix & Feature Importance of winner
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Confusion Matrix Explorer</h4>", unsafe_allow_html=True)
        
        # Dropdown to inspect different models' confusion matrices
        selected_model_cm = st.selectbox("Inspect Model:", list(results.keys()), index=list(results.keys()).index(winner))
        cm = results[selected_model_cm]["Confusion Matrix"]
        
        # Plotly heatmap for Confusion Matrix
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Active (0)', 'Predicted Churned (1)'],
            y=['Actual Active (0)', 'Actual Churned (1)'],
            colorscale='Blues',
            text=[[f"True Negatives<br><b>{cm[0][0]}</b>", f"False Positives<br><b>{cm[0][1]}</b>"],
                  [f"False Negatives<br><b>{cm[1][0]}</b>", f"True Positives<br><b>{cm[1][1]}</b>"]],
            texttemplate="%{text}",
            showscale=False
        ))
        
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            yaxis=dict(autorange="reversed"), # Orient correctly
            margin=dict(t=30, b=20, l=10, r=10),
            height=280
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Top Core Feature Importances</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 11px; color: #94a3b8;'>Displaying relative feature weight impact based on the selected best performing model.</p>", unsafe_allow_html=True)
        
        # Load best model feature importances
        if st.session_state.best_model_bundle is not None:
            model_bundle = st.session_state.best_model_bundle
            feat_imp = model_bundle["feature_importances"]
            
            # Take top 8 feature importances
            top_features = list(feat_imp.keys())[:8]
            top_scores = [feat_imp[f]["importance"] for f in top_features]
            
            # Horizontal Bar Chart
            fig_imp = go.Figure(go.Bar(
                x=top_scores,
                y=top_features,
                orientation='h',
                marker=dict(color='rgba(167, 139, 250, 0.7)', line=dict(color='rgba(167, 139, 250, 1)', width=1))
            ))
            
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                xaxis=dict(title='Relative Importance', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(autorange="reversed", showgrid=False),
                margin=dict(t=10, b=20, l=10, r=10),
                height=250
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Train the winner model bundle to unlock feature importance coefficients.")
            
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    show_training()
