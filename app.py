import os
import sys
import pickle
import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path to allow absolute imports
sys.path.append("D:/customer_churn_ai_agent")

from database.mongodb_client import db_client

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Churn Predictor & Retention Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection for glassmorphic styling
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main Background & Fonts */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 17, 30, 1) 0%, rgba(20, 24, 46, 1) 90.1%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0e18 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphism Card containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(167, 139, 250, 0.3);
    }
    
    /* Text styling gradients */
    .gradient-text {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Premium Headers */
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.5);
        border-left: 4px solid #a78bfa;
        border-radius: 8px;
        padding: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.03);
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    .metric-val {
        font-size: 32px;
        font-weight: 800;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-lbl {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom status badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
    }

    /* Style blockquotes/code templates */
    .email-container {
        background-color: #0c0e17;
        border-radius: 8px;
        padding: 20px;
        border: 1px dashed rgba(255, 255, 255, 0.1);
        font-family: 'Inter', monospace;
        white-space: pre-wrap;
        color: #cbd5e1;
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# State Management Initialization
def initialize_state():
    if "db" not in st.session_state:
        st.session_state.db = db_client
        
    if "best_model_bundle" not in st.session_state:
        st.session_state.best_model_bundle = None
        # Try to load model from disk
        model_path = "D:/customer_churn_ai_agent/saved_models/best_model.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    st.session_state.best_model_bundle = pickle.load(f)
            except Exception as e:
                st.warning(f"Error loading saved model: {e}")
                
    if "comparison_metrics" not in st.session_state:
        st.session_state.comparison_metrics = None
        metrics_path = "D:/customer_churn_ai_agent/saved_models/training_metrics.pkl"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "rb") as f:
                    st.session_state.comparison_metrics = pickle.load(f)
            except Exception as e:
                pass
                
    if "last_prediction" not in st.session_state:
        # Keep track of prediction to quickly feed into the AI retention page
        st.session_state.last_prediction = None

def main():
    initialize_state()
    inject_custom_css()
    
    # Sidebar Navigation Header
    st.sidebar.markdown(
        "<div style='text-align: center; margin-bottom: 20px;'>"
        "<h2 style='margin-bottom: 0px;'>🔮 CHURN AI</h2>"
        "<span style='color: #94a3b8; font-size: 12px;'>Predict & Retain Platform</span>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Sidebar Menu Options
    menu = [
        "🏠 Dashboard Home", 
        "📊 Exploratory Data Analysis", 
        "⚙️ Model Training & Comparison", 
        "🔮 Churn Risk Predictor", 
        "🤖 AI Retention Agent Workspace"
    ]
    
    choice = st.sidebar.radio("Navigation Menu", menu)
    
    # Show Database Status in Sidebar
    db = st.session_state.db
    status_color = "#10b981" if not db.is_mock else "#f59e0b"
    status_text = "MongoDB Connected" if not db.is_mock else "Local Mock DB (JSON)"
    
    st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<div style='font-size: 13px; color: #94a3b8;'>"
        f"Database Status:<br>"
        f"<span style='color: {status_color}; font-weight: 600;'>● {status_text}</span>"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    # Import pages dynamically
    from streamlit_app.home import show_home
    from streamlit_app.eda import show_eda
    from streamlit_app.training import show_training
    from streamlit_app.predict import show_predict
    from streamlit_app.retention import show_retention
    
    # Router
    if choice == "🏠 Dashboard Home":
        show_home()
    elif choice == "📊 Exploratory Data Analysis":
        show_eda()
    elif choice == "⚙️ Model Training & Comparison":
        show_training()
    elif choice == "🔮 Churn Risk Predictor":
        show_predict()
    elif choice == "🤖 AI Retention Agent Workspace":
        show_retention()

if __name__ == "__main__":
    main()
