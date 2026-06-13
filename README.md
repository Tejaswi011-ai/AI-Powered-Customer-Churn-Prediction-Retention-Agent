# AI-Powered Customer Churn Prediction & Retention Agent

An internship-level, production-ready, and end-to-end Machine Learning + Generative AI customer retention platform. Built using **Python**, the **IBM Telco Customer Churn dataset**, **scikit-learn**, **XGBoost**, **SHAP**, **MongoDB**, and **Google Gemini API**, all unified under a gorgeous modern **glassmorphic Streamlit Dashboard**.

## Key Architecture & Features
1. **Interactive Exploratory Data Analysis (EDA)**: Fully interactive Plotly visualizations tracking churn distribution, contract lengths, monthly bills, and customer tenure.
2. **Three ML Classifiers Compared**: Side-by-side performance benchmarks (Accuracy, Precision, Recall, F1 Score, ROC-AUC) for **Logistic Regression**, **Random Forest**, and **XGBoost**, automatically selecting and saving the best model based on F1 Score.
3. **Local SHAP Explainability**: High-fidelity local explanations showing exactly how much each customer feature pushes the attrition probability up (red) or down (blue).
4. **AI Retention Agent Workspace**: Leverages the **Google Gemini API** (with a premium rule-based fallback generator for offline or keyless use) to diagnose reasons for churn, suggest actionable account recovery strategies, authorize loyalty incentives, and draft highly personalized, editable rescue emails.
5. **Robust Database Layer**: Integrates with MongoDB to automatically log customer prediction records and deployed retention campaign details. **Gracefully falls back to a mock local JSON-based database if MongoDB is not running**, enabling immediate out-of-the-box operation!

---

## Directory Structure
```text
customer_churn_ai_agent/
│
├── data/
│   └── data_preprocessing.py      # Programmatic ingestion, cleaning, OHE, and feature engineering
│
├── models/
│   ├── train.py                   # Classifier training, comparison, metric scoring, and winner selection
│   └── explain.py                 # SHAP-based local explainability module with custom attribution fallback
│
├── database/
│   ├── mongodb_client.py          # MongoDB PyMongo client with auto-JSON mock fallback integration
│   └── (mock_db.json)             # Local JSON file created automatically in fallback mode
│
├── ai_agent/
│   └── retention_agent.py         # Google Gemini AI agent + rule-based heuristic fallback generator
│
├── streamlit_app/
│   ├── home.py                    # Home landing workspace and operational live stats
│   ├── eda.py                     # Rich interactive Plotly dashboards and filtered cohort explorer
│   ├── training.py                # Visual ML model comparisons and Plotly confusion matrix heatmaps
│   ├── predict.py                 # Premium input forms, real-time risk gauges, and SHAP bar charts
│   └── retention.py               # Generative retention planner, editable email console, and campaign logs
│
├── saved_models/                  # Pickled champion models, preprocessors, and benchmark sheets
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── training_metrics.pkl
│
├── requirements.txt               # All production and development Python dependencies
├── .env                           # API credentials and local database URIs
├── README.md                      # Comprehensive developer setup guide
└── app.py                         # Unified Streamlit application entrypoint & CSS styles
```

---

## Step-by-Step Execution Instructions

### Phase 1: Environment Setup
Ensure you have **Python 3.9+** and **Git** installed on your system.

1. **Navigate to the target directory** in your terminal:
   ```bash
   cd D:\customer_churn_ai_agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -v env venv
   # or
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   * **PowerShell (Windows)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Command Prompt (Windows)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   * **Bash (macOS/Linux)**:
     ```bash
     source venv/bin/activate
     ```

4. **Install all required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Phase 2: Configuration
1. Open the `.env` file located in `D:\customer_churn_ai_agent\.env`.
2. **MongoDB Connection**: The default `MONGO_URI=mongodb://localhost:27017/` connects to local MongoDB. Update this if you are using MongoDB Atlas or another host.
3. **Gemini API Key**: Replace `YOUR_GEMINI_API_KEY_HERE` with your actual Google Generative AI API Key. 
   *(Note: If left empty or as the placeholder, the AI Retention Agent will automatically operate in **Heuristic Fallback Mode**, allowing you to preview stunning, highly accurate campaign logs without requiring an API key).*

---

### Phase 3: Start Services

#### Options for MongoDB
* **Option A: Local MongoDB Server**
  Start your MongoDB Community Server from your Windows Services menu, or open a Command Prompt and run:
  ```cmd
  mongod
  ```
* **Option B: Docker MongoDB Container**
  If you have Docker Desktop installed, run:
  ```bash
  docker run -d -p 27017:27017 --name local-mongo mongo:latest
  ```
* **Option C: Offline/JSON Mode (No setup required)**
  Do absolutely nothing! The application will automatically detect if MongoDB is running and seamlessly fall back to local JSON database tracking if it is offline.

---

### Phase 4: Run Machine Learning Training Pipeline
Before launching the UI, run the data ingestion and model training routine to download the IBM dataset, perform feature engineering, and train the classifiers.

```bash
python models/train.py
```
*Expected Output:*
- Downloads the Telco CSV programmatically.
- Cleans and engineers two custom domain features (`total_services`, `monthly_charges_per_service`).
- Preprocesses data and trains Logistic Regression, Random Forest, and XGBoost.
- Automatically selects the winner based on F1 Score and saves files in `saved_models/`.

*(Note: You can also execute this entire process directly from the Streamlit UI with 1-click in the "Model Training" tab!)*

---

### Phase 5: Launch Streamlit Dashboard App
Launch the interactive visual dashboard:

```bash
streamlit run app.py
```

Streamlit will boot up and automatically open the application in your default web browser (typically at `http://localhost:8501`).

---

## Technical Specifications
* **Dataset Ingest Source**: IBM Customer Churn Data Raw Ingest (7,043 rows)
* **Model Training Split**: Stratified 80% train, 20% test
* **F1 Selection Threshold**: Model with highest test F1 Score (balances recall and precision on imbalanced churn dataset)
* **API Framework**: Google Generative AI Python SDK (`gemini-1.5-flash`)
* **Explainability Library**: SHAP (`shap` package with dynamic custom attribution visualization fallback)
* **Interactive Engine**: Plotly Express + Plotly Graph Objects
