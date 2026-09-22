# AI-Powered Customer Churn Analysis

> **End-to-end Data Science & ML Portfolio Project**  
> Python · Pandas · SQL · Machine Learning · XGBoost · Plotly · Streamlit · AI/LLM

---

## 📌 Project Overview

This project analyses historical customer data from a telecommunications company to:

- Identify the key factors that drive customer churn.
- Predict churn risk for every individual customer using machine learning.
- Categorise customers into **Low / Medium / High** risk tiers.
- Generate AI-powered natural language insights and retention recommendations.
- Surface everything through a professional, interactive web dashboard.

---

## 🔍 Problem Statement

Customer churn — when customers stop doing business with a company — is one of the most costly problems in any subscription-based industry.  The average cost of acquiring a new customer can be 5–25× more expensive than retaining an existing one.

This project builds an end-to-end analytics solution that helps a business:

1. Understand **why** customers churn.
2. Predict **which** customers are at risk.
3. Know **what actions** to take to reduce churn.

---

## 🎯 Objectives

1. Load and inspect a customer churn dataset from CSV.
2. Clean and preprocess the data with a fully auditable pipeline.
3. Perform detailed Exploratory Data Analysis (EDA).
4. Calculate business KPIs (churn rate, revenue at risk, etc.).
5. Execute SQL-equivalent analytics queries.
6. Train and compare multiple ML models.
7. Select the best model based on ROC-AUC and F1 Score.
8. Generate per-customer churn probability scores.
9. Assign Low / Medium / High risk categories.
10. Provide AI-generated explanations and retention recommendations.
11. Display all results through an interactive Streamlit dashboard.

---

## 📂 Dataset

**Primary:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Expected CSV name:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Place it in:** `data/raw/`

If no dataset is uploaded, the app generates a **synthetic demo dataset** that mirrors the structure and statistical properties of the real data — clearly labelled as demo data.

### Key Columns

| Column | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `gender` | Male / Female |
| `SeniorCitizen` | 0 = Non-senior, 1 = Senior |
| `Partner` | Has a partner (Yes/No) |
| `Dependents` | Has dependents (Yes/No) |
| `tenure` | Months as a customer |
| `Contract` | Month-to-month / One year / Two year |
| `MonthlyCharges` | Monthly billing amount ($) |
| `TotalCharges` | Total billed amount ($) |
| `Churn` | Target variable — Yes/No |

---

## ✨ Features

| Feature | Description |
|---|---|
| **Auto Data Cleaning** | Handles nulls, type errors, duplicates, encoding |
| **EDA Charts** | 15+ Plotly visualisations |
| **Business KPIs** | 9+ live KPI cards |
| **SQL Analysis** | 10 business queries (executed via pandas) |
| **ML Pipeline** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **Model Comparison** | Side-by-side metrics table + grouped bar chart |
| **Churn Prediction** | Per-customer probability + risk category |
| **Risk Explanations** | Data-grounded, per-customer factor summaries |
| **AI Insights** | LLM-enhanced or rule-based findings + recommendations |
| **NL Data Analyst** | Natural language Q&A about the dataset |
| **Download Centre** | CSV, Excel, and text report exports |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Visualisation | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit |
| AI / LLM | OpenAI API (optional) |
| Serialisation | joblib |
| Env management | python-dotenv |
| Notebook | Jupyter |

---

## 🏗️ Project Architecture

```
AI-Powered Customer Churn Analysis/
│
├── app.py                      # Streamlit dashboard (main entry point)
├── requirements.txt
├── .env.example                # Environment variable template
├── README.md
│
├── data/
│   ├── raw/                    # Place raw CSV here
│   └── processed/              # Cleaned data outputs
│
├── src/
│   ├── data_cleaning.py        # Full data-cleaning pipeline
│   ├── eda.py                  # Plotly EDA visualisations
│   ├── preprocessing.py        # Feature engineering + sklearn transformers
│   ├── model_training.py       # Train, compare, select ML models
│   ├── prediction.py           # Churn scoring + risk categorisation
│   ├── ai_insights.py          # LLM + rule-based insight generation
│   └── utils.py                # KPIs, SQL queries, download helpers
│
├── models/
│   ├── churn_model.pkl         # Saved best model
│   └── preprocessor.pkl        # Saved ColumnTransformer
│
├── notebooks/
│   └── churn_analysis.ipynb    # End-to-end walkthrough notebook
│
└── outputs/
    ├── predictions.csv
    └── high_risk_customers.csv
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
git clone https://github.com/your-username/ai-churn-analysis.git
cd ai-churn-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure AI API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

---

## ▶️ How to Run

### Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook churn_analysis.ipynb
```

---

## 📊 Dataset Instructions

**Option A — Real IBM Telco Dataset (Recommended)**

1. Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Save as: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Launch the app, select **Upload CSV**, and upload the file.

**Option B — Synthetic Demo Data**

1. Launch the app.
2. In the sidebar, select **Use Demo Dataset**.
3. Choose the dataset size (500–5000 rows).
4. Click **Generate Demo Data**.

The synthetic data faithfully mirrors the IBM Telco dataset structure and is ideal for development, demonstration, and testing.

---

## 🤖 Model Methodology

### Target Variable

`Churn` — Binary (1 = Churned, 0 = Active)

### Preprocessing

- **Numeric features** → StandardScaler
- **Categorical features** → OneHotEncoder (handle_unknown='ignore')
- **Train/Test split** → 80/20, stratified on the target

### Models Trained

| Model | Key Strength |
|---|---|
| Logistic Regression | Interpretable baseline, fast |
| Random Forest | Handles non-linearity, feature importance |
| Gradient Boosting | Strong ensemble for tabular data |
| XGBoost | High performance, efficient |

### Model Selection

The best model is selected by **ROC-AUC** (primary) and **F1 Score** (tie-break).  Accuracy alone is not used because churn datasets are class-imbalanced.

---

## 📈 Evaluation Metrics

| Metric | Business Meaning |
|---|---|
| **Accuracy** | Overall correctness (misleading with imbalanced data) |
| **Precision** | Of customers predicted to churn, how many actually did? |
| **Recall** | Of all churners, how many did we catch? *(prioritised)* |
| **F1 Score** | Balance of precision and recall |
| **ROC-AUC** | Model's ability to rank churners above non-churners *(primary)* |

Recall and ROC-AUC are emphasised because **missing a churner is more costly** than a false alarm.

---

## 🤖 AI Functionality

### Mode 1 — LLM-Enhanced (requires OpenAI API key)

Add `AI_API_KEY=your_key` to `.env`.  The application sends dataset statistics (never raw customer data) to the OpenAI API and receives enhanced narrative insights.

### Mode 2 — Rule-Based (default, no API key needed)

Statistical analysis and pre-defined rules generate professional insights entirely from computed dataset metrics.  **The app is fully functional without an API key.**

---

## ⚠️ Limitations

- Model accuracy depends on the quality and size of the training dataset.
- Historical patterns may not capture future market changes.
- Class imbalance means recall for the minority class (churners) may still be imperfect.
- Correlation ≠ causation — high feature importance does not mean a feature *causes* churn.
- Predictions are probabilistic and should be validated before acting on them.
- The LLM API call uses only aggregate statistics, never individual customer PII.

---

## 🚀 Future Improvements

- [ ] Real-time prediction API (FastAPI / Flask)
- [ ] SHAP values for individual prediction explainability
- [ ] Survival analysis (time-to-churn modelling)
- [ ] Class imbalance handling (SMOTE, class weights fine-tuning)
- [ ] A/B testing framework for retention campaigns
- [ ] Integration with CRM systems (Salesforce, HubSpot)
- [ ] Automated model retraining pipeline
- [ ] Customer Lifetime Value (CLV) modelling alongside churn

---

## 📸 Screenshots

Run the app and navigate to each page to generate screenshots.  Suggested captures:

1. **Executive Overview** — KPI cards + churn distribution
2. **Customer Analysis** — Filtered customer table with risk levels
3. **Churn Drivers** — Feature importance chart
4. **AI Insights** — Key findings and recommendations
5. **AI Data Analyst** — Natural language Q&A

---

## 💼 How This Project Demonstrates Data Analyst Skills

| Skill | Demonstrated Where |
|---|---|
| **Python** | All modules — clean OOP/functional code |
| **Pandas** | Data cleaning, EDA, SQL-equivalent analysis |
| **SQL** | `utils.py` SQL query bank + live execution in dashboard |
| **Data Cleaning** | `data_cleaning.py` — auditable, multi-step pipeline |
| **EDA** | `eda.py` — 15+ interactive Plotly charts |
| **Statistics** | KPIs, churn rates, correlation analysis, IQR outlier detection |
| **Machine Learning** | `model_training.py` — 4 models, proper evaluation |
| **AI / LLM** | `ai_insights.py` — OpenAI integration with graceful fallback |
| **Visualisation** | Plotly charts in every dashboard page |
| **Dashboarding** | 7-page Streamlit application |
| **Business Insights** | AI insights module + SQL analysis page |

---

## ❓ Interview / Viva Questions & Answers

### Q1: What is customer churn and why is it important?

**A:** Customer churn is when a customer stops using a company's product or service.  It is important because acquiring a new customer is 5–25× more expensive than retaining an existing one.  By predicting churn early, businesses can take proactive retention actions before the customer leaves.

---

### Q2: Why did you choose ROC-AUC as the primary model selection metric instead of accuracy?

**A:** Churn datasets are class-imbalanced — typically ~75% of customers are active and only ~25% churn.  A model that predicts "never churn" achieves 75% accuracy but is completely useless.  ROC-AUC measures the model's ability to rank actual churners above non-churners across all decision thresholds, making it much more informative.  F1 Score was used as a secondary metric because it balances both precision and recall.

---

### Q3: How does your data cleaning pipeline handle missing values?

**A:** The pipeline inspects each column after replacing empty strings with NaN.  Columns with >50% missing values are dropped.  For numeric columns, the median is used as the fill value (robust to outliers).  For categorical columns, the mode is used.  Every decision is logged in a report dictionary, making the process fully auditable.

---

### Q4: What is the difference between Precision and Recall, and which matters more for churn?

**A:** Precision is the fraction of predicted churners who actually churned.  Recall is the fraction of actual churners that the model correctly identified.  For churn prediction, **Recall is more important** — missing a real churner (false negative) is more costly than flagging an active customer for retention contact (false positive).  The cost asymmetry justifies optimising for Recall and F1.

---

### Q5: How do you handle the class imbalance problem?

**A:** Several techniques are applied: (1) `class_weight='balanced'` is set in Logistic Regression and Random Forest to penalise misclassification of the minority class more heavily; (2) stratified train/test split preserves the churn ratio in both sets; (3) ROC-AUC and F1 Score are used instead of accuracy for evaluation.  SMOTE oversampling is listed as a future improvement.

---

### Q6: What is a Confusion Matrix and how do you interpret it for churn?

**A:** A confusion matrix is a 2×2 table showing True Positives (correctly predicted churners), False Positives (active customers predicted as churners), False Negatives (churners missed by the model), and True Negatives (correctly predicted active customers).  For churn, we want to minimise False Negatives because those represent missed churners who received no retention intervention.

---

### Q7: How does your AI insight module work when no API key is available?

**A:** The `ai_insights.py` module has two modes.  If `AI_API_KEY` is set, it sends only aggregate statistics (never raw customer data) to the OpenAI API and formats the response.  If no key is configured, it falls back to a rule-based engine that generates insights by evaluating computed metrics against predefined business rules (e.g., "highest churn contract type", "average monthly charges for churned customers").  The application is fully functional in both modes.

---

### Q8: Why did you choose Streamlit for the dashboard?

**A:** Streamlit is ideal for data science portfolios because it allows building production-quality interactive dashboards entirely in Python — no JavaScript required.  It integrates naturally with pandas DataFrames, Plotly charts, and scikit-learn models.  It also supports file upload/download, session state, and custom CSS, which are all used in this project.

---

### Q9: What are the limitations of your model?

**A:** Key limitations include: (1) the model is trained on historical data which may not reflect future customer behaviour; (2) class imbalance despite mitigation means some churners will still be missed; (3) the model finds correlations, not causal relationships — TechSupport might correlate with churn not because support causes satisfaction, but because at-risk customers are more likely to call support; (4) features like TotalCharges partially encode tenure, which could introduce data leakage.

---

### Q10: How would you deploy this application to production?

**A:** Deployment would involve: (1) containerising the app with Docker; (2) using a managed hosting platform like Streamlit Cloud, AWS EC2, or Google Cloud Run; (3) storing the trained model in cloud storage (S3 / GCS) rather than local files; (4) setting up a model retraining pipeline (weekly/monthly) using Airflow or Prefect; (5) connecting to a live database instead of CSV uploads; (6) adding authentication for the dashboard; and (7) monitoring model drift over time with a framework like Evidently AI.

---

## 📄 License

MIT License — free to use, modify, and distribute for personal and commercial purposes.

---

*Built as a full-stack Data Science portfolio project demonstrating Python, Pandas, Machine Learning, AI/LLM integration, and professional dashboard development.*
