# 🏆 Customer360 Churn Platform
> End-to-end CRM customer churn prediction platform using **Bronze → Silver → Gold Medallion Architecture**, scikit-learn Pipeline, XGBoost with GridSearchCV hyperparameter tuning, and an interactive Streamlit dashboard built on the IBM Telco Customer Churn dataset.

---
🔗 [Streamlit Link](https://customer360-churn-platform.streamlit.app/)

## 👤 Author

**Yonathan Hary Hutagalung**  
Master's in Sustainable Energy Science · Data Analyst & Data Scientist  
📍 Jakarta, Indonesia  
🔗 [LinkedIn — linkedin.com/in/yonathanhary](https://www.linkedin.com/in/yonathanhary/)  

> Feel free to connect if you'd like to discuss data science, energy analytics, or this project!

---

## 📌 Project Overview

Customer churn is one of the most critical business problems in the telecom industry. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a **production-style data pipeline and ML system** that:

- Ingests raw CRM data through a **Medallion Architecture (Bronze → Silver → Gold)**
- Engineers meaningful churn features from customer profile, billing, and service data
- Trains and tunes an **XGBoost churn prediction model** inside an sklearn Pipeline
- Scores every customer with a **churn probability and risk band**
- Serves actionable CRM retention insights via a **Streamlit dashboard**

---

## 🏗️ Architecture

```
[Raw CSV — Kaggle]
       │
       ▼
┌─────────────────────────────────────┐
│  🥉 BRONZE LAYER                    │
│  bronze_raw_telco (as-is ingestion) │
│  + ingestion log + metadata cols    │
└─────────────────────────────────────┘
       │  ETL: clean, normalize, split
       ▼
┌─────────────────────────────────────┐
│  🥈 SILVER LAYER                    │
│  silver_customer_profile            │
│  silver_account_billing             │
│  silver_service_subscription        │
│  silver_churn_label                 │
│  ─────────────────────────────────  │
│  silver_feature_table (joined)      │
│  silver_features_engineered         │
└─────────────────────────────────────┘
       │  ML Modeling + Scoring
       ▼
┌─────────────────────────────────────┐
│  🥇 GOLD LAYER                      │
│  gold_churn_scores                  │
│  (customerID, churn_probability,    │
│   risk_band, recommended_action)    │
│  ─────────────────────────────────  │
│  Streamlit Dashboard / CRM Export   │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
customer360-churn-platform/
│
├── data/
│   ├── telco_raw.csv               ← Place downloaded Kaggle CSV here
│   ├── crm_churn.db                ← database file (automatically generated)
│   └── processed/                  ← Gold output CSV written here
│
├── etl_sequence/
│   ├── 01_ingest_bronze.py         ← Bronze: raw ingestion + metadata
│   ├── 02_bronze_profiler.py       ← Bronze: data profiling report
│   ├── 03_bronze_to_silver.py      ← Silver: clean, split, join feature table
│   └── 04_silver_feature_engineering.py  ← Silver: engineered features
│
├── Notebook/
│   └── 05_gold_eda_modeling.ipynb  ← Gold: EDA + XGBoost Pipeline + GridSearchCV
│
├── sql/
│   ├── bronze_schema.sql           ← Bronze table reference schema
│   ├── silver_schema.sql           ← Silver join query reference
│   └── gold_schema.sql             ← Gold CRM view reference
│
├── models/
│   └── churn_model.pkl             ← Saved tuned Pipeline (scaler + XGBoost)
│
├── app.py                          ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Run Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Save CSV to: data/raw/telco_customer_churn.csv

# 3. Bronze — Raw ingestion
python 01_ingest_bronze.py

# 4. Bronze — Profile report (optional to make sure the table working)
python 02_bronze_profiler.py

# 5. Silver — ETL + feature table creation
python etl_sequence/03_bronze_to_silver.py

# 6. Silver — Feature engineering
python etl_sequence/04_silver_feature_engineering.py

# 7. Gold — Open and run the notebook
#    Notebook/05_gold_eda_modeling.ipynb

# 8. Launch Streamlit dashboard
streamlit run app.py
```

---

## 🤖 ML Model

| Component | Detail |
|---|---|
| **Model** | XGBoost Classifier |
| **Pipeline** | StandardScaler → XGBClassifier |
| **Tuning** | GridSearchCV (5-fold StratifiedKFold, ROC-AUC scoring) |
| **Imbalance handling** | `scale_pos_weight` = neg/pos ratio |
| **Leakage prevention** | Full sklearn Pipeline — scaler fitted on train fold only |
| **Evaluation** | ROC-AUC, Precision, Recall, F1, Confusion Matrix |

### Hyperparameter Grid
| Parameter | Values searched |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 3, 4, 5 |
| `learning_rate` | 0.01, 0.05, 0.1 |

---

## 🎯 Gold Layer Output

Each customer is scored with:

| Column | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `churn_probability` | Model output score (0.0 – 1.0) |
| `risk_band` | 🔴 High / 🟡 Medium / 🟢 Low |
| `recommended_action` | CRM action for retention team |

### Risk Band Definition
| Risk Band | Threshold | Action |
|---|---|---|
| 🔴 High | ≥ 0.70 | Offer retention discount immediately |
| 🟡 Medium | 0.40 – 0.69 | Send personalized engagement email |
| 🟢 Low | < 0.40 | No action required |

---

## 📊 Streamlit Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview | KPI cards, risk distribution, churn by contract, tenure scatter |
| 🔍 Customer Explorer | Searchable customer table, top 10 highest risk |
| 🤖 Predict New Customer | Real-time churn prediction form with gauge chart |
| 📥 Export | Download full or high-risk CRM retention list as CSV |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Engineering | Pandas, SQLite, SQL |
| ML & Pipeline | scikit-learn, XGBoost |
| Hyperparameter Tuning | GridSearchCV, StratifiedKFold |
| Dashboard | Streamlit, Plotly |
| Model Persistence | joblib |
| Architecture Pattern | Medallion (Bronze / Silver / Gold) |

---

## 📦 Dataset Credit

**IBM Telco Customer Churn Dataset**  
- Source: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- Originally published by IBM as a sample dataset  
- 7,043 customers × 21 features  
- Target variable: `Churn` (Yes/No)  

> This dataset is publicly available under Kaggle's terms of use. All credit for the original data goes to IBM and the dataset publisher on Kaggle.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Connect

If you found this project useful or want to collaborate on data science, energy analytics, or quantitative projects — feel free to reach out!

**Yonathan Hary Hutagalung**  
🔗 [linkedin.com/in/yonathanhary](https://www.linkedin.com/in/yonathanhary/)
