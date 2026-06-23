import pandas as pd
import sqlite3
import os

DB_PATH = "data/crm_churn.db"

def load_bronze(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM bronze_raw_telco", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from Bronze layer.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix TotalCharges (raw CSV has spaces that prevent numeric casting)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Standardize binary Yes/No columns to 1/0
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Lowercase and strip whitespace on all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip().str.lower()

    print(f"Data cleaned. Shape: {df.shape}")
    return df

def split_into_silver_tables(df: pd.DataFrame, conn: sqlite3.Connection):
    # customer_profile
    customer_profile = df[[
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure"
    ]].copy()
    customer_profile.to_sql("silver_customer_profile", conn, if_exists="replace", index=False)
    print(f"silver_customer_profile written: {len(customer_profile)} rows.")

    # account_billing
    account_billing = df[[
        "customerID", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]].copy()
    account_billing.to_sql("silver_account_billing", conn, if_exists="replace", index=False)
    print(f"silver_account_billing written: {len(account_billing)} rows.")

    # service_subscription
    service_cols = [
        "customerID", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    service_subscription = df[service_cols].copy()
    service_subscription.to_sql("silver_service_subscription", conn, if_exists="replace", index=False)
    print(f"silver_service_subscription written: {len(service_subscription)} rows.")

    # churn_label
    churn_label = df[["customerID", "Churn"]].copy()
    churn_label.to_sql("silver_churn_label", conn, if_exists="replace", index=False)
    print(f"silver_churn_label written: {len(churn_label)} rows.")

def create_silver_feature_table(conn: sqlite3.Connection):
    """Join all silver CRM tables into one analytics-ready feature table"""
    # DROP first to avoid conflicts on re-runs
    conn.execute("DROP TABLE IF EXISTS silver_feature_table")
    conn.execute("""
        CREATE TABLE silver_feature_table AS
        SELECT
            cp.customerID,
            cp.gender,
            cp.SeniorCitizen,
            cp.Partner,
            cp.Dependents,
            cp.tenure,

            ab.Contract,
            ab.PaperlessBilling,
            ab.PaymentMethod,
            ab.MonthlyCharges,
            ab.TotalCharges,

            ss.PhoneService,
            ss.MultipleLines,
            ss.InternetService,
            ss.OnlineSecurity,
            ss.OnlineBackup,
            ss.DeviceProtection,
            ss.TechSupport,
            ss.StreamingTV,
            ss.StreamingMovies,

            cl.Churn

        FROM silver_customer_profile cp
        LEFT JOIN silver_account_billing ab ON cp.customerID = ab.customerID
        LEFT JOIN silver_service_subscription ss ON cp.customerID = ss.customerID
        LEFT JOIN silver_churn_label cl ON cp.customerID = cl.customerID
    """)
    conn.commit()

    # Verify row count
    count = pd.read_sql("SELECT COUNT(*) as cnt FROM silver_feature_table", conn)["cnt"][0]
    print(f"silver_feature_table created successfully: {count} rows.")

def run():
    conn = sqlite3.connect(DB_PATH)

    # Step 1 — Load Bronze
    df = load_bronze(DB_PATH)

    # Step 2 — Clean data
    df = clean_data(df)

    # Step 3 — Split into Silver CRM tables
    split_into_silver_tables(df, conn)

    # Step 4 — Join into unified Silver feature table
    create_silver_feature_table(conn)

    conn.close()
    print("\nBronze → Silver pipeline complete.")

if __name__ == "__main__":
    run()