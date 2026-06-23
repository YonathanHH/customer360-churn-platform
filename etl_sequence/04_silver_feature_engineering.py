import pandas as pd
import sqlite3

DB_PATH = "data/crm_churn.db"

def load_silver_feature_table(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM silver_feature_table", conn)
    print(f"Loaded silver_feature_table: {len(df)} rows, {len(df.columns)} columns.")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Tenure buckets ---
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"]
    ).astype(str)  # cast to str for SQLite compatibility

    # --- Monthly charge band ---
    df["monthly_charge_band"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 95, 120],
        labels=["low", "medium", "high", "very_high"]
    ).astype(str)

    # --- Charges per tenure ratio ---
    df["charge_per_tenure"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(4)

    # --- Total active services count ---
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Normalize: map "yes"/1 → 1, everything else → 0
    def is_active(val):
        return 1 if str(val).strip().lower() in ["1", "yes"] else 0

    df["service_count"] = df[service_cols].applymap(is_active).sum(axis=1)

    # --- High-risk payment flag ---
    df["is_electronic_check"] = (
        df["PaymentMethod"].str.strip().str.lower() == "electronic check"
    ).astype(int)

    # --- Month-to-month contract flag ---
    df["is_monthly_contract"] = (
        df["Contract"].str.strip().str.lower() == "month-to-month"
    ).astype(int)

    # --- Senior + monthly contract risk combo ---
    df["senior_monthly_risk"] = (
        (df["SeniorCitizen"] == 1) & (df["is_monthly_contract"] == 1)
    ).astype(int)

    print(f"Feature engineering complete. New shape: {df.shape}")
    print(f"New features: tenure_band, monthly_charge_band, charge_per_tenure, "
          f"service_count, is_electronic_check, is_monthly_contract, senior_monthly_risk")
    return df

def validate_features(df: pd.DataFrame):
    """Sanity checks after feature engineering"""
    assert df["service_count"].between(0, 7).all(), "service_count out of range"
    assert df["is_electronic_check"].isin([0, 1]).all(), "is_electronic_check should be binary"
    assert df["is_monthly_contract"].isin([0, 1]).all(), "is_monthly_contract should be binary"
    assert df["charge_per_tenure"].notnull().all(), "charge_per_tenure has nulls"
    print("Feature validation passed.")

def save_engineered_features(df: pd.DataFrame, conn: sqlite3.Connection):
    conn.execute("DROP TABLE IF EXISTS silver_features_engineered")
    df.to_sql("silver_features_engineered", conn, if_exists="replace", index=False)
    count = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM silver_features_engineered", conn
    )["cnt"][0]
    print(f"silver_features_engineered saved: {count} rows.")

def run():
    conn = sqlite3.connect(DB_PATH)

    # Step 1 — Load Silver feature table
    df = load_silver_feature_table(conn)

    # Step 2 — Engineer features
    df = engineer_features(df)

    # Step 3 — Validate
    validate_features(df)

    # Step 4 — Save to Silver engineered table
    save_engineered_features(df, conn)

    conn.close()
    print("\nFeature engineering complete.")

if __name__ == "__main__":
    run()