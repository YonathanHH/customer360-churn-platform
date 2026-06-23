import pandas as pd
import sqlite3
import os
from datetime import datetime

RAW_PATH = "data/telco_raw.csv"
DB_PATH = "data/crm_churn.db"

def create_directories():
    """Ensure required folders exist"""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("Directories ready.")

def load_raw_csv(path: str) -> pd.DataFrame:
    """Load raw CSV exactly as-is — no transformations in Bronze"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw file not found at {path}.\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from raw CSV.")
    return df

def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add ingestion metadata columns for traceability"""
    df["_ingested_at"] = datetime.utcnow().isoformat()
    df["_source_file"] = RAW_PATH
    df["_row_id"] = range(1, len(df) + 1)
    return df

def validate_bronze(df: pd.DataFrame):
    """Basic validation checks before writing to Bronze"""
    assert "customerID" in df.columns, "Missing customerID column"
    assert "Churn" in df.columns, "Missing Churn target column"
    assert len(df) > 0, "Empty dataframe — check raw file"

    null_pct = df.isnull().mean() * 100
    high_null_cols = null_pct[null_pct > 50]
    if not high_null_cols.empty:
        print(f"WARNING: Columns with >50% nulls:\n{high_null_cols}")

    print(f"Validation passed: {len(df)} rows, {len(df.columns)} columns.")

def write_bronze(df: pd.DataFrame, db_path: str):
    """Write raw data as-is to Bronze table in SQLite"""
    conn = sqlite3.connect(db_path)
    df.to_sql("bronze_raw_telco", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Bronze table 'bronze_raw_telco' written to {db_path}.")

def log_ingestion(df: pd.DataFrame, db_path: str):
    """Write ingestion log entry for auditability"""
    log_entry = pd.DataFrame([{
        "table_name": "bronze_raw_telco",
        "row_count": len(df),
        "column_count": len(df.columns),
        "ingested_at": datetime.utcnow().isoformat(),
        "source_file": RAW_PATH,
        "status": "success"
    }])
    conn = sqlite3.connect(db_path)
    log_entry.to_sql("bronze_ingestion_log", conn, if_exists="append", index=False)
    conn.close()
    print("Ingestion log updated.")

def run():
    create_directories()
    df = load_raw_csv(RAW_PATH)
    df = add_metadata(df)
    validate_bronze(df)
    write_bronze(df, DB_PATH)
    log_ingestion(df, DB_PATH)
    print("\nBronze ingestion complete.")

if __name__ == "__main__":
    run()