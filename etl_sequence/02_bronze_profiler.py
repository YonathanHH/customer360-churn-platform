import pandas as pd
import sqlite3

DB_PATH = "data/crm_churn.db"

def profile_bronze():
    """Quick data profiling report on the Bronze table"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM bronze_raw_telco", conn)
    conn.close()

    print("=== BRONZE LAYER PROFILE ===")
    print(f"Shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    print(f"\nSample rows:\n{df.head(3)}")
    print(f"\nIngestion timestamp range: "
          f"{df['_ingested_at'].min()} → {df['_ingested_at'].max()}")

if __name__ == "__main__":
    profile_bronze()