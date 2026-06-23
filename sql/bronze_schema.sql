-- Bronze schema: raw table definition for reference

CREATE TABLE IF NOT EXISTS bronze_raw_telco (
    customerID          TEXT,
    gender              TEXT,
    SeniorCitizen       INTEGER,
    Partner             TEXT,
    Dependents          TEXT,
    tenure              INTEGER,
    PhoneService        TEXT,
    MultipleLines       TEXT,
    InternetService     TEXT,
    OnlineSecurity      TEXT,
    OnlineBackup        TEXT,
    DeviceProtection    TEXT,
    TechSupport         TEXT,
    StreamingTV         TEXT,
    StreamingMovies     TEXT,
    Contract            TEXT,
    PaperlessBilling    TEXT,
    PaymentMethod       TEXT,
    MonthlyCharges      REAL,
    TotalCharges        TEXT,   -- stored as TEXT because raw data has spaces
    Churn               TEXT,

    -- Metadata columns
    _ingested_at        TEXT,
    _source_file        TEXT,
    _row_id             INTEGER
);

-- Ingestion log table
CREATE TABLE IF NOT EXISTS bronze_ingestion_log (
    table_name      TEXT,
    row_count       INTEGER,
    column_count    INTEGER,
    ingested_at     TEXT,
    source_file     TEXT,
    status          TEXT
);