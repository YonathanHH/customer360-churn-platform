-- Silver feature table
CREATE TABLE IF NOT EXISTS silver_feature_table AS
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
LEFT JOIN silver_churn_label cl ON cp.customerID = cl.customerID;