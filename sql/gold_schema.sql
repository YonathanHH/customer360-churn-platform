-- Gold layer view: final CRM churn action list
CREATE VIEW IF NOT EXISTS v_gold_crm_retention AS
SELECT
    g.customerID,
    cp.gender,
    cp.tenure,
    ab.Contract,
    ab.MonthlyCharges,
    g.churn_probability,
    g.risk_band,
    g.recommended_action
FROM gold_churn_scores g
LEFT JOIN silver_customer_profile cp ON g.customerID = cp.customerID
LEFT JOIN silver_account_billing ab ON g.customerID = ab.customerID
ORDER BY g.churn_probability DESC;