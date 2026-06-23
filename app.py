import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CRM Churn Intelligence Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH    = "data/crm_churn.db"
MODEL_PATH = "models/churn_model.pkl"

# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_gold_scores():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM gold_churn_scores", conn)
    conn.close()
    return df

@st.cache_data
def load_silver_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM silver_features_engineered", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

gold    = load_gold_scores()
silver  = load_silver_features()
pipeline = load_model()

# Merge gold scores with silver features for enriched views
df = silver.merge(gold, on="customerID", how="left")

FEATURE_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "service_count", "is_electronic_check", "is_monthly_contract",
    "charge_per_tenure", "senior_monthly_risk"
]

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    width=80
)
st.sidebar.title("🔮 Churn Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Customer Explorer", "🤖 Predict New Customer", "📥 Export"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
risk_filter = st.sidebar.multiselect(
    "Risk Band",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].dropna().unique().tolist(),
    default=df["Contract"].dropna().unique().tolist()
)

# Apply filters
df_filtered = df[
    (df["risk_band"].isin(risk_filter)) &
    (df["Contract"].isin(contract_filter))
]

# ─────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 CRM Churn Intelligence Platform")
    st.markdown("**Medallion Architecture · Bronze → Silver → Gold · XGBoost Pipeline**")
    st.markdown("---")

    # KPI Cards
    total       = len(df_filtered)
    high_risk   = len(df_filtered[df_filtered["risk_band"] == "High"])
    medium_risk = len(df_filtered[df_filtered["risk_band"] == "Medium"])
    low_risk    = len(df_filtered[df_filtered["risk_band"] == "Low"])
    avg_prob    = df_filtered["churn_probability"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers",   f"{total:,}")
    col2.metric("🔴 High Risk",       f"{high_risk:,}",
                f"{high_risk/total*100:.1f}%")
    col3.metric("🟡 Medium Risk",     f"{medium_risk:,}",
                f"{medium_risk/total*100:.1f}%")
    col4.metric("🟢 Low Risk",        f"{low_risk:,}",
                f"{low_risk/total*100:.1f}%")
    col5.metric("Avg Churn Prob",    f"{avg_prob:.2%}")

    st.markdown("---")

    # Row 1 — Risk Distribution + Churn by Contract
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Risk Distribution")
        risk_counts = df_filtered["risk_band"].value_counts().reindex(
            ["High", "Medium", "Low"]
        ).reset_index()
        risk_counts.columns = ["Risk Band", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk Band",
            color="Risk Band",
            color_discrete_map={"High": "tomato", "Medium": "orange", "Low": "mediumseagreen"},
            hole=0.4
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df_filtered.groupby("Contract")["Churn"].mean().reset_index()
        contract_churn["Churn"] = (contract_churn["Churn"] * 100).round(2)
        fig_bar = px.bar(
            contract_churn, x="Contract", y="Churn",
            color="Churn", color_continuous_scale="RdYlGn_r",
            labels={"Churn": "Churn Rate (%)"},
            text="Churn"
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 2 — Tenure vs Churn Probability + Monthly Charges
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Tenure vs Churn Probability")
        fig_scatter = px.scatter(
            df_filtered.sample(min(1000, len(df_filtered))),
            x="tenure", y="churn_probability",
            color="risk_band",
            color_discrete_map={
                "High": "tomato", "Medium": "orange", "Low": "mediumseagreen"
            },
            opacity=0.6,
            labels={"tenure": "Tenure (months)", "churn_probability": "Churn Probability"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_d:
        st.subheader("Monthly Charges by Risk Band")
        fig_box = px.box(
            df_filtered, x="risk_band", y="MonthlyCharges",
            color="risk_band",
            category_orders={"risk_band": ["High", "Medium", "Low"]},
            color_discrete_map={
                "High": "tomato", "Medium": "orange", "Low": "mediumseagreen"
            },
            labels={"risk_band": "Risk Band", "MonthlyCharges": "Monthly Charges (USD)"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2 — CUSTOMER EXPLORER
# ─────────────────────────────────────────
elif page == "🔍 Customer Explorer":
    st.title("🔍 Customer Explorer")
    st.markdown("Browse and search individual customer churn risk profiles.")
    st.markdown("---")

    # Search bar
    search_id = st.text_input("Search by Customer ID", placeholder="e.g. 7590-VHVEG")

    display_cols = [
        "customerID", "tenure", "Contract", "MonthlyCharges",
        "service_count", "churn_probability", "risk_band", "recommended_action"
    ]

    if search_id:
        result = df_filtered[df_filtered["customerID"].str.contains(
            search_id, case=False, na=False
        )][display_cols]
        st.dataframe(result, use_container_width=True)
    else:
        # Sort by churn probability descending
        st.dataframe(
            df_filtered[display_cols].sort_values(
                "churn_probability", ascending=False
            ).reset_index(drop=True),
            use_container_width=True,
            height=500
        )

    st.markdown("---")
    st.subheader("Top 10 Highest Risk Customers")
    top10 = df_filtered[display_cols].sort_values(
        "churn_probability", ascending=False
    ).head(10).reset_index(drop=True)

    fig_top10 = px.bar(
        top10, x="churn_probability", y="customerID",
        orientation="h", color="churn_probability",
        color_continuous_scale="RdYlGn_r",
        labels={"churn_probability": "Churn Probability", "customerID": "Customer ID"},
        text="churn_probability"
    )
    fig_top10.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_top10.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_top10, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3 — PREDICT NEW CUSTOMER
# ─────────────────────────────────────────
elif page == "🤖 Predict New Customer":
    st.title("🤖 Predict Churn for New Customer")
    st.markdown("Enter customer details below to get a real-time churn prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Profile")
        senior         = st.selectbox("Senior Citizen", [0, 1])
        partner        = st.selectbox("Has Partner", [0, 1])
        dependents     = st.selectbox("Has Dependents", [0, 1])
        tenure         = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.subheader("💳 Billing")
        paperless      = st.selectbox("Paperless Billing", [0, 1])
        monthly        = st.number_input("Monthly Charges (USD)", 0.0, 150.0, 65.0)
        total          = st.number_input("Total Charges (USD)", 0.0, 9000.0, monthly * tenure)
        is_elec_check  = st.selectbox("Electronic Check Payment", [0, 1])
        is_monthly     = st.selectbox("Month-to-Month Contract", [0, 1])

    with col3:
        st.subheader("📡 Services")
        service_count  = st.slider("Number of Active Services", 0, 7, 3)
        senior_monthly = int(senior == 1 and is_monthly == 1)
        charge_tenure  = round(total / (tenure + 1), 4)
        st.metric("Charge per Tenure (auto)", f"${charge_tenure:.2f}")
        st.metric("Senior + Monthly Risk (auto)", senior_monthly)

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", type="primary"):
        input_data = pd.DataFrame([{
            "SeniorCitizen":      senior,
            "Partner":            partner,
            "Dependents":         dependents,
            "tenure":             tenure,
            "PaperlessBilling":   paperless,
            "MonthlyCharges":     monthly,
            "TotalCharges":       total,
            "service_count":      service_count,
            "is_electronic_check": is_elec_check,
            "is_monthly_contract": is_monthly,
            "charge_per_tenure":  charge_tenure,
            "senior_monthly_risk": senior_monthly
        }])

        # Pipeline handles scaling + prediction
        prob       = pipeline.predict_proba(input_data)[0][1]
        risk_band  = "High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low"
        action     = {
            "High":   "Offer retention discount immediately",
            "Medium": "Send personalized engagement email",
            "Low":    "No action required"
        }[risk_band]

        color_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Churn Probability", f"{prob:.2%}")
        col_r2.metric("Risk Band", f"{color_map[risk_band]} {risk_band}")
        col_r3.metric("Recommended Action", action)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "tomato" if risk_band == "High"
                         else "orange" if risk_band == "Medium" else "mediumseagreen"},
                "steps": [
                    {"range": [0, 40],  "color": "lightgreen"},
                    {"range": [40, 70], "color": "lightyellow"},
                    {"range": [70, 100],"color": "lightsalmon"}
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": prob * 100
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 4 — EXPORT
# ─────────────────────────────────────────
elif page == "📥 Export":
    st.title("📥 Export CRM Retention List")
    st.markdown("Download the Gold layer churn scores for CRM import.")
    st.markdown("---")

    export_df = df_filtered[[
        "customerID", "tenure", "Contract", "MonthlyCharges",
        "churn_probability", "risk_band", "recommended_action"
    ]].sort_values("churn_probability", ascending=False).reset_index(drop=True)

    st.dataframe(export_df, use_container_width=True, height=400)

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.download_button(
            label="📥 Download Full List (CSV)",
            data=export_df.to_csv(index=False),
            file_name="crm_churn_retention_list.csv",
            mime="text/csv"
        )

    with col_e2:
        high_risk_df = export_df[export_df["risk_band"] == "High"]
        st.download_button(
            label="🔴 Download High Risk Only (CSV)",
            data=high_risk_df.to_csv(index=False),
            file_name="crm_high_risk_customers.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.subheader("Export Summary")
    st.markdown(f"""
    | Segment | Count |
    |---|---|
    | 🔴 High Risk | {len(export_df[export_df['risk_band']=='High']):,} |
    | 🟡 Medium Risk | {len(export_df[export_df['risk_band']=='Medium']):,} |
    | 🟢 Low Risk | {len(export_df[export_df['risk_band']=='Low']):,} |
    | **Total** | **{len(export_df):,}** |
    """)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by Yonathan Hary Hutagalung using **Medallion Architecture**\n\n"
    "Bronze → Silver → Gold → Streamlit"
)