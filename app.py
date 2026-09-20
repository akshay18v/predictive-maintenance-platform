import sqlite3
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="IoT Predictive Maintenance Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# 1. LOAD DATA & SERIALIZED ARTIFACTS (NO IN-APP RETRAINING)
# -------------------------------------------------------------
@st.cache_data
def load_telemetry_data():
    conn = sqlite3.connect("data/maintenance.db")
    df = pd.read_sql_query("SELECT * FROM machine_telemetry_final", conn)
    conn.close()
    return df

@st.cache_resource
def load_production_artifacts():
    artifacts = joblib.load("model_artifact.joblib")
    return artifacts

df = load_telemetry_data()
artifacts = load_production_artifacts()

model = artifacts["model"]
feature_cols = artifacts["feature_cols"]
optimal_threshold = artifacts.get("optimal_threshold", 0.31)

# -------------------------------------------------------------
# 2. HEADER & FLEET-WIDE KPIS
# -------------------------------------------------------------
st.title("Industrial IoT Predictive Maintenance Platform")
st.markdown(
    "Real-time condition monitoring, cost-calibrated failure risk scoring, "
    "and physics-informed diagnostics across fleet telemetry."
)

total_machines = len(df)
total_failures = int(df["Machine_failure"].sum())
anomalies_detected = int(df["Is_Anomaly"].sum())
high_stress_units = int((df["Operating_Regime"] == 0).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Monitored Fleet", f"{total_machines:,} Records")
col2.metric("Historical Failures", f"{total_failures} ({total_failures/total_machines*100:.2f}%)")
col3.metric("High-Stress Units", f"{high_stress_units:,}")
col4.metric("Unsupervised Anomalies", f"{anomalies_detected}")

st.markdown("---")

# -------------------------------------------------------------
# 3. SIDEBAR: REAL-TIME INFERENCE & THRESHOLD GATE
# -------------------------------------------------------------
st.sidebar.header("Machine Telemetry Simulator")
st.sidebar.markdown("Adjust live sensor values to evaluate condition risk in real-time.")

sim_tier = st.sidebar.selectbox("Equipment Quality Tier", ["L", "M", "H"], index=0)
sim_air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
sim_proc_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
sim_rpm = st.sidebar.slider("Rotational Speed [RPM]", 1100, 2900, 1500, 10)
sim_torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
sim_wear = st.sidebar.slider("Tool Wear [Minutes]", 0, 260, 120, 1)

# Derive physical kinematic features from live inputs
sim_power = (2 * np.pi * sim_rpm * sim_torque) / 60
sim_temp_diff = sim_proc_temp - sim_air_temp
sim_strain = sim_torque * sim_wear

# Approximate operational regime assignment: 0=High Stress, 1=Nominal, 2=High Speed
if sim_torque > 45 or sim_power > 6000:
    sim_regime = 0
elif sim_rpm > 1750:
    sim_regime = 2
else:
    sim_regime = 1

input_payload = pd.DataFrame(
    [[
        sim_air_temp,
        sim_proc_temp,
        sim_rpm,
        sim_torque,
        sim_wear,
        sim_power,
        sim_temp_diff,
        sim_strain,
        sim_regime,
    ]],
    columns=feature_cols,
)

prob_failure = model.predict_proba(input_payload)[0][1]

st.sidebar.markdown("---")
st.sidebar.subheader("Diagnostic Assessment")
st.sidebar.caption(f"Economically calibrated decision threshold: **{optimal_threshold:.2f}**")

if prob_failure >= optimal_threshold:
    st.sidebar.error(f"CRITICAL RISK: {prob_failure*100:.1f}% Breakdown Probability")
    st.sidebar.warning(
        f"Threshold Exceeded (>{optimal_threshold:.2f}). Triggering preventative inspection. "
        "Estimated cost avoidance: $1,000 failure vs. $100 inspection."
    )
    st.sidebar.info("Action: Stop spindle, check cutter blade wear, and verify coolant flow.")
elif prob_failure >= (optimal_threshold * 0.70):
    st.sidebar.warning(f"ELEVATED RISK: {prob_failure*100:.1f}% Failure Probability")
    st.sidebar.info("Action: Log machine state; queue for inspection at upcoming shift change.")
else:
    st.sidebar.success(f"NOMINAL: {prob_failure*100:.1f}% Failure Probability")
    st.sidebar.caption("Unit operating within acceptable physical safety margins.")

# -------------------------------------------------------------
# 4. VISUAL DIAGNOSTICS & SYSTEM BENCHMARKS
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Operational Regimes & Anomalies", 
    "Feature Importance & Forensics",
    "Model Architecture & Test Validation"
])

with tab1:
    st.subheader("Operational Stress Map: Torque vs. Spindle Speed")
    regime_labels = {0: "0: High Stress", 1: "1: Nominal", 2: "2: High Speed"}
    plot_df = df.copy()
    plot_df["Regime_Label"] = plot_df["Operating_Regime"].map(regime_labels)

    fig = px.scatter(
        plot_df,
        x="Rotational_speed_rpm",
        y="Torque_Nm",
        color="Regime_Label",
        symbol="Machine_failure",
        opacity=0.6,
        color_discrete_sequence=["#EF553B", "#00CC96", "#AB63FA"],
        labels={"Rotational_speed_rpm": "Spindle Speed (RPM)", "Torque_Nm": "Torque (Nm)"},
    )
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("Random Forest Gini Feature Importance")
    importances = pd.DataFrame(
        {"Feature": feature_cols, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        importances,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_imp, width="stretch")

with tab3:
    st.subheader("Unbiased Test Set Validation Metrics")
    st.markdown(
        "Evaluated on an independent, held-out test split of **1,500 unseen records** "
        "using a frozen decision threshold tuned solely on validation data."
    )
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Production PR-AUC", "0.7968", "+0.1311 Lift vs. Raw Sensors")
    m_col2.metric("Test Failure Recall", "92.16%", "47 / 51 Breakdowns Prevented")
    m_col3.metric("Test Precision", "38.52%", "11.3x Lift over 3.4% Base Rate")
    
    st.markdown("### Production Cost-Benefit Comparison")
    st.table(pd.DataFrame({
        "Model Configuration": [
            "Baseline (Raw Sensors Only)", 
            "Production (+ Physics Features)",
            "Unsupervised Anomaly (Isolation Forest)"
        ],
        "Input Features": [
            "5 Raw Sensors (RPM, Torque, Temps, Wear)",
            "Raw Sensors + Power, Delta T, Strain Index, Regime",
            "Speed, Torque, Wear, Power"
        ],
        "PR-AUC / Precision": ["0.6656 PR-AUC", "0.7968 PR-AUC", "31.14% Precision (9.2x Base Lift)"],
        "Role in Platform": [
            "Benchmark floor", 
            "Primary automated dispatch model", 
            "Zero-label cold-start detector"
        ]
    }))