import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="IoT Predictive Maintenance Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# 1. DATA CACHING & MODEL TRAINING
# -------------------------------------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect("data/maintenance.db")
    df = pd.read_sql_query("SELECT * FROM machine_telemetry_final", conn)
    conn.close()
    return df

@st.cache_resource
def train_production_model(df):
    feature_cols = [
        "Air_temperature_K",
        "Process_temperature_K",
        "Rotational_speed_rpm",
        "Torque_Nm",
        "Tool_wear_min",
        "Power_Watts",
        "Temp_Diff_K",
        "Strain_Index",
        "Operating_Regime",
    ]
    X = df[feature_cols]
    y = df["Machine_failure"]

    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, class_weight="balanced", random_state=42
    )
    model.fit(X, y)
    return model, feature_cols

df = load_data()
model, feature_cols = train_production_model(df)

# -------------------------------------------------------------
# 2. HEADER & FLEET KPIS
# -------------------------------------------------------------
st.title("Industrial IoT Predictive Maintenance Platform")
st.markdown("Real-time condition monitoring, failure risk scoring, and operational regime analysis.")

total_machines = len(df)
total_failures = int(df["Machine_failure"].sum())
anomalies_detected = int(df["Is_Anomaly"].sum())
high_stress_units = int((df["Operating_Regime"] == 0).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Monitored Fleet", f"{total_machines:,} Units")
col2.metric("Historical Failures", f"{total_failures} ({total_failures/total_machines*100:.2f}%)")
col3.metric("High-Stress Regime Units", f"{high_stress_units:,}")
col4.metric("Active Anomaly Flags", f"{anomalies_detected}")

st.markdown("---")

# -------------------------------------------------------------
# 3. SIDEBAR: REAL-TIME UNIT INFERENCE SIMULATOR
# -------------------------------------------------------------
st.sidebar.header("Machine Telemetry Simulator")
st.sidebar.markdown("Adjust live sensor feeds to evaluate real-time failure probability.")

sim_tier = st.sidebar.selectbox("Machine Tier", ["L", "M", "H"], index=0)
sim_air_temp = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
sim_proc_temp = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
sim_rpm = st.sidebar.slider("Rotational Speed [RPM]", 1100, 2900, 1500, 10)
sim_torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
sim_wear = st.sidebar.slider("Tool Wear [Minutes]", 0, 260, 120, 1)

# Derived Features
sim_power = (2 * np.pi * sim_rpm * sim_torque) / 60
sim_temp_diff = sim_proc_temp - sim_air_temp
sim_strain = sim_torque * sim_wear

# Approximate Operating Regime: 0=High Stress, 1=Nominal, 2=High Speed
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
if prob_failure >= 0.50:
    st.sidebar.error(f"CRITICAL RISK: {prob_failure*100:.1f}% Breakdown Probability")
    st.sidebar.warning("Recommended Action: Halt spindle, inspect tool bit, and check thermal dissipation.")
elif prob_failure >= 0.25:
    st.sidebar.warning(f"ELEVATED RISK: {prob_failure*100:.1f}% Failure Probability")
    st.sidebar.info("Recommended Action: Schedule maintenance check at shift end.")
else:
    st.sidebar.success(f"NOMINAL: {prob_failure*100:.1f}% Failure Probability")

# -------------------------------------------------------------
# 4. VISUAL DIAGNOSTICS PANELS
# -------------------------------------------------------------
tab1, tab2 = st.tabs(["Operational Regimes & Anomalies", "Feature Importance & Forensics"])

with tab1:
    st.subheader("Torque vs. Rotational Speed by Regime")
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
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Model Feature Importance")
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
    st.plotly_chart(fig_imp, use_container_width=True)