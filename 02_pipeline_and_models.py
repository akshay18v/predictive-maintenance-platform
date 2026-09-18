import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================
# PART 1: LOAD TELEMETRY FROM SQLITE
# =============================================================
print("--- 1. Loading Cleaned Data from SQLite ---")
conn = sqlite3.connect("data/maintenance.db")
df = pd.read_sql_query("SELECT * FROM equipment_telemetry", conn)
conn.close()
print(f"Loaded {len(df)} records.\n")

# =============================================================
# PART 2: STATISTICAL HYPOTHESIS TESTING
# =============================================================
print("--- 2. Running Inferential Hypothesis Tests ---")

# A. Welch's t-test on Torque (Continuous Variable)
healthy_torque = df[df["Machine_failure"] == 0]["Torque_Nm"]
failed_torque = df[df["Machine_failure"] == 1]["Torque_Nm"]

t_stat, p_val_torque = stats.ttest_ind(failed_torque, healthy_torque, equal_var=False)

# Cohen's d calculation for practical effect size
n1, n2 = len(failed_torque), len(healthy_torque)
s1, s2 = np.var(failed_torque, ddof=1), np.var(healthy_torque, ddof=1)
pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
cohens_d = (failed_torque.mean() - healthy_torque.mean()) / pooled_std

print(f"[Welch's t-test] Torque p-value: {p_val_torque:.4e} | Cohen's d: {cohens_d:.2f}")
if p_val_torque < 0.05:
    print(">> Result: Statistically significant. Torque shift prior to failure is real.")

# B. Chi-Square Test on Equipment Tier (Categorical Variable)
contingency = pd.crosstab(df["Type"], df["Machine_failure"])
chi2, p_val_chi2, dof, _ = stats.chi2_contingency(contingency)
print(f"[Chi-Square test] Tier vs Failure p-value: {p_val_chi2:.4e} (Chi2 = {chi2:.2f})")
if p_val_chi2 < 0.05:
    print(">> Result: Failure rate is statistically dependent on equipment tier (L vs M vs H).\n")

# =============================================================
# PART 3: PHYSICS-BASED FEATURE ENGINEERING
# =============================================================
print("--- 3. Engineering Physical & Kinematic Features ---")

# Mechanical Power in Watts = (2 * pi * RPM * Torque) / 60
df["Power_Watts"] = (2 * np.pi * df["Rotational_speed_rpm"] * df["Torque_Nm"]) / 60

# Temperature Differential = Process Temp - Ambient Temp
df["Temp_Diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]

# Overstrain Index = Torque * Tool Wear (Interaction term that prevents false alarms)
df["Strain_Index"] = df["Torque_Nm"] * df["Tool_wear_min"]

print("Engineered: 'Power_Watts', 'Temp_Diff_K', 'Strain_Index'.\n")

# =============================================================
# PART 4: UNSUPERVISED K-MEANS REGIMES
# =============================================================
print("--- 4. Segmenting Factory Operating Regimes (K-Means) ---")
regime_cols = ["Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min", "Power_Watts"]
scaler = StandardScaler()
scaled_regimes = scaler.fit_transform(df[regime_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Operating_Regime"] = kmeans.fit_predict(scaled_regimes)

regime_summary = df.groupby("Operating_Regime")[["Torque_Nm", "Rotational_speed_rpm", "Tool_wear_min", "Machine_failure"]].mean()
regime_summary["Units"] = df["Operating_Regime"].value_counts()
regime_summary["Failure_Rate_%"] = (regime_summary["Machine_failure"] * 100).round(2)
print(regime_summary[["Units", "Torque_Nm", "Rotational_speed_rpm", "Failure_Rate_%"]].to_string())

print("\n")

# =============================================================
# PART 5: UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
# =============================================================
print("--- 5. Training Isolation Forest (Zero-Day Anomaly Detection) ---")
# Contamination is set close to the known failure rate (~3.5%)
iso_forest = IsolationForest(contamination=0.035, random_state=42)
iso_preds = iso_forest.fit_predict(scaled_regimes)
df["Is_Anomaly"] = np.where(iso_preds == -1, 1, 0)

overlap = df[(df["Is_Anomaly"] == 1) & (df["Machine_failure"] == 1)]
print(f"Isolation Forest flagged {df['Is_Anomaly'].sum()} anomalies.")
print(f"Direct overlap with actual failures: {len(overlap)} / {df['Machine_failure'].sum()} units.\n")

# =============================================================
# PART 6: SUPERVISED IMBALANCED CLASSIFIER (RANDOM FOREST)
# =============================================================
print("--- 6. Training Balanced Random Forest Classifier ---")


feature_cols = [
    "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
    "Torque_Nm", "Tool_wear_min", "Power_Watts", "Temp_Diff_K", 
    "Strain_Index", "Operating_Regime"
]

X = df[feature_cols]
y = df["Machine_failure"]

# Stratified split preserves the 3.4% failure proportion in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Balanced class weights penalize missed breakdowns heavily
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

print("\n--- Model Evaluation (Test Set: 2,000 Machines) ---")
print(f"PR-AUC (Precision-Recall Area Under Curve): {pr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Nominal (0)", "Failure (1)"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Safe):         {cm[0][0]}")
print(f"False Positives (False Alarm): {cm[0][1]}")
print(f"False Negatives (Missed Break):{cm[1][0]}  <-- Critical to minimize")
print(f"True Positives (Caught Break):  {cm[1][1]}")

# =============================================================
# PART 7: EXPORT ENRICHED DATA FOR DASHBOARD
# =============================================================
conn = sqlite3.connect("data/maintenance.db")
df.to_sql("machine_telemetry_final", conn, if_exists="replace", index=False)
conn.close()
print("\nFinal enriched telemetry saved to SQLite table 'machine_telemetry_final'. Ready for Streamlit!")