import sqlite3
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================
# PART 1: LOAD TELEMETRY FROM SQLITE
# =============================================================
print("--- 1. Loading Cleaned Data from SQLite ---")
conn = sqlite3.connect("data/maintenance.db")
df = pd.read_sql_query("SELECT * FROM equipment_telemetry", conn)
conn.close()
print(f"Loaded {len(df)} records from UCI AI4I 2020 dataset.\n")

# =============================================================
# PART 2: STATISTICAL HYPOTHESIS TESTING
# =============================================================
print("--- 2. Running Inferential Hypothesis Tests ---")

healthy_torque = df[df["Machine_failure"] == 0]["Torque_Nm"]
failed_torque = df[df["Machine_failure"] == 1]["Torque_Nm"]

t_stat, p_val_torque = stats.ttest_ind(failed_torque, healthy_torque, equal_var=False)
n1, n2 = len(failed_torque), len(healthy_torque)
s1, s2 = np.var(failed_torque, ddof=1), np.var(healthy_torque, ddof=1)
pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
cohens_d = (failed_torque.mean() - healthy_torque.mean()) / pooled_std

print(f"[Welch's t-test] Torque p-value: {p_val_torque:.4e} | Cohen's d: {cohens_d:.2f}")

contingency = pd.crosstab(df["Type"], df["Machine_failure"])
chi2, p_val_chi2, _, _ = stats.chi2_contingency(contingency)
print(f"[Chi-Square test] Tier vs Failure p-value: {p_val_chi2:.4e} (Chi2 = {chi2:.2f})\n")

# =============================================================
# PART 3: PHYSICS-BASED FEATURE ENGINEERING
# =============================================================
print("--- 3. Engineering Physical & Kinematic Features ---")
df["Power_Watts"] = (2 * np.pi * df["Rotational_speed_rpm"] * df["Torque_Nm"]) / 60
df["Temp_Diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
df["Strain_Index"] = df["Torque_Nm"] * df["Tool_wear_min"]
print("Engineered: 'Power_Watts', 'Temp_Diff_K', 'Strain_Index'.\n")

# =============================================================
# PART 4: K-MEANS REGIMES & ISOLATION FOREST
# =============================================================
print("--- 4. Unsupervised Regimes & Anomaly Detection ---")
regime_cols = ["Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min", "Power_Watts"]
scaler = StandardScaler()
scaled_regimes = scaler.fit_transform(df[regime_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Operating_Regime"] = kmeans.fit_predict(scaled_regimes)

iso_forest = IsolationForest(contamination=0.035, random_state=42)
iso_preds = iso_forest.fit_predict(scaled_regimes)
df["Is_Anomaly"] = np.where(iso_preds == -1, 1, 0)

iso_precision = precision_score(df["Machine_failure"], df["Is_Anomaly"])
iso_recall = recall_score(df["Machine_failure"], df["Is_Anomaly"])
print(f"Isolation Forest (Unsupervised Baseline):")
print(f"  - Precision: {iso_precision:.4f} | Recall: {iso_recall:.4f}")
print(f"  - Lift over random guess (3.4% base rate): ~{iso_precision / 0.034:.1f}x\n")

# =============================================================
# PART 5: 3-WAY SPLIT (TRAIN: 70%, VAL: 15%, TEST: 15%)
# =============================================================
raw_sensor_cols = [
    "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
    "Torque_Nm", "Tool_wear_min"
]
all_feature_cols = raw_sensor_cols + [
    "Power_Watts", "Temp_Diff_K", "Strain_Index", "Operating_Regime"
]

X = df[all_feature_cols]
y = df["Machine_failure"]

# Split 1: 70% Train, 30% Temp (Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Split 2: Divide the 30% equally into 15% Validation and 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Split sizes -> Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# =============================================================
# PART 6: REAL BASELINE VS. PHYSICS-ENRICHED MODEL
# =============================================================
print("\n--- 5. Training Models & Benchmarking ---")

# Baseline Model: Trained on RAW sensor features only
rf_baseline = RandomForestClassifier(
    n_estimators=150, max_depth=8, class_weight="balanced", random_state=42
)
rf_baseline.fit(X_train[raw_sensor_cols], y_train)
val_probs_base = rf_baseline.predict_proba(X_val[raw_sensor_cols])[:, 1]
p_base, r_base, _ = precision_recall_curve(y_val, val_probs_base)
baseline_pr_auc = auc(r_base, p_base)

# Production Model: Trained on RAW + PHYSICS + REGIME features
rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=8, class_weight="balanced", random_state=42
)
rf_model.fit(X_train, y_train)
val_probs = rf_model.predict_proba(X_val)[:, 1]
p_rf, r_rf, _ = precision_recall_curve(y_val, val_probs)
production_pr_auc = auc(r_rf, p_rf)

print(f"Baseline (Raw Sensors Only) PR-AUC:       {baseline_pr_auc:.4f}")
print(f"Production (+ Physics Features) PR-AUC:   {production_pr_auc:.4f}")
print(f"Proven Feature Engineering Lift:          +{production_pr_auc - baseline_pr_auc:.4f}")

# =============================================================
# PART 7: THRESHOLD TUNING ON VALIDATION DATA (NO LEAKAGE)
# =============================================================
print("\n--- 6. Tuning Cost Threshold on Validation Set ---")
COST_FN = 1000  # $1,000 for missed breakdown
COST_FP = 100   # $100 for false alarm inspection

threshold_candidates = np.linspace(0.10, 0.90, 81)
val_costs = []

for t in threshold_candidates:
    preds = (val_probs >= t).astype(int)
    fn = np.sum((y_val == 1) & (preds == 0))
    fp = np.sum((y_val == 0) & (preds == 1))
    val_costs.append((fn * COST_FN) + (fp * COST_FP))

best_idx = np.argmin(val_costs)
optimal_threshold = float(threshold_candidates[best_idx])
print(f"Selected Optimal Threshold on Validation Set: {optimal_threshold:.2f}")

# =============================================================
# PART 8: HONEST FINAL EVALUATION ON UNSEEN TEST SET
# =============================================================
print("\n--- 7. Final Unbiased Evaluation on Test Set ---")
test_probs = rf_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= optimal_threshold).astype(int)

cm = confusion_matrix(y_test, test_preds)
test_precision = precision_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)

test_cost = (cm[1][0] * COST_FN) + (cm[0][1] * COST_FP)
default_preds = (test_probs >= 0.50).astype(int)
cm_default = confusion_matrix(y_test, default_preds)
default_cost = (cm_default[1][0] * COST_FN) + (cm_default[0][1] * COST_FP)

print(f"Test Confusion Matrix at Locked Threshold {optimal_threshold:.2f}:")
print(f"  True Negatives (Safe):         {cm[0][0]}")
print(f"  False Positives (Inspections): {cm[0][1]}")
print(f"  False Negatives (Missed):      {cm[1][0]}")
print(f"  True Positives (Caught):       {cm[1][1]}")
print(f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}")
print(f"Total Incurred Cost on Test Set: ${test_cost:,} (vs. ${default_cost:,} at default 0.50)")

# =============================================================
# PART 9: SAVE ARTIFACTS
# =============================================================
artifacts = {
    "model": rf_model,
    "feature_cols": all_feature_cols,
    "optimal_threshold": optimal_threshold,
}
joblib.dump(artifacts, "model_artifact.joblib")
print("\nSaved serialized model bundle to 'model_artifact.joblib'.")

conn = sqlite3.connect("data/maintenance.db")
df.to_sql("machine_telemetry_final", conn, if_exists="replace", index=False)
conn.close()
print("Saved enriched telemetry to SQLite table 'equipment_telemetry_final'.")