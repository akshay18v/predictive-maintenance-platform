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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================
# 0. GLOBAL REPRODUCIBILITY SEED
# =============================================================
SEED = 42
np.random.seed(SEED)

COST_FN = 1000  # $1,000 for missed breakdown (Unplanned downtime)
COST_FP = 100   # $100 for false alarm inspection (Labor cost)

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
# PART 4: K-MEANS REGIMES & ISOLATION FOREST (SEEDED)
# =============================================================
print("--- 4. Unsupervised Regimes & Anomaly Detection ---")
regime_cols = ["Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min", "Power_Watts"]
scaler = StandardScaler()
scaled_regimes = scaler.fit_transform(df[regime_cols])

kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
df["Operating_Regime"] = kmeans.fit_predict(scaled_regimes)

iso_forest = IsolationForest(contamination=0.034, random_state=SEED)
iso_preds = iso_forest.fit_predict(scaled_regimes)
df["Is_Anomaly"] = np.where(iso_preds == -1, 1, 0)

iso_precision = precision_score(df["Machine_failure"], df["Is_Anomaly"])
iso_recall = recall_score(df["Machine_failure"], df["Is_Anomaly"])
print(f"Isolation Forest (Unsupervised Baseline):")
print(f"  - Precision: {iso_precision:.4f} | Recall: {iso_recall:.4f}")
print(f"  - Lift over base rate (3.39%): ~{iso_precision / 0.0339:.1f}x\n")

# =============================================================
# PART 5: UNBIASED SPLIT (85% DEV, 15% HELD-OUT TEST)
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

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

print(f"Dataset split -> Development: {len(X_dev)} records ({y_dev.sum()} failures)")
print(f"                 Test (Held-Out): {len(X_test)} records ({y_test.sum()} failures)\n")

# =============================================================
# PART 6: STRATIFIED 5-FOLD CV (OOF BENCHMARKING)
# =============================================================
print("--- 5. Cross-Validated Training & Threshold Optimization ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_probs_baseline = np.zeros(len(X_dev))
oof_probs_prod = np.zeros(len(X_dev))

for train_idx, val_idx in skf.split(X_dev, y_dev):
    X_f_train, X_f_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
    y_f_train, y_f_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
    
    # Baseline Model: Raw Sensors Only
    rf_b = RandomForestClassifier(
        n_estimators=150, max_depth=8, class_weight="balanced", random_state=SEED
    )
    rf_b.fit(X_f_train[raw_sensor_cols], y_f_train)
    oof_probs_baseline[val_idx] = rf_b.predict_proba(X_f_val[raw_sensor_cols])[:, 1]
    
    # Production Model: Raw + Physics
    rf_p = RandomForestClassifier(
        n_estimators=150, max_depth=8, class_weight="balanced", random_state=SEED
    )
    rf_p.fit(X_f_train, y_f_train)
    oof_probs_prod[val_idx] = rf_p.predict_proba(X_f_val)[:, 1]

# PR-AUC Calculation
p_base, r_base, _ = precision_recall_curve(y_dev, oof_probs_baseline)
baseline_pr_auc = auc(r_base, p_base)

p_prod, r_prod, _ = precision_recall_curve(y_dev, oof_probs_prod)
prod_pr_auc = auc(r_prod, p_prod)

print(f"5-Fold OOF Baseline PR-AUC (Raw Sensors):      {baseline_pr_auc:.4f}")
print(f"5-Fold OOF Production PR-AUC (+ Physics):       {prod_pr_auc:.4f}")
print(f"Verified Feature Engineering Lift:             +{prod_pr_auc - baseline_pr_auc:.4f}\n")

# =============================================================
# PART 7: EXPLICIT COST MATRIX THRESHOLD SEARCH
# =============================================================
print(f"Running Threshold Optimization (FN Cost: ${COST_FN}, FP Cost: ${COST_FP})...")

threshold_candidates = np.linspace(0.10, 0.90, 81)
oof_costs = []

for t in threshold_candidates:
    preds = (oof_probs_prod >= t).astype(int)
    fn = int(np.sum((y_dev == 1) & (preds == 0)))
    fp = int(np.sum((y_dev == 0) & (preds == 1)))
    cost = (fn * COST_FN) + (fp * COST_FP)
    oof_costs.append(cost)

# Compute default 0.50 cost on OOF
default_oof_preds = (oof_probs_prod >= 0.50).astype(int)
default_fn_oof = int(np.sum((y_dev == 1) & (default_oof_preds == 0)))
default_fp_oof = int(np.sum((y_dev == 0) & (default_oof_preds == 1)))
default_oof_cost = (default_fn_oof * COST_FN) + (default_fp_oof * COST_FP)

best_idx = np.argmin(oof_costs)
optimal_threshold = float(threshold_candidates[best_idx])
optimal_oof_cost = oof_costs[best_idx]

print(f"OOF Cost at Default 0.50: ${default_oof_cost:,} (FN: {default_fn_oof}, FP: {default_fp_oof})")
print(f"OOF Cost at Tuned {optimal_threshold:.2f}:   ${optimal_oof_cost:,} (Savings on Dev: ${default_oof_cost - optimal_oof_cost:,})")

# =============================================================
# PART 8: FINAL PRODUCTION MODEL & FROZEN TEST EVALUATION
# =============================================================
print("\n--- 6. Final Blind Test Set Evaluation ---")
final_model = RandomForestClassifier(
    n_estimators=150, max_depth=8, class_weight="balanced", random_state=SEED
)
final_model.fit(X_dev, y_dev)

test_probs = final_model.predict_proba(X_test)[:, 1]

# 1. Performance at Tuned Threshold
test_preds = (test_probs >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, zero_division=0)
test_recall = recall_score(y_test, test_preds, zero_division=0)
test_cost = int((cm[1][0] * COST_FN) + (cm[0][1] * COST_FP))

# 2. Performance at Default 0.50 Threshold
default_preds = (test_probs >= 0.50).astype(int)
cm_default = confusion_matrix(y_test, default_preds)
default_precision = precision_score(y_test, default_preds, zero_division=0)
default_recall = recall_score(y_test, default_preds, zero_division=0)
default_cost = int((cm_default[1][0] * COST_FN) + (cm_default[0][1] * COST_FP))

print(f"Test Evaluation at Tuned Threshold ({optimal_threshold:.2f}):")
print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
print(f"  Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")
print(f"  Incurred Cost: ${test_cost:,}")

print(f"\nTest Evaluation at Default Threshold (0.50):")
print(f"  Confusion Matrix: TN={cm_default[0][0]}, FP={cm_default[0][1]}, FN={cm_default[1][0]}, TP={cm_default[1][1]}")
print(f"  Precision: {default_precision:.4f} | Recall: {default_recall:.4f}")
print(f"  Incurred Cost: ${default_cost:,}")

# =============================================================
# PART 9: SAVE PRODUCTION ARTIFACTS
# =============================================================
artifacts = {
    "model": final_model,
    "feature_cols": all_feature_cols,
    "optimal_threshold": optimal_threshold,
    "baseline_pr_auc": baseline_pr_auc,
    "prod_pr_auc": prod_pr_auc,
    "test_metrics": {
        "precision": test_precision,
        "recall": test_recall,
        "cm": cm.tolist(),
        "cost": test_cost,
        "default_cost": default_cost
    }
}
joblib.dump(artifacts, "model_artifact.joblib")
print("\n[SUCCESS] Saved serialized production model to 'model_artifact.joblib'")

conn = sqlite3.connect("data/maintenance.db")
df.to_sql("machine_telemetry_final", conn, if_exists="replace", index=False)
conn.close()
print("[SUCCESS] Enriched telemetry saved to SQLite table 'machine_telemetry_final'.")