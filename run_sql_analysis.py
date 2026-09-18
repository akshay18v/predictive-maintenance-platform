import sqlite3
import pandas as pd

DB_PATH = "data/maintenance.db"
conn = sqlite3.connect(DB_PATH)

def query(sql_stmt: str) -> pd.DataFrame:
    return pd.read_sql_query(sql_stmt, conn)

print("--- Query 1: Failure Rates Across Equipment Tiers ---")
# Business logic: Are budget (L) machines failing at a higher rate than premium (H) variants?
q1 = """
SELECT 
    Type AS machine_tier,
    COUNT(*) AS total_units,
    SUM(Machine_failure) AS total_failures,
    ROUND(AVG(Machine_failure) * 100.0, 2) AS failure_rate_pct
FROM equipment_telemetry
GROUP BY Type
ORDER BY failure_rate_pct DESC;
"""
print(query(q1).to_string(index=False))

print("\n--- Query 2: Physical Stress Averages by Specific Failure Mode ---")
# Business logic: What physical characteristics trigger specific breakdown modes?
q2 = """
SELECT 
    CASE 
        WHEN TWF = 1 THEN 'Tool Wear Failure'
        WHEN HDF = 1 THEN 'Heat Dissipation Failure'
        WHEN PWF = 1 THEN 'Power Failure'
        WHEN OSF = 1 THEN 'Overstrain Failure'
        WHEN RNF = 1 THEN 'Random Failure'
        ELSE 'Nominal (No Failure)'
    END AS operational_status,
    COUNT(*) AS incident_count,
    ROUND(AVG(Torque_Nm), 2) AS avg_torque,
    ROUND(AVG(Rotational_speed_rpm), 2) AS avg_rpm,
    ROUND(AVG(Tool_wear_min), 2) AS avg_tool_wear
FROM equipment_telemetry
GROUP BY operational_status
ORDER BY incident_count DESC;
"""
print(query(q2).to_string(index=False))

print("\n--- Query 3: Extreme Tool Wear Exposure with Zero Failures ---")
# Business logic: Surface units nearing operational limits that have not yet triggered a failure
q3 = """
SELECT 
    UDI, 
    Product_ID, 
    Type, 
    Tool_wear_min, 
    Torque_Nm, 
    Rotational_speed_rpm
FROM equipment_telemetry
WHERE Machine_failure = 0 AND Tool_wear_min > 220
ORDER BY Tool_wear_min DESC
LIMIT 5;
"""
print(query(q3).to_string(index=False))

conn.close()