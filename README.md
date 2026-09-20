# Industrial IoT Predictive Maintenance & Condition-Monitoring Platform

An end-to-end industrial telemetry platform designed to predict equipment failures and identify operational anomalies across 10,000 manufacturing units.

---

## Technical Highlights
- **Relational Ingestion & Analysis:** Processed telemetry in SQLite to examine failure distributions across equipment tiers (L, M, H).
- **Statistical Hypothesis Testing:** Statistically validated operational torque spikes prior to breakdown via Welch's t-test ($p = 3.64 \times 10^{-27}$, Cohen's $d = 1.08$) and tier failure dependence via Chi-Square ($\chi^2 = 13.75, p < 0.002$).
- **Physics-Informed Feature Engineering:** Formulated interaction terms including Mechanical Power ($P = \tau \cdot \omega$), Thermal Dissipation ($\Delta T$), and Overstrain Index ($\text{Torque} \times \text{Tool Wear}$) to capture multi-sensor failure mechanics.
- **Unsupervised Regime Discovery:** Applied K-Means clustering ($k=3$) to segment equipment into distinct operational stress regimes, alongside Isolation Forest anomaly detection for zero-day fault catching.
- **Cost-Sensitive Supervised Learning:** Addressed severe 96.6% class imbalance using a Balanced Random Forest, achieving a **0.8115 PR-AUC** and **87% failure recall** (catching 59 of 68 test-set failures).
- **Interactive Triage Dashboard:** Built a responsive Streamlit UI featuring fleet-wide KPIs, interactive sensor simulators, and feature importance diagnostics.

---

## Project Structure
```text
├── data/
│   ├── ai4i2020.csv              # Raw UCI AI4I 2020 dataset
│   └── maintenance.db            # SQLite relational database
├── run_sql_analysis.py           # SQL ingestion and telemetry aggregation
├── 02_pipeline_and_models.py     # Statistical tests, feature engineering, ML models
├── app.py                        # Streamlit condition-monitoring dashboard
├── requirements.txt              # Frozen dependencies
└── README.md                     # Platform documentation
