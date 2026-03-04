# UK Telecom Cease Risk Prediction

A production-style machine learning project to help **UK Telecoms LTD. prioritise retention outreach** by predicting which customers are most likely to **place a cease within the next 30 days**. The solution outputs a **risk score**, assigns **High/Medium/Low** risk bands aligned to outreach capacity, and recommends **actionable interventions**.

**Live Streamlit App:** https://uk-telecom-cease-prediction.streamlit.app/

---

## Business Objective

The business wants to prioritise limited retention resources (e.g., outbound calls) by focusing on customers most likely to leave. This project:
- Predicts cease risk at a monthly customer snapshot level
- Creates capacity-based risk segments (High/Medium/Low)
- Produces an operational call list and recommended next-best actions

---

## Target Definition

**Target:** `target_30d = 1` if a customer places a cease in the **next 30 days** following a snapshot date (`datevalue`), otherwise `0`.

- **Snapshot date:** monthly (1st of each month)
- **Leakage prevention:** features are computed only from data **up to** the snapshot date

---

## Data Overview

Inputs include:
- `customer_info` (monthly snapshot features)
- `usage` (daily usage signals)
- `calls` (customer contact history)
- `cease` (label source)

Large parquet sources are processed with **DuckDB** to keep memory usage stable (8GB RAM friendly).

---

## Methodology (High Level)

1. **Feature Engineering (DuckDB)**
   - Usage and calls aggregated over the **previous 30 days**
   - Categorical text standardised (`strip()`) to prevent label drift
   - Missingness handled meaningfully (e.g., no calls ⇒ avg durations = 0)

2. **Data Quality**
   - Deduplication: enforce one row per `(customer, snapshot_date)`
   - Time-based splits: train/val/test with no overlap

3. **Modeling**
   - Baseline: scaled Logistic Regression
   - Final: HistGradientBoostingClassifier (best validation PR-AUC)
   - Metric focus: **PR-AUC** (high class imbalance)

4. **Business Segmentation**
   - Capacity-based bands (example):
     - **High:** top 5% risk (call list)
     - **Medium:** next 15% (low-cost interventions)
     - **Low:** remaining 80% (monitor)
   - KPIs reported: **lift** and **capture** by risk band

5. **Deployment**
   - Streamlit app supports:
     - Single customer scoring (top drivers form)
     - Batch scoring via CSV upload + downloadable output

---

## Results Summary (What to Expect)

The project is designed to show clear business value via:
- **High lift** in cease rate for the High band vs baseline
- **High capture** of total ceases using limited call capacity
- Reason-based recommended interventions for the retention team

(Exact figures depend on the final run; the app and notebooks display the latest metrics.)

---

## Repository Structure

- `notebooks/` — end-to-end pipeline + business solution notebooks  
- `src/` — reusable modules (feature logic, utilities)  
- `app/` — Streamlit application  
- `models/` — trained model artifact + metadata (used by Streamlit)  
- `reports/` — exported call lists / KPI tables (optional)  

---

## Notes on Reproducibility
Risk band thresholds are derived on validation and stored in models/metadata.json

Categorical dropdown options are also stored in metadata to avoid category mismatch in UI

Logs and transforms (e.g., log1p_*) are applied consistently at inference time

## License / Disclaimer
This repository is for assessment and demonstration purposes. Any resemblance to real customer identifiers is incidental; data is treated as confidential and is not included in this repository.