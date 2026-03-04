import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="UK Telecom Cease Risk", layout="wide")


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cease_risk_model.joblib"
META_PATH = PROJECT_ROOT / "models" / "metadata.json"


# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds deterministic log1p transforms used during training.
    Safe to run at inference.
    """
    df = df.copy()
    for col in ["sum_download_30d", "sum_upload_30d", "avg_talk_time_30d", "avg_hold_time_30d"]:
        if col in df.columns:
            # clip to avoid negatives; log1p(0)=0 OK
            df[f"log1p_{col}"] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))
    return df


def assign_band(score: float, q_high: float, q_medium: float) -> str:
    if score >= q_high:
        return "High"
    if score >= q_medium:
        return "Medium"
    return "Low"


ACTION_MAP = {
    "High":   "Outbound call within 48h + tailored retention offer/support",
    "Medium": "Targeted email/SMS + self-serve retention options; monitor for escalation",
    "Low":    "No proactive outreach; monitor monthly and include in general comms",
}


def make_template(feature_cols: list[str]) -> pd.DataFrame:
    """
    Create a CSV template with the required feature columns.
    Uses sensible defaults; user can overwrite.
    """
    row = {}
    for c in feature_cols:
        # crude defaulting: strings -> "Unknown", numbers -> 0
        if any(k in c for k in ["status", "technology", "channel", "package", "name"]):
            row[c] = "Unknown"
        else:
            row[c] = 0
    return pd.DataFrame([row])


def ensure_features(df: pd.DataFrame, feature_cols: list[str], fill_defaults: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Ensures df has all required model feature columns.
    - If fill_defaults=True, missing features are created with defaults (0 / "Unknown").
    - Returns (df_aligned, missing_features_list).
    """
    df = df.copy()
    missing = sorted(set(feature_cols) - set(df.columns))

    if missing and fill_defaults:
        for c in missing:
            if any(k in c for k in ["status", "technology", "channel", "package", "name"]):
                df[c] = "Unknown"
            else:
                df[c] = 0

    # Align column order
    still_missing = sorted(set(feature_cols) - set(df.columns))
    return df, still_missing


def score_dataframe(df: pd.DataFrame, model, feature_cols: list[str], q_high: float, q_medium: float) -> pd.DataFrame:
    """
    Adds risk_score, risk_band, recommended_action.
    """
    df = add_log_features(df)
    proba = model.predict_proba(df[feature_cols])[:, 1]
    out = df.copy()
    out["risk_score"] = proba
    out["risk_band"] = [assign_band(float(p), q_high, q_medium) for p in proba]
    out["recommended_action"] = out["risk_band"].map(ACTION_MAP)
    return out


# ----------------------------
# App header + artifact checks
# ----------------------------
st.title("UK Telecom Cease Risk Prediction")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "Model artifacts not found.\n\n"
        "Expected:\n"
        f"- {MODEL_PATH}\n"
        f"- {META_PATH}\n"
    )
    st.stop()

model, meta = load_artifacts()

feature_cols: list[str] = meta.get("features", [])
thresholds = meta.get("band_thresholds_from_val", {})

# Use saved thresholds if present; otherwise fallback to defaults
q_high = float(thresholds.get("q_high", thresholds.get("q95", 0.95)))
q_medium = float(thresholds.get("q_medium", thresholds.get("q80", 0.80)))

# Caption bar
st.caption(
    f"Model: {meta.get('model_name', 'Unknown')} | "
    f"Trained at: {meta.get('trained_at', 'Unknown')} | "
    f"Lookback: {meta.get('lookback_days', '?')}d | Horizon: {meta.get('horizon_days', '?')}d"
)

# ----------------------------
# Top menu (no sidebar)
# ----------------------------
menu = st.radio(
    "Menu",
    ["Overview", "Single prediction", "Batch scoring", "Model & Features"],
    horizontal=True,
    label_visibility="collapsed",
)

# ----------------------------
# Overview
# ----------------------------
if menu == "Overview":
    st.subheader("What this app does")
    st.write(
        "This tool scores customers for the probability of placing a cease in the next 30 days. "
        "It segments customers into **High / Medium / Low** risk bands to help retention teams prioritise outreach."
    )

    # Show business KPIs if present in metadata
    kpi_val = meta.get("business_kpis_val", {})
    kpi_test = meta.get("business_kpis_test", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Val call-list size", str(kpi_val.get("call_list_size", "—")))
    c2.metric("Val precision", f"{kpi_val.get('precision', float('nan')):.3f}" if "precision" in kpi_val else "—")
    c3.metric("Val recall", f"{kpi_val.get('recall', float('nan')):.3f}" if "recall" in kpi_val else "—")
    c4.metric("Val PR-AUC", f"{meta.get('val_pr_auc', float('nan')):.3f}" if "val_pr_auc" in meta else "—")

    st.divider()

    st.subheader("Risk band definitions")
    st.write(
        f"- **High**: risk_score ≥ {q_high:.4f} (top ~5%)\n"
        f"- **Medium**: {q_medium:.4f} ≤ risk_score < {q_high:.4f} (next ~15%)\n"
        f"- **Low**: risk_score < {q_medium:.4f} (remaining ~80%)"
    )

    st.subheader("Recommended interventions")
    st.write(
        "- **High**: outbound call in 48h + retention offer/support\n"
        "- **Medium**: targeted email/SMS + self-serve retention options\n"
        "- **Low**: monitor + general comms"
    )

# ----------------------------
# Single prediction
# ----------------------------
elif menu == "Single prediction":
    st.subheader("Single customer scoring")

    st.info(
        "This form is business-friendly and fills any missing model features with defaults. "
        "For full control, use **Batch scoring** with the template."
    )

    # We keep the single form minimal but robust; missing features are defaulted.
    col1, col2, col3 = st.columns(3)

    with col1:
        contract_status = st.selectbox("contract_status", ["In Contract", "Out of Contract", "Unknown"], index=2)
        technology = st.selectbox("technology", ["FTTP", "FTTC", "ADSL", "Unknown"], index=3)
        sales_channel = st.selectbox("sales_channel", ["Online", "Retail", "Phone", "Partner", "Unknown"], index=4)
        crm_package_name = st.text_input("crm_package_name", value="Unknown")

    with col2:
        tenure_days = st.number_input("tenure_days", min_value=0, value=365)
        ooc_days = st.number_input("ooc_days", min_value=0, value=0)
        dd_cancel_60_day = st.number_input("dd_cancel_60_day (0/1)", min_value=0, max_value=1, value=0)
        contract_dd_cancels = st.number_input("contract_dd_cancels", min_value=0, value=0)

    with col3:
        speed = st.number_input("speed", min_value=0, value=100)
        line_speed = st.number_input("line_speed", min_value=0.0, value=100.0)
        calls_30d = st.number_input("calls_30d", min_value=0, value=0)
        loyalty_calls_30d = st.number_input("loyalty_calls_30d", min_value=0, value=0)

    st.markdown("Usage (30d)")
    u1, u2, u3, u4 = st.columns(4)
    with u1:
        avg_download_30d = st.number_input("avg_download_30d", min_value=0.0, value=0.0)
    with u2:
        avg_upload_30d = st.number_input("avg_upload_30d", min_value=0.0, value=0.0)
    with u3:
        sum_download_30d = st.number_input("sum_download_30d", min_value=0.0, value=0.0)
    with u4:
        sum_upload_30d = st.number_input("sum_upload_30d", min_value=0.0, value=0.0)

    # Build row with defaults for all model features
    base = make_template(feature_cols).iloc[0].to_dict()
    base.update({
        "contract_status": contract_status,
        "technology": technology,
        "sales_channel": sales_channel,
        "crm_package_name": crm_package_name,
        "tenure_days": tenure_days,
        "ooc_days": ooc_days,
        "dd_cancel_60_day": dd_cancel_60_day,
        "contract_dd_cancels": contract_dd_cancels,
        "speed": speed,
        "line_speed": line_speed,
        "calls_30d": calls_30d,
        "loyalty_calls_30d": loyalty_calls_30d,
        "avg_download_30d": avg_download_30d,
        "avg_upload_30d": avg_upload_30d,
        "sum_download_30d": sum_download_30d,
        "sum_upload_30d": sum_upload_30d,
    })

    df_one = pd.DataFrame([base])
    df_one = add_log_features(df_one)

    if st.button("Predict risk", type="primary"):
        df_one, still_missing = ensure_features(df_one, feature_cols, fill_defaults=True)
        if still_missing:
            st.error(f"Internal error: still missing required features: {still_missing}")
            st.stop()

        proba = float(model.predict_proba(df_one[feature_cols])[:, 1][0])
        band = assign_band(proba, q_high, q_medium)

        # KPI row
        c1, c2, c3 = st.columns(3)
        c1.metric("Cease risk (next 30d)", f"{proba:.3f}")
        c2.metric("Risk band", band)
        c3.metric("Recommended action", "See below")

        st.write(f"**Recommended action:** {ACTION_MAP[band]}")

        # Simple gauge-like bar
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption("Risk score shown as probability (0 to 1).")

# ----------------------------
# Batch scoring
# ----------------------------
elif menu == "Batch scoring":
    st.subheader("Batch scoring (CSV upload)")

    st.write("Upload customer feature data. The app will score each row and add risk outputs.")

    # Template download
    template = make_template(feature_cols)
    template_bytes = template.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV template (required columns)",
        data=template_bytes,
        file_name="cease_risk_input_template.csv",
        mime="text/csv",
    )

    strict = st.toggle("Strict schema (error if columns missing)", value=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_in = pd.read_csv(uploaded)
        df_in = add_log_features(df_in)

        df_in, still_missing = ensure_features(df_in, feature_cols, fill_defaults=not strict)

        if still_missing:
            st.error(f"Missing required columns: {still_missing}")
            st.stop()

        df_out = score_dataframe(df_in, model, feature_cols, q_high, q_medium)

        st.success(f"Scored {len(df_out):,} rows.")

        # Display options (feature selection for output display, not model)
        st.markdown("### Output preview settings")
        base_cols = ["risk_score", "risk_band", "recommended_action"]
        optional_cols = [c for c in df_out.columns if c not in base_cols]
        show_cols = st.multiselect(
            "Choose extra columns to display",
            options=optional_cols,
            default=[c for c in ["unique_customer_identifier", "snapshot_date"] if c in optional_cols],
        )
        display_cols = show_cols + base_cols

        st.dataframe(df_out[display_cols].head(50))

        # Charts
        st.markdown("### Risk distribution")
        fig = plt.figure()
        plt.hist(df_out["risk_score"], bins=30)
        plt.xlabel("risk_score")
        plt.ylabel("count")
        st.pyplot(fig)

        st.markdown("### Risk band counts")
        st.bar_chart(df_out["risk_band"].value_counts())

        # Download
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download scored CSV",
            data=csv_bytes,
            file_name="scored_customers.csv",
            mime="text/csv",
        )

# ----------------------------
# Model & Features
# ----------------------------
else:
    st.subheader("Model & feature set")

    st.write("The model uses a fixed feature set saved at training time (metadata.json).")
    st.write(f"**Number of features:** {len(feature_cols)}")

    st.markdown("### Feature list")
    st.code("\n".join(feature_cols))

    st.markdown("### Risk band thresholds")
    st.write({"q_high": q_high, "q_medium": q_medium})

    st.markdown("### Notes on feature selection")
    st.write(
        "- Feature selection is performed during training and stored in metadata.\n"
        "- At inference, the app must supply the same features the model was trained with.\n"
        "- For usability, the app can *display* a subset of columns (output selection), but it should not drop model features unless the model is retrained."
    )