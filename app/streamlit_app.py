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
# Constants used in app (fix NameError)
# ----------------------------
TARGET = "target_30d"
ID_COL = "unique_customer_identifier"
DATE_COL = "snapshot_date"

# Features we keep for scoring but hide from the single-input UI
HIDE_FROM_UI = {"null_score", "rn", "calls_nan_30d"}  # model may require them


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
    """Deterministic inference-time feature engineering."""
    df = df.copy()
    for col in ["sum_download_30d", "sum_upload_30d", "avg_talk_time_30d", "avg_hold_time_30d"]:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))
    return df


def standardise_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("Unknown").str.strip()
    return df


def assign_band(score: float, q_high: float, q_medium: float) -> str:
    if score >= q_high:
        return "High"
    if score >= q_medium:
        return "Medium"
    return "Low"


def build_reason_code(df: pd.DataFrame) -> pd.Series:
    """
    Reason codes based on key drivers available in inference input.
    Works even if some columns are absent.
    """
    df = df.copy()

    dd_cancel = df["dd_cancel_60_day"] if "dd_cancel_60_day" in df.columns else 0
    dd_cnt = df["contract_dd_cancels"] if "contract_dd_cancels" in df.columns else 0
    ooc_days = df["ooc_days"] if "ooc_days" in df.columns else 0
    calls_30d = df["calls_30d"] if "calls_30d" in df.columns else 0
    avg_dl = df["avg_download_30d"] if "avg_download_30d" in df.columns else 0

    calls_hi = float(np.quantile(pd.to_numeric(calls_30d, errors="coerce").fillna(0), 0.90)) if len(df) > 10 else 0
    usage_lo = float(np.quantile(pd.to_numeric(avg_dl, errors="coerce").fillna(0), 0.10)) if len(df) > 10 else 0

    reasons = []
    for i in range(len(df)):
        ddc = int(pd.to_numeric(dd_cancel.iloc[i] if hasattr(dd_cancel, "iloc") else dd_cancel, errors="coerce") or 0)
        ddcnt = float(pd.to_numeric(dd_cnt.iloc[i] if hasattr(dd_cnt, "iloc") else dd_cnt, errors="coerce") or 0)
        ooc = float(pd.to_numeric(ooc_days.iloc[i] if hasattr(ooc_days, "iloc") else ooc_days, errors="coerce") or 0)
        c30 = float(pd.to_numeric(calls_30d.iloc[i] if hasattr(calls_30d, "iloc") else calls_30d, errors="coerce") or 0)
        dl = float(pd.to_numeric(avg_dl.iloc[i] if hasattr(avg_dl, "iloc") else avg_dl, errors="coerce") or 0)

        if ddc == 1 or ddcnt > 0:
            reasons.append("Payment disruption")
        elif ooc > 0:
            reasons.append("Out of contract (OOC)")
        elif dl <= usage_lo:
            reasons.append("Low usage / disengagement")
        elif c30 >= calls_hi and calls_hi > 0:
            reasons.append("High contact volume")
        else:
            reasons.append("General risk")

    return pd.Series(reasons, index=df.index)


def action_plan(band: str, reason: str) -> str:
    if band == "High":
        if reason == "Payment disruption":
            return "Urgent call (48h): billing support + DD recovery + retention offer"
        if reason == "Out of contract (OOC)":
            return "Urgent call (48h): renewal discussion + tailored offer"
        if reason == "High contact volume":
            return "Urgent call (48h): resolve issues + goodwill credit if needed"
        if reason == "Low usage / disengagement":
            return "Call (48h): service health check + engagement offer"
        return "Urgent call (48h): retention specialist + tailored support"

    if band == "Medium":
        if reason == "Payment disruption":
            return "SMS/email: billing help + payment link; escalate if no response"
        if reason == "Out of contract (OOC)":
            return "Email/SMS: renewal reminder + online offer; monitor"
        if reason == "High contact volume":
            return "Email/SMS: troubleshooting resources + option to book support"
        return "Targeted email/SMS + monitor"

    return "No outreach; monitor monthly; general comms"


def ensure_features(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str], fill_defaults: bool) -> tuple[pd.DataFrame, list[str]]:
    """Ensure all required feature columns exist. If fill_defaults=False, return missing."""
    df = df.copy()
    missing = sorted(set(feature_cols) - set(df.columns))

    if missing and fill_defaults:
        for c in missing:
            if c in cat_cols:
                df[c] = "Unknown"
            else:
                df[c] = 0

    still_missing = sorted(set(feature_cols) - set(df.columns))
    return df, still_missing


def score_df(df: pd.DataFrame, model, meta: dict, q_high: float, q_medium: float, strict: bool) -> pd.DataFrame:
    feature_cols = meta["features"]
    cat_cols = meta.get("categorical_features", [])

    df = standardise_categoricals(df, cat_cols)
    df = add_log_features(df)

    df, missing = ensure_features(df, feature_cols, cat_cols, fill_defaults=not strict)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    proba = model.predict_proba(df[feature_cols])[:, 1]
    out = df.copy()
    out["risk_score"] = proba
    out["risk_band"] = [assign_band(float(p), q_high, q_medium) for p in proba]
    out["primary_reason"] = build_reason_code(out)
    out["recommended_action"] = [action_plan(b, r) for b, r in zip(out["risk_band"], out["primary_reason"])]
    return out


# ----------------------------
# App start
# ----------------------------
st.title("UK Telecom Cease Risk Prediction")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model artifacts not found. Ensure models/cease_risk_model.joblib and models/metadata.json exist in the repo.")
    st.stop()

model, meta = load_artifacts()

feature_cols = meta["features"]
cat_cols = meta.get("categorical_features", [])
categorical_levels = meta.get("categorical_levels", {})
top_features_for_ui = meta.get("top_features_for_ui", [])
kpi_val = meta.get("kpis_table_val", None)
kpi_test = meta.get("kpis_table_test", None)

thresholds = meta.get("band_thresholds_from_val", {})
q_high = float(thresholds.get("q_high", 0.95))
q_medium = float(thresholds.get("q_medium", 0.80))

st.caption(
    f"Model: {meta.get('model_name', 'Unknown')} | "
    f"Trained at: {meta.get('trained_at', 'Unknown')} | "
    f"Lookback: {meta.get('lookback_days', '?')}d | Horizon: {meta.get('horizon_days', '?')}d"
)

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
    st.subheader("Business value: prioritise retention outreach")

    bkv = meta.get("business_kpis_val", {})
    bkt = meta.get("business_kpis_test", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Val call-list size (High)", str(bkv.get("call_list_size", "—")))
    c2.metric("Val precision", f"{bkv.get('precision', float('nan')):.3f}" if "precision" in bkv else "—")
    c3.metric("Val recall", f"{bkv.get('recall', float('nan')):.3f}" if "recall" in bkv else "—")
    c4.metric("Val PR-AUC", f"{meta.get('val_pr_auc_hgb', float('nan')):.3f}" if "val_pr_auc_hgb" in meta else "—")

    st.write(
        f"Risk bands derived on validation thresholds:\n"
        f"- High: risk_score ≥ **{q_high:.4f}** (top ~5%)\n"
        f"- Medium: **{q_medium:.4f}** ≤ risk_score < {q_high:.4f} (next ~15%)\n"
        f"- Low: risk_score < {q_medium:.4f}"
    )

    if kpi_val and kpi_test:
        st.subheader("Lift & capture by risk band")
        st.write("Validation")
        st.dataframe(pd.DataFrame(kpi_val))
        st.write("Test")
        st.dataframe(pd.DataFrame(kpi_test))
    else:
        st.info("KPI tables not found in metadata (kpis_table_val/test).")

# ----------------------------
# Single prediction (top features only)
# ----------------------------
elif menu == "Single prediction":
    st.subheader("Single customer scoring (top drivers)")

    st.info(
        "This form shows the most important features. "
        "Any remaining model features are filled with safe defaults."
    )

    # Default row for ALL model features
    row = {c: ("Unknown" if c in cat_cols else 0) for c in feature_cols}

    # Decide UI features (avoid NameError and hide dedup/noise columns)
    if not top_features_for_ui:
        top_features_for_ui = [c for c in feature_cols if c not in HIDE_FROM_UI and not c.startswith("log1p_")][:12]
    else:
        top_features_for_ui = [c for c in top_features_for_ui if c not in HIDE_FROM_UI and not c.startswith("log1p_")]

    cols = st.columns(3)
    for i, feat in enumerate(top_features_for_ui):
        with cols[i % 3]:
            if feat in cat_cols:
                options = categorical_levels.get(feat, ["Unknown"])
                options = [str(o).strip() for o in options if str(o).strip()]
                if "Unknown" not in options:
                    options = ["Unknown"] + options
                else:
                    options = ["Unknown"] + [o for o in options if o != "Unknown"]
                row[feat] = st.selectbox(feat, options=options, index=0)
            else:
                row[feat] = st.number_input(feat, value=float(row[feat]), min_value=0.0)

    df_one = pd.DataFrame([row])
    df_one = standardise_categoricals(df_one, cat_cols)
    df_one = add_log_features(df_one)

    # Ensure hidden-but-required columns exist (dedup artifacts)
    if "null_score" in feature_cols and "null_score" not in df_one.columns:
        df_one["null_score"] = 0
    if "rn" in feature_cols and "rn" not in df_one.columns:
        df_one["rn"] = 1
    if "calls_nan_30d" in feature_cols and "calls_nan_30d" not in df_one.columns:
        df_one["calls_nan_30d"] = 0

    if st.button("Predict risk", type="primary"):
        df_one, missing = ensure_features(df_one, feature_cols, cat_cols, fill_defaults=True)
        if missing:
            st.error(f"Internal error: still missing required columns: {missing}")
            st.stop()

        proba = float(model.predict_proba(df_one[feature_cols])[:, 1][0])
        band = assign_band(proba, q_high, q_medium)

        primary_reason = build_reason_code(df_one).iloc[0]
        recommended_action = action_plan(band, primary_reason)

        c1, c2, c3 = st.columns(3)
        c1.metric("Cease risk (next 30d)", f"{proba:.3f}")
        c2.metric("Risk band", band)
        c3.metric("Primary reason", primary_reason)

        st.write("Recommended action:", recommended_action)
        st.progress(min(max(proba, 0.0), 1.0))

        with st.expander("Show full feature row used for scoring"):
            st.dataframe(df_one[feature_cols].T.rename(columns={0: "value"}))

# ----------------------------
# Batch scoring
# ----------------------------
elif menu == "Batch scoring":
    st.subheader("Batch scoring (CSV upload)")
    st.write("Upload a CSV file. The app will add risk_score, risk_band, primary_reason, recommended_action.")

    strict = st.toggle("Strict schema (error if required columns missing)", value=True)

    # template
    template = pd.DataFrame([{c: ("Unknown" if c in cat_cols else 0) for c in feature_cols}])
    template = add_log_features(template)
    st.download_button(
        "Download input template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="cease_risk_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)

        try:
            df_out = score_df(df_in, model, meta, q_high, q_medium, strict=strict)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        st.success(f"Scored {len(df_out):,} rows.")
        st.dataframe(df_out.head(50))

        st.write("Risk band distribution")
        st.bar_chart(df_out["risk_band"].value_counts())

        st.write("Risk score distribution")
        fig = plt.figure()
        plt.hist(df_out["risk_score"], bins=30)
        plt.xlabel("risk_score")
        plt.ylabel("count")
        st.pyplot(fig)

        st.download_button(
            "Download scored CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="scored_customers.csv",
            mime="text/csv",
        )

# ----------------------------
# Model & features
# ----------------------------
else:
    st.subheader("Model & features")
    st.write("Feature set is fixed at training time and stored in metadata.json.")
    st.write(f"**Number of model features:** {len(feature_cols)}")

    st.markdown("### Top features for UI")
    st.code("\n".join(top_features_for_ui) if top_features_for_ui else "Not found in metadata")

    st.markdown("### Band thresholds")
    st.write({"q_high": q_high, "q_medium": q_medium})

    st.markdown("### Categorical options (sample)")
    for c in cat_cols:
        opts = categorical_levels.get(c, [])
        st.write(c, opts[:15])