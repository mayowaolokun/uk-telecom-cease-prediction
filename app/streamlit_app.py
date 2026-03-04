import streamlit as st
import pandas as pd

st.set_page_config(page_title="UK Telecom Cease Risk", layout="wide")
st.title("UK Telecom Cease Risk Prediction")

st.caption("This is a scaffold app. We'll connect the trained model later.")

tab1, tab2 = st.tabs(["Single prediction", "Batch scoring"])

with tab1:
    st.subheader("Single prediction")
    st.info("Model not connected yet. We'll add inputs after defining target + features.")
    example = {
        "unique_customer_identifier": "CUST_0001",
        "tenure_days": 120,
        "ooc_days": 0,
        "dd_cancel_60_day": 0,
        "recent_calls_30d": 1,
        "avg_download_mbs_30d": 2500,
    }
    st.json(example)

with tab2:
    st.subheader("Batch scoring")
    file = st.file_uploader("Upload a CSV (we'll score it once the model is connected)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(df.head(20))
        st.warning("Scoring not connected yet.")