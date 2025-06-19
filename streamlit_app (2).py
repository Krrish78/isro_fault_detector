import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ISRO Fault Detector")
st.title("ISRO Satellite Fault/Anomaly Detector")
st.markdown("Upload telemetry data and detect anomalies in real-time.")

# 📦 Load models
scaler = joblib.load("scaler.pkl")
iso_model = joblib.load("iso_model.pkl")

try:
    clf = joblib.load("xgb_model.pkl")
    xgb_available = True
except:
    clf = None
    xgb_available = False

# 📁 File upload
uploaded_file = st.file_uploader("Upload telemetry CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file).dropna()
    df_numeric = df.select_dtypes(include=[np.number])
    df_scaled = scaler.transform(df_numeric)

    scores = iso_model.decision_function(df_scaled)
    preds = iso_model.predict(df_scaled)

    df['Anomaly_Score'] = scores
    df['Prediction'] = preds

    st.subheader("Anomaly Detection Output")
    st.dataframe(df.head())
    st.line_chart(df['Anomaly_Score'])

    threshold = st.slider("Threshold", float(np.min(scores)), float(np.max(scores)), float(np.median(scores)))
    flagged = df[df['Anomaly_Score'] < threshold]
    st.warning(f"{len(flagged)} anomalies flagged.")

    if xgb_available:
        xgb_preds = clf.predict(df_scaled)
        df['XGB_Fault_Prediction'] = xgb_preds
        st.subheader("XGBoost Fault Predictions")
        st.dataframe(df[['XGB_Fault_Prediction']].value_counts())
