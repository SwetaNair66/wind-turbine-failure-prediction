import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Load model and feature names
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# App config
st.set_page_config(page_title="Wind Turbine Failure Prediction", page_icon="🌬️")

# Title
st.title("🌬️ Wind Turbine Failure Prediction")
st.write("This app predicts failures in a wind turbine using SCADA sensor data.")
st.info("⚠️ Upload SCADA data with the same sensor features as the training turbine")

# Tabs
tab1, tab2 = st.tabs(["🔍 Predict Failures", "📊 SHAP Explainability"])

# ============================================================
# TAB 1 — PREDICTIONS
# ============================================================
with tab1:
    st.subheader("Upload SCADA CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.write("### Data Preview (First 5 Rows)")
        st.dataframe(df.head())

        # Check if all required features exist
        missing = [f for f in feature_names if f not in df.columns]

        if missing:
            st.error(f"❌ Missing columns in uploaded file: {missing}")
        else:
            df = df.replace(r'[\[\]]', '', regex=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(0)
            X = df[feature_names]
            predictions = model.predict(X)
            proba = model.predict_proba(X)[:, 1]

            df['Prediction'] = predictions
            df['Failure Probability'] = (proba * 100).round(2)
            df['Status'] = df['Prediction'].map({
                0: '✅ Normal',
                1: '⚠️ Failure'
            })

            st.write("### Prediction Results")
            st.dataframe(df[['Status', 'Failure Probability']].rename(
                columns={'Failure Probability': 'Failure Probability (%)'}
            ))

            # Summary metrics
            total = len(predictions)
            failures = (predictions == 1).sum()
            normal = (predictions == 0).sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Readings", total)
            col2.metric("⚠️ Failures Detected", failures)
            col3.metric("✅ Normal Readings", normal)

            # Failure % warning
            failure_pct = (failures / total) * 100
            if failure_pct > 20:
                st.error(f"🚨 High failure rate detected: {failure_pct:.1f}% of readings show failure!")
            elif failure_pct > 5:
                st.warning(f"⚠️ Moderate failure rate: {failure_pct:.1f}% of readings show failure")
            else:
                st.success(f"✅ Turbine looks healthy! Only {failure_pct:.1f}% failure readings")

# ============================================================
# TAB 2 — SHAP
# ============================================================
with tab2:
    st.subheader("SHAP Explainability Analysis")
    st.write("Upload the same CSV to see which sensors are causing failures")

    uploaded_shap = st.file_uploader(
        "Choose CSV for SHAP Analysis", type="csv", key="shap"
    )

    if uploaded_shap is not None:
        df_shap = pd.read_csv(uploaded_shap)
        df_shap.columns = df_shap.columns.str.strip()

        missing_shap = [f for f in feature_names if f not in df_shap.columns]

        if missing_shap:
            st.error(f"❌ Missing columns: {missing_shap}")
        else:
            df_shap = df_shap.replace(r'[\[\]]', '', regex=True)
            df_shap = df_shap.apply(pd.to_numeric, errors='coerce')
            df_shap = df_shap.fillna(0)
            X_shap = df_shap[feature_names]
            
            sample = X_shap.sample(n=min(500, len(X_shap)), random_state=42)

            with st.spinner("Calculating SHAP values... please wait ⏳"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample)

            # Plot 1 — Summary
            st.write("### Feature Impact on Failure Prediction")
            fig1, ax1 = plt.subplots()
            shap.summary_plot(
                shap_values, sample,
                feature_names=feature_names,
                show=False
            )
            st.pyplot(fig1)
            plt.clf()

            # Plot 2 — Bar
            st.write("### Feature Importance Ranking")
            fig2, ax2 = plt.subplots()
            shap.summary_plot(
                shap_values, sample,
                feature_names=feature_names,
                plot_type='bar',
                show=False
            )
            st.pyplot(fig2)
            plt.clf()
