import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Set Page Config
st.set_page_config(page_title="AQI XAI Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    # Load your optimized XGBoost model and the explainer
    model = xgb.Booster()
    model.load_model("aqi_xgboost_91.model")
    explainer = shap.TreeExplainer(model)
    return model, explainer

def main():
    st.title("🌐 Graph‑Summarized SHAP: AQI Prediction")
    st.markdown("### Real-time Explainable AI for Air Quality Monitoring")
    
    model, explainer = load_assets()

    # --- SIDEBAR: INPUT FEATURES ---
    st.sidebar.header("Input Pollutant Levels")
    pm10 = st.sidebar.slider("PM10", 0, 500, 100)
    no2 = st.sidebar.slider("NO2", 0, 200, 40)
    so2 = st.sidebar.slider("SO2", 0, 150, 15)
    rolling_mean = st.sidebar.slider("Rolling Mean (3-hr)", 0, 400, 80)
    pm25_delta = st.sidebar.slider("PM2.5 Delta (Momentum)", -50, 50, 5)
    
    # --- DATA PREPARATION ---
    # Create the DataFrame with all 12 expected columns
    # Using 0 as a baseline for placeholders to ensure model compatibility
    input_data = pd.DataFrame([[
        0, 0, 0, 0, 0, 0, 0, pm10, no2, so2, rolling_mean, pm25_delta
    ]], columns=[
        'rolling_std_3', 'lag_1', 'lag_3', 'lag_2', 'CO', 'NH3', 'OZONE', 
        'PM10', 'NO2', 'SO2', 'rolling_mean_3', 'pm25_delta'
    ])

    # Strictly enforce the order provided in the error message
    expected_order = [
        'CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'SO2', 
        'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 
        'rolling_std_3', 'pm25_delta'
    ]
    input_data = input_data[expected_order]
    
    # Convert to DMatrix for high-performance inference
    dmatrix = xgb.DMatrix(input_data)

    # --- MAIN SECTION: PREDICTION ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        prediction = model.predict(dmatrix)[0]
        st.metric(label="Predicted PM2.5 Concentration", value=f"{prediction:.2f}")
        
        # Determine AQI Category (Standard Indian CPCB Scale)
        if prediction <= 50: status = "Good"
        elif prediction <= 100: status = "Satisfactory"
        else: status = "Poor/Hazardous"
        st.info(f"Air Quality Status: **{status}**")

    # --- EXPLANATION SECTION ---
    st.divider()
    st.subheader("📊 Why this prediction?")
    
    # Local SHAP Force Plot
    shap_values = explainer.shap_values(input_data)
    st.write("#### Feature-Level Influence (Force Plot)")
    
    # Generate and display the SHAP plot
    fig = shap.force_plot(
        explainer.expected_value, 
        shap_values[0,:], 
        input_data.iloc[0,:], 
        matplotlib=True, 
        show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()

if __name__ == "__main__":
    main()
