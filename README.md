# AQI Prediction and SHAP Explainability Dashboard

This project predicts PM2.5 air pollution levels using an XGBoost model and explains the prediction with SHAP values. It includes a Streamlit dashboard where users can adjust pollutant and time-series inputs, view the predicted PM2.5 concentration, and inspect which features influenced the prediction.

## Project Objective

Air quality models can give useful predictions, but their output is hard to trust if the reason behind the prediction is not visible. This project focuses on two parts:

- predicting PM2.5 concentration from pollutant and lag-based features
- explaining each prediction using SHAP feature attribution

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- XGBoost
- SHAP
- Matplotlib

## Repository Structure

```text
.
├── app.py
├── aqi_xgboost_91.model
├── Data Export.csv
├── requirements.txt
└── README.md
```

## Features

- Interactive Streamlit dashboard
- PM2.5 prediction using a trained XGBoost model
- Input sliders for pollutant and time-series values
- Air quality category display
- SHAP force plot for local model explanation

## Visual Output

The repository includes saved explainability images such as SHAP plots and feature impact charts. These help explain how the model interprets pollutant and time-series features.

## How To Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Inputs Used In The App

The dashboard accepts these visible inputs:

- PM10
- NO2
- SO2
- rolling mean over 3 hours
- PM2.5 delta

The model also expects additional lag and pollutant features. In the current dashboard version, those extra values are filled with baseline values so the trained model receives the correct feature order.

## What I Learned

- How to load and use a saved XGBoost model
- How to build an interactive ML dashboard with Streamlit
- How SHAP helps explain model predictions
- Why feature order matters during model inference
- How to present model output in a user-friendly way

## Limitations

- Some model inputs are currently set to baseline values in the app.
- The dashboard is a prototype and should not be used for real public-health decisions.
- More validation is needed before using this model outside a learning or academic context.

## Future Improvements

- Add full input support for all model features
- Display global SHAP plots in the dashboard
- Add dataset preprocessing and training notebook
- Add model evaluation metrics and charts
- Deploy the app online

