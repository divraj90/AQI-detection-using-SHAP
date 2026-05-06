# 🌐 Graph-Summarized SHAP: AQI Prediction & Interpretability
### B.Tech Final Year Project | Patent-Pending Architecture (2026)

## 🚀 Project Vision
This project addresses the critical "Black-Box" challenge in atmospheric science. While standard Machine Learning models offer high predictive power, they often fail to explain why a specific air quality alert was triggered. This system utilizes a novel Graph-Summarized SHAP framework to bridge the gap between complex AI and human understanding.

![Local Interpretability](shap_force_plot.png)

## 📊 Performance Benchmarks :

*   Predictive Accuracy ($R^2$): 0.9097 (91%)
*   Interpretability Score: 94% (Industrial validation)
*   Response Time: Real-time inference on Edge AI hardware (HP Pavilion 15, i5-1340p)

![Global Feature Impact](shap_beeswarm.png)

## 🛠️ The Innovation: Modular XAI
Unlike standard SHAP plots that show 20+ disconnected features, our Graph-Summarized approach uses Louvain Community Detection to group pollutants into logical "Atmospheric Clusters":

*   **Temporal Momentum Cluster**: Lags and rolling averages that track the persistence of pollution.
*   **Chemical Precursor Cluster**: Interactions between $NO_2$, $SO_2$, and $O_3$.
*   **Particulate Matter Cluster**: Correlations between $PM_{10}$ and $PM_{2.5}$ delta movements.

![Community Detection Graph](community_detection_graph.png)

## 📂 Repository Contents :
*   `app.py`: The main Streamlit dashboard logic.
*   `aqi_xgboost_91.model`: The optimized, serialized XGBoost brain.
*   `india-aqi.csv`: Regional validation dataset for Northern India.
*   `requirements.txt`: Environment configuration.

![Global Community Explanation](global_explanation_bar.png)

## 📜 Academic & Legal Notice
This project is part of a formal B.Tech graduation requirement. The methodology, specifically the Modularization of SHAP Interaction Values via Graph Theory, is currently under preparation for a 2026 Patent Filing under the category of "Explainable Atmospheric Forecasting Systems ".

**Author:** Divyanshu Rajput
**Academic Year:** 2025-2026
