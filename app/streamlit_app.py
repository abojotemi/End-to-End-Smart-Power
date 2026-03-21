from __future__ import annotations


import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.config import ARTIFACT_PATH
    from src.pipeline import (
        load_artifacts,
        predict_next_hour,
    )
    from src.train import run_training
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from src.config import ARTIFACT_PATH
    from src.pipeline import (
        load_artifacts,
        predict_next_hour,
    )
    from src.train import run_training

st.set_page_config(page_title="Smart Power Usage Forecast", layout="wide")
st.title("Smart Power Usage Forecasting Dashboard")

st.caption(
    "Train models, compare metrics, detect peak periods, and forecast next-hour demand."
)

with st.sidebar:
    st.header("Configuration")

    run_training_clicked = st.button("Train / Retrain", type="primary")


def load_or_train(force_retrain: bool = False):
    if force_retrain or not ARTIFACT_PATH.exists():
        with st.spinner("Training models..."):
            artifact = run_training()
            st.success(f"✅ Model trained and saved to: {ARTIFACT_PATH}")
            return artifact
    else:
        st.info(f"📦 Loading cached model from: {ARTIFACT_PATH}")
        return load_artifacts(ARTIFACT_PATH)


artifact = None
try:
    artifact = load_or_train(force_retrain=run_training_clicked)

except Exception as exc:
    st.error(f"Failed to load/train pipeline: {exc}")
    st.stop()

if artifact is None:
    st.error("Artifact could not be loaded.")
    st.stop()

results_df = artifact["results_df"].copy()
comparison_df = artifact["comparison_df"].copy().reset_index()
peak_periods_df = artifact["peak_periods"].copy().reset_index()
next_hour = predict_next_hour(artifact)

col1, col2, col3 = st.columns(3)
col1.metric("Best Model", artifact["best_model_name"])
col2.metric("Peak Threshold (kW)", f"{artifact['peak_threshold']:.3f}")
col3.metric(
    "Next-Hour Forecast (kW)", f"{next_hour['predicted_next_hour_power_kw']:.3f}"
)

st.subheader("Model Comparison")
st.dataframe(results_df, width="stretch")

st.subheader("Actual vs Predicted (Test Window)")
line_fig = px.line(
    comparison_df,
    x="datetime",
    y=["actual", "predicted"],
    labels={
        "value": "Global Active Power (kW)",
        "datetime": "Time",
        "variable": "Series",
    },
)
line_fig.add_hline(y=artifact["peak_threshold"], line_dash="dash", line_color="red")
st.plotly_chart(line_fig, width="stretch")

st.subheader("Detected Peak Periods")
if peak_periods_df.empty:
    st.info("No peak periods detected for the selected threshold.")
else:
    st.dataframe(peak_periods_df.head(50), width="stretch")

    scatter = px.scatter(
        peak_periods_df,
        x="datetime",
        y="actual",
        title="Peak Period Points",
        labels={"actual": "Actual Power (kW)", "datetime": "Time"},
    )
    st.plotly_chart(scatter, width="stretch")

st.subheader("Operational Insight")
peak_text = "Yes" if next_hour["is_predicted_peak"] else "No"
st.write(
    pd.DataFrame(
        {
            "latest_timestamp": [next_hour["timestamp_of_latest_input"]],
            "predicted_next_hour_power_kw": [next_hour["predicted_next_hour_power_kw"]],
            "predicted_peak": [peak_text],
        }
    )
)

# ============================================================================
# INFERENCE SECTION
# ============================================================================

st.subheader("🔮 Make a Prediction")

inference_tab1, inference_tab2 = st.tabs(
    ["Test Historical Data", "Custom Input"]
)

with inference_tab1:
    st.write(
        "Select a date from the test set to see what the model predicted vs. actual value."
    )
    
    if not comparison_df.empty:
        available_dates = comparison_df.index.tolist()
        selected_date = st.selectbox(
            "Select a date",
            available_dates,
            format_func=lambda x: str(x),
        )
        
        if selected_date:
            actual_value = comparison_df.loc[selected_date, "actual"]
            predicted_value = comparison_df.loc[selected_date, "predicted"]
            is_peak = actual_value >= artifact["peak_threshold"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Actual Power (kW)", f"{actual_value:.3f}")
            col2.metric("Predicted Power (kW)", f"{predicted_value:.3f}")
            col3.metric("Error (kW)", f"{abs(actual_value - predicted_value):.3f}")
            
            st.metric("Peak Period?", "Yes 🔴" if is_peak else "No 🟢")

with inference_tab2:
    st.write(
        "Provide custom feature values to generate a prediction. "
        "Use recent values as a reference."
    )
    
    # Get the latest values from the data as defaults
    latest_row = artifact["latest_features"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        global_active_power = st.number_input(
            "Global Active Power (kW) [Current Hour]",
            min_value=0.0,
            max_value=10.0,
            value=float(latest_row.get("Global_active_power", 1.0)),
            step=0.1,
        )
    with col2:
        voltage = st.number_input(
            "Voltage (V)",
            min_value=200.0,
            max_value=260.0,
            value=float(latest_row.get("Voltage", 240.0)),
            step=1.0,
        )
    with col3:
        global_intensity = st.number_input(
            "Global Intensity (A)",
            min_value=0.0,
            max_value=50.0,
            value=float(latest_row.get("Global_intensity", 10.0)),
            step=0.5,
        )
    
    if st.button("🚀 Generate Prediction", type="primary"):
        # Create a simple feature vector using available columns
        # We'll use the latest features as a base and modify the key inputs
        custom_features = latest_row.copy()
        custom_features["Global_active_power"] = global_active_power
        custom_features["Voltage"] = voltage
        custom_features["Global_intensity"] = global_intensity
        
        # Recompute lagged and rolling features based on the new current value
        # For simplicity, we assume the lags are proportional
        for lag in [1, 2, 3, 6, 12, 24]:
            custom_features[f"power_lag_{lag}"] = global_active_power * 0.95 ** lag
        
        feature_cols = artifact["feature_cols"]
        custom_features_df = pd.DataFrame([custom_features])[feature_cols]
        
        best_model = artifact["best_model"]
        prediction = float(best_model.predict(custom_features_df)[0])
        is_peak_pred = prediction >= artifact["peak_threshold"]
        
        st.success("✅ Prediction Generated!")
        col1, col2 = st.columns(2)
        col1.metric(
            "Predicted Next Hour Power (kW)", f"{prediction:.3f}", delta=f"{prediction - global_active_power:.3f}"
        )
        col2.metric(
            "Peak Period?",
            "Yes 🔴" if is_peak_pred else "No 🟢",
        )
        st.info(
            f"📊 With current power at {global_active_power:.2f} kW, "
            f"the model predicts the next hour will be {prediction:.2f} kW."
        )

st.download_button(
    label="Download metrics CSV",
    data=results_df.to_csv(index=False),
    file_name="model_metrics.csv",
    mime="text/csv",
)
