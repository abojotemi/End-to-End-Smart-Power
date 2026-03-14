from __future__ import annotations


import pandas as pd
import plotly.express as px
import streamlit as st

from power_forecast.config import ARTIFACT_PATH
from power_forecast.pipeline import (
    load_artifacts,
    predict_next_hour,
)
from power_forecast.train import run_training

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
            return run_training()
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

st.download_button(
    label="Download metrics CSV",
    data=results_df.to_csv(index=False),
    file_name="model_metrics.csv",
    mime="text/csv",
)
