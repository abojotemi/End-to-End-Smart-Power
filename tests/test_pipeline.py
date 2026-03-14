from pathlib import Path

from power_forecast.pipeline import (
    build_model_frame,
    load_raw_data,
    preprocess_hourly,
    train_and_evaluate,
)


def test_pipeline_builds_features_and_metrics():
    raw = load_raw_data()
    hourly = preprocess_hourly(raw)
    model_df = build_model_frame(hourly)

    assert not model_df.empty
    assert "target_next_hour" in model_df.columns
    assert "power_lag_24" in model_df.columns

    trained = train_and_evaluate(model_df)
    assert "results_df" in trained
    assert len(trained["results_df"]) >= 3
