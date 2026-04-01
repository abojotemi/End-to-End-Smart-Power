import numpy as np
import pandas as pd

from src.pipeline import (
    build_model_frame,
    preprocess_hourly,
    train_and_evaluate,
)


def _make_synthetic_raw_data(days: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=days * 24 * 60, freq="min")

    hour_signal = np.sin(2 * np.pi * idx.hour / 24)
    day_signal = np.cos(2 * np.pi * idx.dayofweek / 7)
    base_power = 2.8 + 0.9 * hour_signal + 0.4 * day_signal
    noise = rng.normal(0.0, 0.08, size=len(idx))
    power = np.clip(base_power + noise, 0.2, None)

    voltage = 236.0 + rng.normal(0.0, 2.0, size=len(idx))
    intensity = np.clip(power * 4.2 + rng.normal(0.0, 0.3, size=len(idx)), 0.1, None)

    return pd.DataFrame(
        {
            "Date": idx.strftime("%d/%m/%Y"),
            "Time": idx.strftime("%H:%M:%S"),
            "Global_active_power": power,
            "Global_reactive_power": np.clip(power * 0.25, 0.01, None),
            "Voltage": voltage,
            "Global_intensity": intensity,
            "Sub_metering_1": rng.uniform(0.0, 2.0, size=len(idx)),
            "Sub_metering_2": rng.uniform(0.0, 2.0, size=len(idx)),
            "Sub_metering_3": rng.uniform(0.0, 2.0, size=len(idx)),
        }
    )


def test_pipeline_builds_features_and_metrics():
    raw = _make_synthetic_raw_data()
    hourly = preprocess_hourly(raw)
    model_df = build_model_frame(hourly)

    assert not model_df.empty
    assert "target_next_6h_avg" in model_df.columns
    assert "power_lag_24" in model_df.columns

    trained = train_and_evaluate(model_df, hourly)
    assert "results_df" in trained
    assert len(trained["results_df"]) >= 4
    assert "RAE" in trained["results_df"].columns
    assert "Relative_Accuracy" in trained["results_df"].columns
    assert "daily_peak_accuracy" in trained
