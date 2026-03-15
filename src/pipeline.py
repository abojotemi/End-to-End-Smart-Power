from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_raw_data() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo     
    # fetch dataset 
    return fetch_ucirepo(id=235).data.features # type: ignore



def preprocess_hourly(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["datetime", "Global_active_power", "Voltage", "Global_intensity"]
    )
    df = df.sort_values("datetime")

    df_hourly = (
        df.set_index("datetime")[["Global_active_power", "Voltage", "Global_intensity"]]
        .resample("h")
        .mean()
        .dropna()
    )
    return df_hourly


def build_model_frame(df_hourly: pd.DataFrame) -> pd.DataFrame:
    model_df = df_hourly.copy()
    dt_index = pd.DatetimeIndex(model_df.index)
    model_df["target_next_hour"] = model_df["Global_active_power"].shift(-1)

    model_df["hour"] = dt_index.hour
    model_df["day_of_week"] = dt_index.dayofweek
    model_df["month"] = dt_index.month
    model_df["is_weekend"] = (model_df["day_of_week"] >= 5).astype(int)

    for lag in [1, 2, 3, 6, 12, 24]:
        model_df[f"power_lag_{lag}"] = model_df["Global_active_power"].shift(lag)
        model_df[f"voltage_lag_{lag}"] = model_df["Voltage"].shift(lag)
        model_df[f"current_lag_{lag}"] = model_df["Global_intensity"].shift(lag)

    for window in [3, 6, 24]:
        model_df[f"power_roll_mean_{window}"] = (
            model_df["Global_active_power"].rolling(window).mean()
        )
        model_df[f"current_roll_mean_{window}"] = (
            model_df["Global_intensity"].rolling(window).mean()
        )

    return model_df.dropna()


def _build_models() -> dict[str, Any]:
    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_evaluate(
    model_df: pd.DataFrame, split_ratio: float = 0.8
) -> dict[str, Any]:
    target_col = "target_next_hour"
    feature_cols = [c for c in model_df.columns if c != target_col]

    X = model_df[feature_cols]
    y = model_df[target_col]

    split_idx = int(len(model_df) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = model_df.index[split_idx:]

    models = _build_models()
    results = []
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred),
            }
        )
        predictions[name] = y_pred
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = str(results_df.loc[0, "Model"])
    best_model = fitted_models[best_model_name]
    best_pred = predictions[best_model_name]

    comparison_df = pd.DataFrame(
        {
            "datetime": dates_test,
            "actual": y_test.values,
            "predicted": best_pred,
        }
    ).set_index("datetime")

    peak_threshold = comparison_df["actual"].quantile(0.90)
    peak_periods = comparison_df[comparison_df["actual"] >= peak_threshold].copy()

    return {
        "results_df": results_df,
        "comparison_df": comparison_df,
        "peak_periods": peak_periods,
        "peak_threshold": float(peak_threshold),
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_cols": feature_cols,
        "latest_features": X.iloc[-1],
        "latest_timestamp": str(model_df.index[-1]),
    }


def predict_next_hour(artifact: dict[str, Any]) -> dict[str, float | str | bool]:
    model = artifact["best_model"]
    feature_cols = artifact["feature_cols"]
    latest_features = artifact["latest_features"]

    if isinstance(latest_features, pd.Series):
        latest_features = latest_features.to_frame().T

    latest_features = latest_features[feature_cols]
    prediction = float(model.predict(latest_features)[0])

    return {
        "timestamp_of_latest_input": artifact["latest_timestamp"],
        "predicted_next_hour_power_kw": prediction,
        "is_predicted_peak": prediction >= float(artifact["peak_threshold"]),
    }


def save_artifacts(artifact_path: Path | str, payload: dict[str, Any]) -> None:
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_path)


def load_artifacts(artifact_path: Path | str) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")
    return joblib.load(artifact_path)
