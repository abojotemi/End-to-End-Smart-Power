from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
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

DEFAULT_FORECAST_HORIZON = 6
ENSEMBLE_MODEL_NAME = "Weighted Ensemble"


def load_raw_data() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    return fetch_ucirepo(id=235).data.features  # type: ignore


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


def build_model_frame(
    df_hourly: pd.DataFrame, forecast_horizon: int = DEFAULT_FORECAST_HORIZON
) -> pd.DataFrame:
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1")

    model_df = df_hourly.copy()
    dt_index = pd.DatetimeIndex(model_df.index)

    future_power = pd.concat(
        [
            model_df["Global_active_power"].shift(-step)
            for step in range(1, forecast_horizon + 1)
        ],
        axis=1,
    )
    model_df["target_next_6h_avg"] = future_power.mean(axis=1)

    model_df["hour"] = dt_index.hour
    model_df["day_of_week"] = dt_index.dayofweek
    model_df["day_of_year"] = dt_index.dayofyear
    model_df["month"] = dt_index.month
    model_df["is_weekend"] = (model_df["day_of_week"] >= 5).astype(int)

    model_df["hour_sin"] = np.sin(2 * np.pi * model_df["hour"] / 24.0)
    model_df["hour_cos"] = np.cos(2 * np.pi * model_df["hour"] / 24.0)

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        model_df[f"power_lag_{lag}"] = model_df["Global_active_power"].shift(lag)
        model_df[f"voltage_lag_{lag}"] = model_df["Voltage"].shift(lag)
        model_df[f"current_lag_{lag}"] = model_df["Global_intensity"].shift(lag)

    for window in [3, 6, 12, 24]:
        model_df[f"power_roll_mean_{window}"] = (
            model_df["Global_active_power"].rolling(window).mean()
        )
        model_df[f"power_roll_std_{window}"] = (
            model_df["Global_active_power"].rolling(window).std()
        )
        model_df[f"current_roll_mean_{window}"] = (
            model_df["Global_intensity"].rolling(window).mean()
        )

    return model_df.dropna()


def build_daily_peak_frame(df_hourly: pd.DataFrame) -> pd.DataFrame:
    hourly_dt_index = pd.DatetimeIndex(df_hourly.index)
    daily_stats = (
        df_hourly.assign(date=hourly_dt_index.date)
        .groupby("date")
        .agg(
            day_mean=("Global_active_power", "mean"),
            day_max=("Global_active_power", "max"),
            day_std=("Global_active_power", "std"),
            peak_hour=("Global_active_power", lambda s: int(s.idxmax().hour)),
        )
    )

    daily_stats.index = pd.to_datetime(daily_stats.index)
    daily_dt_index = pd.DatetimeIndex(daily_stats.index)
    daily_stats["day_of_week"] = daily_dt_index.dayofweek
    daily_stats["month"] = daily_dt_index.month
    daily_stats["is_weekend"] = (daily_stats["day_of_week"] >= 5).astype(int)

    for col in ["day_mean", "day_max", "day_std"]:
        daily_stats[f"prev_{col}"] = daily_stats[col].shift(1)

    return daily_stats.dropna()


def _build_models() -> dict[str, Any]:
    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "MLP Compact": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP Deep": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=5e-5,
                        learning_rate_init=8e-4,
                        max_iter=600,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def _relative_absolute_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true - float(np.mean(y_true))).sum()
    if denominator == 0:
        return 0.0
    numerator = np.abs(y_true - y_pred).sum()
    return float(numerator / denominator)


def _evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rae = _relative_absolute_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "RAE": rae,
        "Relative_Accuracy": float(1.0 - rae),
    }


def _predict_with_selected_model(
    features: pd.DataFrame, artifact: dict[str, Any]
) -> np.ndarray:
    best_model_name = str(artifact["best_model_name"])
    if best_model_name == ENSEMBLE_MODEL_NAME:
        models: dict[str, Any] = artifact["ensemble_models"]
        weights: dict[str, float] = artifact["ensemble_weights"]
        weighted = np.zeros(len(features), dtype=float)
        for name, model in models.items():
            weighted += weights[name] * model.predict(features)
        return weighted

    model = artifact["best_model"]
    return model.predict(features)


def train_and_evaluate(
    model_df: pd.DataFrame,
    df_hourly: pd.DataFrame,
    split_ratio: float = 0.8,
) -> dict[str, Any]:
    target_col = "target_next_6h_avg"
    feature_cols = [c for c in model_df.columns if c != target_col]

    X = model_df[feature_cols]
    y = model_df[target_col]

    split_idx = int(len(model_df) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = model_df.index[split_idx:]

    val_split_idx = int(len(X_train) * 0.8)
    X_fit, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_fit, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]

    models = _build_models()
    results = []
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}
    validation_rmse: dict[str, float] = {}

    for name, model in models.items():
        model.fit(X_fit, y_fit)
        val_pred = model.predict(X_val)
        validation_rmse[name] = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = _evaluate_regression(y_test, y_pred)
        results.append({"Model": name, **metrics})
        predictions[name] = y_pred
        fitted_models[name] = model

    inv_errors = {name: 1.0 / max(err, 1e-8) for name, err in validation_rmse.items()}
    inv_total = float(sum(inv_errors.values()))
    ensemble_weights = {
        name: float(score / inv_total) for name, score in inv_errors.items()
    }

    ensemble_pred = np.zeros(len(y_test), dtype=float)
    for name, y_pred in predictions.items():
        ensemble_pred += ensemble_weights[name] * y_pred

    ensemble_metrics = _evaluate_regression(y_test, ensemble_pred)
    results.append({"Model": ENSEMBLE_MODEL_NAME, **ensemble_metrics})

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = str(results_df.loc[0, "Model"])

    if best_model_name == ENSEMBLE_MODEL_NAME:
        best_model = None
        best_pred = ensemble_pred
    else:
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

    peak_daily_df = build_daily_peak_frame(df_hourly)
    peak_target_col = "peak_hour"
    peak_feature_cols = [c for c in peak_daily_df.columns if c != peak_target_col]

    peak_split_idx = int(len(peak_daily_df) * split_ratio)
    X_peak = peak_daily_df[peak_feature_cols]
    y_peak = peak_daily_df[peak_target_col]

    X_peak_train, X_peak_test = (
        X_peak.iloc[:peak_split_idx],
        X_peak.iloc[peak_split_idx:],
    )
    y_peak_train, y_peak_test = (
        y_peak.iloc[:peak_split_idx],
        y_peak.iloc[peak_split_idx:],
    )

    peak_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    peak_model.fit(X_peak_train, y_peak_train)
    peak_pred_test = peak_model.predict(X_peak_test)
    peak_accuracy = float(accuracy_score(y_peak_test, peak_pred_test))

    daily_peak_comparison_df = pd.DataFrame(
        {
            "date": peak_daily_df.index[peak_split_idx:],
            "actual_peak_hour": y_peak_test.values,
            "predicted_peak_hour": peak_pred_test,
        }
    ).set_index("date")

    latest_daily_features = X_peak.iloc[-1]
    predicted_peak_hour_next_day = int(
        peak_model.predict(latest_daily_features.to_frame().T)[0]
    )

    return {
        "results_df": results_df,
        "comparison_df": comparison_df,
        "peak_periods": peak_periods,
        "peak_threshold": float(peak_threshold),
        "best_model_name": best_model_name,
        "best_model": best_model,
        "ensemble_models": fitted_models,
        "ensemble_weights": ensemble_weights,
        "feature_cols": feature_cols,
        "latest_features": X.iloc[-1],
        "latest_timestamp": str(model_df.index[-1]),
        "target_col": target_col,
        "daily_peak_model": peak_model,
        "daily_peak_feature_cols": peak_feature_cols,
        "latest_daily_features": latest_daily_features,
        "daily_peak_accuracy": peak_accuracy,
        "daily_peak_comparison_df": daily_peak_comparison_df,
        "predicted_peak_hour_next_day": predicted_peak_hour_next_day,
    }


def predict_next_6_hours(
    artifact: dict[str, Any],
) -> dict[str, float | str | bool | int]:
    feature_cols = artifact["feature_cols"]
    latest_features = artifact["latest_features"]

    if isinstance(latest_features, pd.Series):
        latest_features = latest_features.to_frame().T

    latest_features = latest_features[feature_cols]
    prediction = float(_predict_with_selected_model(latest_features, artifact)[0])
    peak_hour = int(artifact["predicted_peak_hour_next_day"])

    return {
        "timestamp_of_latest_input": artifact["latest_timestamp"],
        "predicted_next_6h_avg_power_kw": prediction,
        "is_predicted_peak_period": prediction >= float(artifact["peak_threshold"]),
        "predicted_peak_hour_next_day": peak_hour,
        "backup_power_time_window": f"{peak_hour:02d}:00 - {peak_hour:02d}:59",
    }


def predict_next_hour(artifact: dict[str, Any]) -> dict[str, float | str | bool | int]:
    """Backward-compatible alias for previous API naming."""
    return predict_next_6_hours(artifact)


def save_artifacts(artifact_path: Path | str, payload: dict[str, Any]) -> None:
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_path)


def load_artifacts(artifact_path: Path | str) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")
    return joblib.load(artifact_path)
