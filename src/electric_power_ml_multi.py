"""Training pipeline for multiple time intervals (30min, 1hr, 2hr, 4hr, 6hr).

Mirrors the logic of electric_power_ml.py but parameterised on resample interval.
Each trained artifact is saved separately so the Streamlit dashboard can switch
between intervals at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACT_SCHEMA_VERSION,
    DATA_DIR,
    MODEL_EXPORT_DIR,
    RANDOM_STATE,
    artifact_path_for_interval,
    metrics_path_for_interval,
    comparison_path_for_interval,
    peak_periods_path_for_interval,
    SUPPORTED_INTERVALS,
)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore

TARGET = "Global_active_power"
NUM_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
TIME_OF_DAY_LABELS = {0: "Night", 1: "Morning", 2: "Afternoon", 3: "Evening"}
ENSEMBLE_MODEL_NAME = "Weighted Ensemble"
STACKING_MODEL_NAME = "StackingEnsemble"

# Mapping from user-facing interval label to pandas resample rule
INTERVAL_RESAMPLE_RULES: dict[str, str] = {
    "30min": "30min",
    "1hr": "1h",
    "2hr": "2h",
    "4hr": "4h",
    "6hr": "6h",
}

# Lag and rolling window parameters differ by interval so that they remain
# physically meaningful.  Each key maps to (lag_periods, rolling_windows).
# Lags are expressed in *periods* of the chosen interval.
INTERVAL_FEATURE_PARAMS: dict[str, dict[str, list[int]]] = {
    "30min": {
        "lags": [1, 2, 3, 5, 10, 24, 48, 672],      # 30min periods
        "rolling_windows": [2, 6, 24],
        "max_lag": 672,
    },
    "1hr": {
        "lags": [1, 2, 3, 5, 10, 24, 48, 672],       # 1hr periods
        "rolling_windows": [2, 6, 24],
        "max_lag": 672,
    },
    "2hr": {
        "lags": [1, 2, 3, 6, 12, 24, 84],             # 2hr periods
        "rolling_windows": [2, 6, 12],
        "max_lag": 84,
    },
    "4hr": {
        "lags": [1, 2, 3, 6, 12, 42],                 # 4hr periods
        "rolling_windows": [2, 6],
        "max_lag": 42,
    },
    "6hr": {
        "lags": [1, 2, 4, 8, 28],                     # 6hr periods
        "rolling_windows": [2, 4],
        "max_lag": 28,
    },
}


def _fetch_uci_household_power() -> pd.DataFrame:
    """Download UCI Household Power Consumption (id=235) when no local CSV exists."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=235)
    features = dataset.data.features if dataset.data is not None else None
    if features is None:
        raise RuntimeError("UCI dataset fetch returned no feature data")
    return features.copy()


def load_raw_dataframe() -> pd.DataFrame:
    """Load household power data from a local CSV or UCI (for cloud deploys)."""
    csv_path = DATA_DIR / "household_power_consumption.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, delimiter=";")
    return _fetch_uci_household_power()


def clean_dataframe(raw_df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
    """Clean raw data and resample to the requested interval."""
    df = raw_df.copy()

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df.drop(columns=["Date", "Time"], errors="ignore")
    df = df.dropna(subset=["Global_active_power"])
    df[NUM_COLS] = df[NUM_COLS].ffill().fillna(df[NUM_COLS].median())

    # Remove leaky raw sensor columns
    leaky = ["Voltage", "Global_intensity", "Global_reactive_power"]
    df = df.drop(columns=[c for c in leaky if c in df.columns])

    # Resample
    df = df.resample(resample_rule).mean()
    remaining_num = [c for c in NUM_COLS if c in df.columns]
    df[remaining_num] = df[remaining_num].ffill().fillna(df[remaining_num].median())

    return df


def engineer_features(df: pd.DataFrame, interval_key: str) -> pd.DataFrame:
    """Build features appropriate for the chosen interval."""
    d = df.copy()
    gap = d["Global_active_power"]

    params = INTERVAL_FEATURE_PARAMS[interval_key]
    lags = params["lags"]
    windows = params["rolling_windows"]

    # ── Temporal features ─────────────────────────────────────────────────
    d["hour"] = d.index.hour
    d["minute"] = d.index.minute
    d["dayofweek"] = d.index.dayofweek
    d["dayofyear"] = d.index.dayofyear
    d["month"] = d.index.month
    d["year"] = d.index.year
    d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
    d["quarter"] = d.index.quarter
    d["season"] = d["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    d["time_of_day"] = pd.cut(
        d["hour"], bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]
    ).astype(int)

    # Cyclical encoding
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["dayofweek"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dayofweek"] / 7)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["doy_sin"] = np.sin(2 * np.pi * d["dayofyear"] / 365)
    d["doy_cos"] = np.cos(2 * np.pi * d["dayofyear"] / 365)

    # ── Physics-derived features ──────────────────────────────────────────
    d["sub_total"] = d["Sub_metering_1"] + d["Sub_metering_2"] + d["Sub_metering_3"]

    # ── Lag features ──────────────────────────────────────────────────────
    for lag in lags:
        d[f"lag_{lag}"] = gap.shift(lag)

    # ── Rolling statistics (shifted to prevent leakage) ───────────────────
    gap_lagged = gap.shift(1)
    for w in windows:
        rolled = gap_lagged.rolling(window=w, min_periods=1)
        d[f"roll_mean_{w}"] = rolled.mean()
        d[f"roll_std_{w}"] = rolled.std().fillna(0)
        d[f"roll_min_{w}"] = rolled.min()
        d[f"roll_max_{w}"] = rolled.max()

    # ── Interaction features ──────────────────────────────────────────────
    d["hour_x_weekend"] = d["hour"] * d["is_weekend"]
    d["hour_x_season"] = d["hour"] * d["season"]

    return d


def _peak_hour_to_time_of_day(hour: int) -> int:
    if hour < 6:
        return 0
    if hour < 12:
        return 1
    if hour < 18:
        return 2
    return 3


def _time_of_day_to_window(idx: int) -> tuple[int, int]:
    return {0: (0, 5), 1: (6, 11), 2: (12, 17), 3: (18, 23)}.get(idx, (0, 23))


def _time_of_day_to_representative_hour(idx: int) -> int:
    return {0: 2, 1: 9, 2: 15, 3: 21}.get(idx, 12)


def _summarize_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    den = np.abs(y_true - y_true.mean()).sum()
    rae = float(np.abs(y_true - y_pred).sum() / den) if den > 0 else 0.0
    return {
        "Model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100),
        "RAE": rae,
    }


def _export_models(
    ensemble_models: dict[str, Any],
    best_model_name: str,
    best_model: Any,
    ensemble_weights: dict[str, float],
    interval_key: str,
) -> dict[str, str]:
    export_dir = Path(MODEL_EXPORT_DIR) / interval_key
    export_dir.mkdir(parents=True, exist_ok=True)
    model_paths: dict[str, str] = {}

    for name, model in ensemble_models.items():
        path = export_dir / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, path)
        model_paths[name] = str(path)

    best_path = export_dir / f"best_{best_model_name.lower().replace(' ', '_')}.joblib"
    if best_model is not None:
        joblib.dump(best_model, best_path)
    else:
        joblib.dump(
            {
                "best_model_name": best_model_name,
                "models": ensemble_models,
                "weights": ensemble_weights,
            },
            best_path,
        )
    model_paths["best_model"] = str(best_path)
    return model_paths


def train_for_interval(
    interval_key: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Train all models for one specific time interval.

    ``interval_key`` must be one of ``SUPPORTED_INTERVALS``.
    Returns the artifact dict ready for persistence.
    """
    if interval_key not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unknown interval: {interval_key!r}. Choose from {SUPPORTED_INTERVALS}")
    if xgb is None:
        raise RuntimeError("xgboost is required for training")
    if lgb is None:
        raise RuntimeError("lightgbm is required for training")

    resample_rule = INTERVAL_RESAMPLE_RULES[interval_key]
    params = INTERVAL_FEATURE_PARAMS[interval_key]

    raw_df = load_raw_dataframe()
    if max_rows is not None and max_rows > 0 and len(raw_df) > max_rows:
        raw_df = raw_df.tail(max_rows).copy()

    df_clean = clean_dataframe(raw_df, resample_rule)

    # Split BEFORE feature engineering to prevent data leakage
    split_ts = df_clean.index[int(len(df_clean) * 0.80)]
    df_raw_train = df_clean[df_clean.index < split_ts]
    df_raw_test = df_clean[df_clean.index >= split_ts]

    df_feat_train = engineer_features(df_raw_train, interval_key).dropna()

    max_lag = params["max_lag"]
    context = df_raw_train.iloc[-max_lag:]
    df_feat_test_full = engineer_features(pd.concat([context, df_raw_test]), interval_key)
    df_feat_test = df_feat_test_full.loc[df_raw_test.index].dropna()

    feature_cols = [c for c in df_feat_train.columns if c != TARGET]
    X_train = df_feat_train[feature_cols].astype(np.float32)
    y_train = df_feat_train[TARGET].astype(np.float32)
    X_test = df_feat_test[feature_cols].astype(np.float32)
    y_test = df_feat_test[TARGET].astype(np.float32)

    df_feat = pd.concat([df_feat_train, df_feat_test])

    # ── Scaler for Ridge ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # ── Ridge ─────────────────────────────────────────────────────────────
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_tr_sc, y_train)
    y_pred_ridge = ridge.predict(X_te_sc)

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=4, min_samples_leaf=4,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method="hist", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)

    # ── LightGBM ──────────────────────────────────────────────────────────
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=127,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    y_pred_lgb = lgb_model.predict(X_test)

    # ── Stacking / Weighted Ensemble ──────────────────────────────────────
    y_pred_stack = (y_pred_rf + y_pred_xgb + y_pred_lgb) / 3.0
    y_pred_weighted = y_pred_stack.copy()

    predictions_map = {
        "Ridge": y_pred_ridge,
        "RandomForest": y_pred_rf,
        "XGBoost": y_pred_xgb,
        "LightGBM": y_pred_lgb,
        ENSEMBLE_MODEL_NAME: y_pred_weighted,
    }

    results_df = pd.DataFrame(
        [_summarize_metrics(name, y_test.values, pred) for name, pred in predictions_map.items()]
    ).sort_values("RMSE").reset_index(drop=True)

    best_model_name = str(results_df.iloc[0]["Model"])
    ensemble_models = {
        "Ridge": ridge,
        "RandomForest": rf,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
    }
    ensemble_weights = {name: 0.0 for name in ensemble_models}
    ensemble_weights.update({"RandomForest": 1 / 3, "XGBoost": 1 / 3, "LightGBM": 1 / 3})

    if best_model_name == ENSEMBLE_MODEL_NAME:
        best_model = None
        app_best_model_name = ENSEMBLE_MODEL_NAME
    else:
        best_model = ensemble_models[best_model_name]
        app_best_model_name = best_model_name

    best_pred = predictions_map[best_model_name]
    comparison_df = pd.DataFrame(
        {"datetime": y_test.index, "actual": y_test.values, "predicted": best_pred}
    ).set_index("datetime")

    peak_threshold = float(comparison_df["actual"].quantile(0.90))
    peak_periods = comparison_df[comparison_df["actual"] >= peak_threshold].copy()

    # ── Daily peak‑hour classifier ────────────────────────────────────────
    df_hourly = df_clean.resample("1h").mean().dropna().copy()
    daily_df = (
        df_hourly.assign(date=df_hourly.index.date)
        .groupby("date")
        .agg(
            day_mean=("Global_active_power", "mean"),
            day_max=("Global_active_power", "max"),
            day_std=("Global_active_power", "std"),
            peak_hour=("Global_active_power", lambda s: int(s.idxmax().hour)),
        )
    )
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df["day_of_week"] = daily_df.index.dayofweek
    daily_df["month"] = daily_df.index.month
    daily_df["is_weekend"] = (daily_df["day_of_week"] >= 5).astype(int)
    for col in ["day_mean", "day_max", "day_std"]:
        daily_df[f"prev_{col}"] = daily_df[col].shift(1)

    daily_df["peak_time_of_day"] = daily_df["peak_hour"].apply(_peak_hour_to_time_of_day)
    daily_df = daily_df.dropna().copy()

    peak_feature_cols = [c for c in daily_df.columns if c != "peak_time_of_day"]
    split_daily = int(len(daily_df) * 0.8)
    X_peak_train = daily_df.iloc[:split_daily][peak_feature_cols]
    y_peak_train = daily_df.iloc[:split_daily]["peak_time_of_day"]
    X_peak_test = daily_df.iloc[split_daily:][peak_feature_cols]
    y_peak_test = daily_df.iloc[split_daily:]["peak_time_of_day"]

    peak_model = RandomForestClassifier(
        n_estimators=350, max_depth=12, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    peak_model.fit(X_peak_train, y_peak_train)
    y_peak_pred = peak_model.predict(X_peak_test)
    peak_accuracy = float(accuracy_score(y_peak_test, y_peak_pred))

    latest_features_daily = daily_df.iloc[-1][peak_feature_cols]
    predicted_peak_tod = int(peak_model.predict(latest_features_daily.to_frame().T)[0])
    predicted_peak_hour = _time_of_day_to_representative_hour(predicted_peak_tod)
    backup_start, backup_end = _time_of_day_to_window(predicted_peak_tod)

    daily_peak_comparison_df = pd.DataFrame({
        "date": daily_df.index[split_daily:],
        "actual_peak_time_of_day": y_peak_test.values,
        "predicted_peak_time_of_day": y_peak_pred,
    }).set_index("date")

    model_paths = _export_models(
        ensemble_models=ensemble_models,
        best_model_name=app_best_model_name,
        best_model=best_model,
        ensemble_weights=ensemble_weights,
        interval_key=interval_key,
    )

    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_pipeline": "electric_power_ml_multi",
        "interval_key": interval_key,
        "resample_rule": resample_rule,
        "results_df": results_df,
        "comparison_df": comparison_df,
        "peak_periods": peak_periods,
        "peak_threshold": peak_threshold,
        "best_model_name": app_best_model_name,
        "best_model": best_model,
        "ensemble_models": ensemble_models,
        "ensemble_weights": ensemble_weights,
        "stacking_model_names": ["RandomForest", "XGBoost", "LightGBM"],
        "feature_cols": feature_cols,
        "latest_features": pd.concat([X_train, X_test]).iloc[-1],
        "latest_timestamp": str(df_feat.index[-1]),
        "target_col": TARGET,
        "scaler": scaler,
        "daily_peak_model": {"RandomForest": peak_model},
        "daily_peak_ensemble_weights": {"RandomForest": 1.0},
        "daily_peak_feature_cols": peak_feature_cols,
        "latest_daily_features": latest_features_daily,
        "daily_peak_accuracy": peak_accuracy,
        "daily_peak_comparison_df": daily_peak_comparison_df,
        "predicted_peak_time_of_day_next_day": predicted_peak_tod,
        "predicted_peak_time_of_day_label": TIME_OF_DAY_LABELS[predicted_peak_tod],
        "predicted_peak_hour_next_day": predicted_peak_hour,
        "backup_power_time_window": f"{backup_start:02d}:00 - {backup_end:02d}:59",
        "predicted_peak": bool(best_pred[-1] >= peak_threshold),
        "model_paths": model_paths,
        "best_metrics": results_df.iloc[0].to_dict(),
    }


def train_all_intervals(max_rows: int | None = None) -> dict[str, dict[str, Any]]:
    """Train models for every supported interval and return all artifacts keyed by interval."""
    all_artifacts: dict[str, dict[str, Any]] = {}
    for interval_key in SUPPORTED_INTERVALS:
        print(f"\n{'='*60}")
        print(f"  Training models for interval: {interval_key}")
        print(f"{'='*60}")
        artifact = train_for_interval(interval_key, max_rows=max_rows)
        all_artifacts[interval_key] = artifact
    return all_artifacts


def predict_power(custom_features: pd.DataFrame, artifact: dict[str, Any]) -> float:
    """Predict active power for one feature row (matches notebook inference)."""
    feature_cols = artifact["feature_cols"]
    row = custom_features[feature_cols]

    best_name = str(artifact["best_model_name"])
    if best_name == ENSEMBLE_MODEL_NAME:
        total = 0.0
        for name, model in artifact["ensemble_models"].items():
            weight = artifact["ensemble_weights"].get(name, 0.0)
            if weight <= 0:
                continue
            if name == "Ridge":
                scaled = artifact["scaler"].transform(row)
                total += weight * float(model.predict(scaled)[0])
            else:
                total += weight * float(model.predict(row)[0])
        return total

    model = artifact["best_model"]
    if model is None:
        names = artifact.get("stacking_model_names", ["RandomForest", "XGBoost", "LightGBM"])
        preds = [float(artifact["ensemble_models"][n].predict(row)[0]) for n in names]
        return float(np.mean(preds))

    if best_name == "Ridge":
        scaled = artifact["scaler"].transform(row)
        return float(model.predict(scaled)[0])
    return float(model.predict(row)[0])
