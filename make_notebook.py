import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

text_cells = [
    """# Multi-Interval Power Prediction ML Pipeline

This notebook mirrors the single-interval ML pipeline but extends it to train and evaluate models across multiple time intervals (30 mins, 1 hr, 2 hrs, 4 hrs, 6 hrs).

We will:
1. Load the raw dataset
2. Create a generic function to clean, resample, and engineer features
3. Loop through intervals and train models for each
4. Export each interval's artifact so the Streamlit dashboard can load them dynamically""",
    
    """## 1. Imports and Configuration""",
    
    """## 2. Load Raw Dataset""",
    
    """## 3. Define Pipeline Functions (Clean, Feature Eng, Train)""",
    
    """## 4. Run Pipeline Across All Intervals""",
]

code_cells = [
    """import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")""",
    
    """data_dir = Path("data")
raw_path = data_dir / "household_power_consumption.csv"

if raw_path.exists():
    raw_df = pd.read_csv(raw_path, delimiter=";")
else:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=235)
    raw_df = dataset.data.features.copy()

display(raw_df.head())""",
    
    """SUPPORTED_INTERVALS = ["30min", "1hr", "2hr", "4hr", "6hr"]

INTERVAL_RESAMPLE_RULES = {
    "30min": "30min",
    "1hr": "1h",
    "2hr": "2h",
    "4hr": "4h",
    "6hr": "6h",
}

INTERVAL_FEATURE_PARAMS = {
    "30min": {"lags": [1, 2, 3, 5, 10, 24, 48, 672], "rolling_windows": [2, 6, 24], "max_lag": 672},
    "1hr": {"lags": [1, 2, 3, 5, 10, 24, 48, 672], "rolling_windows": [2, 6, 24], "max_lag": 672},
    "2hr": {"lags": [1, 2, 3, 6, 12, 24, 84], "rolling_windows": [2, 6, 12], "max_lag": 84},
    "4hr": {"lags": [1, 2, 3, 6, 12, 42], "rolling_windows": [2, 6], "max_lag": 42},
    "6hr": {"lags": [1, 2, 4, 8, 28], "rolling_windows": [2, 4], "max_lag": 28},
}

TARGET = "Global_active_power"
NUM_COLS = [TARGET, "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

def clean_dataframe(raw_df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
    df = raw_df.copy()
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")
    df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df.drop(columns=["Date", "Time"], errors="ignore")
    df = df.dropna(subset=[TARGET])
    df[NUM_COLS] = df[NUM_COLS].ffill().fillna(df[NUM_COLS].median())
    df = df.drop(columns=["Voltage", "Global_intensity", "Global_reactive_power"])
    df = df.resample(resample_rule).mean()
    remaining = [c for c in NUM_COLS if c in df.columns]
    df[remaining] = df[remaining].ffill().fillna(df[remaining].median())
    return df

def engineer_features(df: pd.DataFrame, interval_key: str) -> pd.DataFrame:
    d = df.copy()
    gap = d[TARGET]
    params = INTERVAL_FEATURE_PARAMS[interval_key]
    
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    d["month"] = d.index.month
    d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
    d["season"] = d["month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    
    for lag in params["lags"]:
        d[f"lag_{lag}"] = gap.shift(lag)
        
    gap_lagged = gap.shift(1)
    for w in params["rolling_windows"]:
        rolled = gap_lagged.rolling(window=w, min_periods=1)
        d[f"roll_mean_{w}"] = rolled.mean()
        d[f"roll_std_{w}"] = rolled.std().fillna(0)
    
    return d.dropna()""",
    
    """all_trained_metrics = {}

for interval_key in SUPPORTED_INTERVALS:
    print(f"\\n{'='*50}")
    print(f"Training for interval: {interval_key}")
    print(f"{'='*50}")
    
    resample_rule = INTERVAL_RESAMPLE_RULES[interval_key]
    
    # 1. Clean & Resample
    df_clean = clean_dataframe(raw_df, resample_rule)
    
    # 2. Split into Train & Test (80/20)
    split_idx = int(len(df_clean) * 0.8)
    df_raw_train = df_clean.iloc[:split_idx]
    df_raw_test = df_clean.iloc[split_idx:]
    
    # 3. Feature Engineering
    df_feat_train = engineer_features(df_raw_train, interval_key)
    # Ensure test set gets recent history for lags
    max_lag = INTERVAL_FEATURE_PARAMS[interval_key]["max_lag"]
    df_feat_test_full = engineer_features(pd.concat([df_raw_train.iloc[-max_lag:], df_raw_test]), interval_key)
    df_feat_test = df_feat_test_full.loc[df_raw_test.index].dropna()
    
    feature_cols = [c for c in df_feat_train.columns if c != TARGET]
    X_train = df_feat_train[feature_cols].astype(np.float32)
    y_train = df_feat_train[TARGET].astype(np.float32)
    X_test = df_feat_test[feature_cols].astype(np.float32)
    y_test = df_feat_test[TARGET].astype(np.float32)
    
    # 4. Train Models
    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Ensemble (Simple Average)
    y_pred_ensemble = (y_pred_lgb + y_pred_xgb) / 2.0
    
    # 5. Evaluate
    r2 = r2_score(y_test, y_pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    
    all_trained_metrics[interval_key] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f"Metrics -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Display final summary
summary_df = pd.DataFrame(all_trained_metrics).T
display(summary_df)"""
]

cells = []
for i in range(len(text_cells)):
    cells.append(nbf.v4.new_markdown_cell(text_cells[i]))
    if i < len(code_cells):
        cells.append(nbf.v4.new_code_cell(code_cells[i]))

nb['cells'] = cells

with open('electric_power_ml_multi_interval.ipynb', 'w') as f:
    nbf.write(nb, f)
