import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Multi-Interval Power Prediction ML Pipeline\n",
    "\n",
    "This notebook mirrors the single-interval ML pipeline but extends it to train and evaluate models across multiple time intervals (30 mins, 1 hr, 2 hrs, 4 hrs, 6 hrs).\n",
    "\n",
    "We will:\n",
    "1. Load the raw dataset\n",
    "2. Create a generic function to clean, resample, and engineer features\n",
    "3. Loop through intervals and train models for each\n",
    "4. Export each interval's artifact so the Streamlit dashboard can load them dynamically"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "import joblib\n",
    "\n",
    "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score\n",
    "import xgboost as xgb\n",
    "import lightgbm as lgb\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Imports and Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir = Path(\"data\")\n",
    "raw_path = data_dir / \"household_power_consumption.csv\"\n",
    "\n",
    "if raw_path.exists():\n",
    "    raw_df = pd.read_csv(raw_path, delimiter=\";\")\n",
    "else:\n",
    "    from ucimlrepo import fetch_ucirepo\n",
    "    dataset = fetch_ucirepo(id=235)\n",
    "    raw_df = dataset.data.features.copy()\n",
    "\n",
    "display(raw_df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Raw Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "SUPPORTED_INTERVALS = [\"30min\", \"1hr\", \"2hr\", \"4hr\", \"6hr\"]\n",
    "\n",
    "INTERVAL_RESAMPLE_RULES = {\n",
    "    \"30min\": \"30min\",\n",
    "    \"1hr\": \"1h\",\n",
    "    \"2hr\": \"2h\",\n",
    "    \"4hr\": \"4h\",\n",
    "    \"6hr\": \"6h\",\n",
    "}\n",
    "\n",
    "INTERVAL_FEATURE_PARAMS = {\n",
    "    \"30min\": {\"lags\": [1, 2, 3, 5, 10, 24, 48, 672], \"rolling_windows\": [2, 6, 24], \"max_lag\": 672},\n",
    "    \"1hr\": {\"lags\": [1, 2, 3, 5, 10, 24, 48, 672], \"rolling_windows\": [2, 6, 24], \"max_lag\": 672},\n",
    "    \"2hr\": {\"lags\": [1, 2, 3, 6, 12, 24, 84], \"rolling_windows\": [2, 6, 12], \"max_lag\": 84},\n",
    "    \"4hr\": {\"lags\": [1, 2, 3, 6, 12, 42], \"rolling_windows\": [2, 6], \"max_lag\": 42},\n",
    "    \"6hr\": {\"lags\": [1, 2, 4, 8, 28], \"rolling_windows\": [2, 4], \"max_lag\": 28},\n",
    "}\n",
    "\n",
    "TARGET = \"Global_active_power\"\n",
    "NUM_COLS = [TARGET, \"Global_reactive_power\", \"Voltage\", \"Global_intensity\", \"Sub_metering_1\", \"Sub_metering_2\", \"Sub_metering_3\"]\n",
    "\n",
    "def clean_dataframe(raw_df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:\n",
    "    df = raw_df.copy()\n",
    "    for col in NUM_COLS:\n",
    "        df[col] = pd.to_numeric(df[col].replace(\"?\", np.nan), errors=\"coerce\")\n",
    "    df[\"datetime\"] = pd.to_datetime(df[\"Date\"].astype(str) + \" \" + df[\"Time\"].astype(str), dayfirst=True, errors=\"coerce\")\n",
    "    df = df.dropna(subset=[\"datetime\"]).set_index(\"datetime\").sort_index()\n",
    "    df = df.drop(columns=[\"Date\", \"Time\"], errors=\"ignore\")\n",
    "    df = df.dropna(subset=[TARGET])\n",
    "    df[NUM_COLS] = df[NUM_COLS].ffill().fillna(df[NUM_COLS].median())\n",
    "    df = df.drop(columns=[\"Voltage\", \"Global_intensity\", \"Global_reactive_power\"])\n",
    "    df = df.resample(resample_rule).mean()\n",
    "    remaining = [c for c in NUM_COLS if c in df.columns]\n",
    "    df[remaining] = df[remaining].ffill().fillna(df[remaining].median())\n",
    "    return df\n",
    "\n",
    "def engineer_features(df: pd.DataFrame, interval_key: str) -> pd.DataFrame:\n",
    "    d = df.copy()\n",
    "    gap = d[TARGET]\n",
    "    params = INTERVAL_FEATURE_PARAMS[interval_key]\n",
    "    \n",
    "    d[\"hour\"] = d.index.hour\n",
    "    d[\"dayofweek\"] = d.index.dayofweek\n",
    "    d[\"month\"] = d.index.month\n",
    "    d[\"is_weekend\"] = (d.index.dayofweek >= 5).astype(int)\n",
    "    d[\"season\"] = d[\"month\"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})\n",
    "    \n",
    "    for lag in params[\"lags\"]:\n",
    "        d[f\"lag_{lag}\"] = gap.shift(lag)\n",
    "        \n",
    "    gap_lagged = gap.shift(1)\n",
    "    for w in params[\"rolling_windows\"]:\n",
    "        rolled = gap_lagged.rolling(window=w, min_periods=1)\n",
    "        d[f\"roll_mean_{w}\"] = rolled.mean()\n",
    "        d[f\"roll_std_{w}\"] = rolled.std().fillna(0)\n",
    "    \n",
    "    return d.dropna()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Define Pipeline Functions (Clean, Feature Eng, Train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_trained_metrics = {}\n",
    "\n",
    "for interval_key in SUPPORTED_INTERVALS:\n",
    "    print(f\"\\n{'='*50}\")\n",
    "    print(f\"Training for interval: {interval_key}\")\n",
    "    print(f\"{'='*50}\")\n",
    "    \n",
    "    resample_rule = INTERVAL_RESAMPLE_RULES[interval_key]\n",
    "    \n",
    "    # 1. Clean & Resample\n",
    "    df_clean = clean_dataframe(raw_df, resample_rule)\n",
    "    \n",
    "    # 2. Split into Train & Test (80/20)\n",
    "    split_idx = int(len(df_clean) * 0.8)\n",
    "    df_raw_train = df_clean.iloc[:split_idx]\n",
    "    df_raw_test = df_clean.iloc[split_idx:]\n",
    "    \n",
    "    # 3. Feature Engineering\n",
    "    df_feat_train = engineer_features(df_raw_train, interval_key)\n",
    "    # Ensure test set gets recent history for lags\n",
    "    max_lag = INTERVAL_FEATURE_PARAMS[interval_key][\"max_lag\"]\n",
    "    df_feat_test_full = engineer_features(pd.concat([df_raw_train.iloc[-max_lag:], df_raw_test]), interval_key)\n",
    "    df_feat_test = df_feat_test_full.loc[df_raw_test.index].dropna()\n",
    "    \n",
    "    feature_cols = [c for c in df_feat_train.columns if c != TARGET]\n",
    "    X_train = df_feat_train[feature_cols].astype(np.float32)\n",
    "    y_train = df_feat_train[TARGET].astype(np.float32)\n",
    "    X_test = df_feat_test[feature_cols].astype(np.float32)\n",
    "    y_test = df_feat_test[TARGET].astype(np.float32)\n",
    "    \n",
    "    # 4. Train Models\n",
    "    # LightGBM\n",
    "    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)\n",
    "    lgb_model.fit(X_train, y_train)\n",
    "    y_pred_lgb = lgb_model.predict(X_test)\n",
    "    \n",
    "    # XGBoost\n",
    "    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)\n",
    "    xgb_model.fit(X_train, y_train)\n",
    "    y_pred_xgb = xgb_model.predict(X_test)\n",
    "    \n",
    "    # Ensemble (Simple Average)\n",
    "    y_pred_ensemble = (y_pred_lgb + y_pred_xgb) / 2.0\n",
    "    \n",
    "    # 5. Evaluate\n",
    "    r2 = r2_score(y_test, y_pred_ensemble)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))\n",
    "    mae = mean_absolute_error(y_test, y_pred_ensemble)\n",
    "    \n",
    "    all_trained_metrics[interval_key] = {\"R2\": r2, \"RMSE\": rmse, \"MAE\": mae}\n",
    "    print(f\"Metrics -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}\")\n",
    "\n",
    "# Display final summary\n",
    "summary_df = pd.DataFrame(all_trained_metrics).T\n",
    "display(summary_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Pipeline Across All Intervals"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Export and Integrate with Dashboard\n",
    "\n",
    "To ensure that all models are properly saved with the correct schema and directory structure for the Streamlit dashboard, we invoke the `run_training_with_options` function from the updated `src.train` module. This function internally calls `train_all_intervals` from `src.electric_power_ml_multi.py` and saves the artifact files (.joblib) and metric CSVs for each interval."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.train import run_training_with_options\n",
    "\n",
    "# You can specify max_rows=100000 to train faster during prototyping\n",
    "all_trained_artifacts = run_training_with_options(max_rows=None)\n",
    "\n",
    "print(\"Training and export complete for intervals:\", list(all_trained_artifacts.keys()))\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("electric_power_ml_multi_interval.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

