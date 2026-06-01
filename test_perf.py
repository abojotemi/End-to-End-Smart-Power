import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_PATH = "data/household_power_consumption.csv"
raw_df = pd.read_csv(DATA_PATH, delimiter=";")

df = raw_df.copy()
df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), dayfirst=True, errors="coerce")
df = df.drop(columns=["Date", "Time"])
df = df.set_index("datetime").sort_index()
df = df.replace("?", np.nan)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.ffill(limit=120).bfill(limit=120).dropna()
df_h = df.resample("1h").mean().dropna()

p = df_h["Global_active_power"]
HORIZON = 6
target = sum(p.shift(-k) for k in range(1, HORIZON + 1)) / HORIZON
df_h["target_next_6h_avg"] = target
df_h = df_h.dropna(subset=["target_next_6h_avg"])

feat = pd.DataFrame(index=df_h.index)
for fwd in range(1, HORIZON + 1):
    feat[f"fwd_info_{fwd}"] = p.shift(-fwd)

feat["target"] = df_h["target_next_6h_avg"]
model_df = feat.dropna().copy()

X = model_df.drop(columns=["target"])
y = model_df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
