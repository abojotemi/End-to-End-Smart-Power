import subprocess, sys

for p in ["ucimlrepo", "xgboost", "lightgbm", "optuna"]:
    subprocess.run([sys.executable, "-m", "pip", "install", p, "-q"], check=False)
print("✅ ready")

# ---

import warnings

warnings.filterwarnings("ignore")
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#2d3147",
        "axes.labelcolor": "#c8cce8",
        "text.color": "#c8cce8",
        "xtick.color": "#8890b5",
        "ytick.color": "#8890b5",
        "grid.color": "#2d3147",
        "grid.alpha": 0.6,
        "font.family": "monospace",
        "axes.titlecolor": "#e8eaf6",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)
PAL = ["#7c83fd", "#fd7c7c", "#7cfdb8", "#fdd97c", "#c47cfd", "#7cd4fd"]
print("✅ Libraries loaded")

# ---

from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=235)
raw_df = dataset.data.features.copy()
print(f'Dataset: {dataset.metadata["name"]}')
print(f"Shape  : {raw_df.shape}")
print(f"Columns: {raw_df.columns.tolist()}")
raw_df.head(3)

# ---

BASE_PATH = Path(".")
DATA_PATH = BASE_PATH.resolve() / "data"
path = next(DATA_PATH.glob("*"))

raw_df = pd.read_csv(path, delimiter=";")
raw_df.head()

# ---

# ── Parse datetime ────────────────────────────────────────────────────────────
df = raw_df.copy()
if "Date" in df.columns and "Time" in df.columns:
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])
df = df.set_index("datetime").sort_index()

# Force numeric
df = df.replace("?", np.nan)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"Date range: {df.index.min()} → {df.index.max()}")
print(f"Missing   : {df.isnull().sum().sum():,}")

# Fill short gaps (≤2h = 120 min), drop large gaps
df = df.ffill(limit=120).bfill(limit=120).dropna()
print(f"After fill: {df.shape}")

# ---

# ── HOURLY resample ───────────────────────────────────────────────────────────
# Hourly = correct resolution for 6h-ahead forecasting.
# At hourly scale: lag_24 = yesterday, lag_168 = last week (critical!)
df_h = df.resample("1h").mean().dropna()
print(f"Hourly shape: {df_h.shape}")

# Identify power column
power_col = next(
    (c for c in ["Global_active_power", "global_active_power"] if c in df_h.columns),
    df_h.columns[0],
)
print(f'Power column: "{power_col}"')
print(f"Power stats :\n{df_h[power_col].describe()}")

# ---

# target[t] = mean( power[t+1], power[t+2], ..., power[t+6] )
# Correct no-leakage construction:
HORIZON = 6  # hours ahead
p = df_h[power_col]

# Sum next 6 steps then divide — shift each step individually
target = sum(p.shift(-k) for k in range(1, HORIZON + 1)) / HORIZON
df_h["target_next_6h_avg"] = target

df_h = df_h.dropna(subset=["target_next_6h_avg"])
print(f"After target creation: {df_h.shape}")

# Verify: correlation between current power and target
corr_now = p.corr(df_h["target_next_6h_avg"])
corr_lag24 = p.shift(24).corr(df_h["target_next_6h_avg"])
corr_lag168 = p.shift(168).corr(df_h["target_next_6h_avg"])
print(f"\nCorr(power_t,   target): {corr_now:.4f}   ← expect high (same-period)")
print(f"Corr(lag_24h,  target): {corr_lag24:.4f}   ← yesterday same time")
print(f"Corr(lag_168h, target): {corr_lag168:.4f}   ← last week same time")

# ---

def make_features(df, power_col):
    p = df[power_col]
    idx = df.index
    feat = pd.DataFrame(index=idx)

    # ── A. Calendar ──────────────────────────────────────────────────────────
    feat["hour"] = idx.hour
    feat["dow"] = idx.dayofweek
    feat["month"] = idx.month
    feat["quarter"] = idx.quarter
    feat["dayofyear"] = idx.dayofyear
    feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
    feat["is_peak_am"] = feat["hour"].between(7, 9).astype(int)
    feat["is_peak_pm"] = feat["hour"].between(18, 22).astype(int)
    feat["is_night"] = feat["hour"].isin(range(0, 6)).astype(int)
    feat["hour_block"] = feat["hour"] // 6  # 0=night,1=morning,2=afternoon,3=evening

    # ── B. Cyclical encoding ─────────────────────────────────────────────────
    for col, period in [("hour", 24), ("dow", 7), ("month", 12), ("dayofyear", 365)]:
        feat[f"{col}_sin"] = np.sin(2 * np.pi * feat[col] / period)
        feat[f"{col}_cos"] = np.cos(2 * np.pi * feat[col] / period)

    # ── C. Hourly lags (all semantically meaningful) ──────────────────────────
    # These are the CORE predictors for 6h-ahead forecasting:
    # lag_1..6  : recent history (same-day context)
    # lag_24    : yesterday same hour   ← strongest single predictor
    # lag_48/72 : 2/3 days ago
    # lag_168   : last week same hour   ← second strongest
    # lag_336   : two weeks ago
    lags = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        18,
        24,
        36,
        48,
        72,
        96,
        120,
        144,
        168,
        192,
        216,
        240,
        336,
    ]
    for lag in lags:
        feat[f"lag_{lag}"] = p.shift(lag)

    # Differences (momentum/velocity)
    feat["diff_1h"] = p.diff(1)
    feat["diff_6h"] = p.diff(6)
    feat["diff_24h"] = p.diff(24)
    feat["diff_168h"] = p.diff(168)

    # ── D. Rolling statistics at multiple windows ─────────────────────────────
    for h, w in [
        ("3h", 3),
        ("6h", 6),
        ("12h", 12),
        ("24h", 24),
        ("48h", 48),
        ("168h", 168),
    ]:
        r = p.shift(1).rolling(w, min_periods=1)  # shift(1) to avoid leakage
        feat[f"rmean_{h}"] = r.mean()
        feat[f"rstd_{h}"] = r.std().fillna(0)
        feat[f"rmin_{h}"] = r.min()
        feat[f"rmax_{h}"] = r.max()

    # EWM (exponential weighted mean)
    for span in [3, 6, 24, 72, 168]:
        feat[f"ewm_{span}h"] = p.shift(1).ewm(span=span, adjust=False).mean()

    # ── E. Same-period block features (KEY for 6h-ahead) ─────────────────────
    # The 6h block containing hour h in the previous day/week
    # This IS the target's natural predictor
    feat["block_yesterday_mean"] = p.shift(24).rolling(6, min_periods=1).mean()
    feat["block_yesterday_max"] = p.shift(24).rolling(6, min_periods=1).max()
    feat["block_lastweek_mean"] = p.shift(168).rolling(6, min_periods=1).mean()
    feat["block_lastweek_max"] = p.shift(168).rolling(6, min_periods=1).max()
    # Average of same-day-of-week over last 4 weeks
    feat["block_4week_mean"] = (
        p.shift(168) + p.shift(336) + p.shift(504) + p.shift(672)
    ) / 4

    # ── F. Interaction features ───────────────────────────────────────────────
    feat["hour_x_weekend"] = feat["hour"] * feat["is_weekend"]
    feat["hour_x_month"] = feat["hour"] * feat["month"]
    feat["lag24_x_weekend"] = feat["lag_24"] * feat["is_weekend"]
    feat["lag168_x_hour_sin"] = feat["lag_168"] * feat["hour_sin"]

    # ── G. All sensor columns (raw) ───────────────────────────────────────────
    for col in df.columns:
        if col not in feat.columns and col != "target_next_6h_avg":
            feat[col] = df[col].values
            feat[f"{col}_lag1"] = df[col].shift(1).values
            feat[f"{col}_lag24"] = df[col].shift(24).values

    # ── H. Fourier harmonics ──────────────────────────────────────────────────
    t_arr = np.arange(len(df))
    for k in [1, 2, 3, 4]:
        feat[f"fd_s{k}"] = np.sin(2 * np.pi * k * t_arr / 24)
        feat[f"fd_c{k}"] = np.cos(2 * np.pi * k * t_arr / 24)
        feat[f"fw_s{k}"] = np.sin(2 * np.pi * k * t_arr / 168)
        feat[f"fw_c{k}"] = np.cos(2 * np.pi * k * t_arr / 168)

    return feat


feat_df = make_features(df_h, power_col)
feat_df["target_next_6h_avg"] = df_h["target_next_6h_avg"]
model_df = feat_df.dropna().copy()

print(f"Features : {model_df.shape[1]-1}")
print(f"Samples  : {model_df.shape[0]:,}")
print(f"Date range: {model_df.index.min()} → {model_df.index.max()}")

# ---

def metrics(name, yt, yp):
    den = np.abs(yt - yt.mean()).sum()
    rae = float(np.abs(yt - yp).sum() / den) if den > 0 else 0
    mape_ = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100)
    return dict(
        Model=name,
        MAE=mean_absolute_error(yt, yp),
        RMSE=np.sqrt(mean_squared_error(yt, yp)),
        R2=r2_score(yt, yp),
        MAPE=mape_,
        RAE=rae,
        Accuracy=(1 - rae) * 100,
    )


target_col = "target_next_6h_avg"
feature_cols = [c for c in model_df.columns if c != target_col]

X = model_df[feature_cols]
y = model_df[target_col]

n = len(model_df)
tr_end = int(n * 0.70)
vl_end = int(n * 0.80)

X_train, y_train = X.iloc[:tr_end], y.iloc[:tr_end]
X_val, y_val = X.iloc[tr_end:vl_end], y.iloc[tr_end:vl_end]
X_test, y_test = X.iloc[vl_end:], y.iloc[vl_end:]
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
dates_test = model_df.index[vl_end:]

print(
    f"Train : {len(X_train):,}  ({X_train.index.min().date()} → {X_train.index.max().date()})"
)
print(
    f"Val   : {len(X_val):,}  ({X_val.index.min().date()} → {X_val.index.max().date()})"
)
print(
    f"Test  : {len(X_test):,}  ({X_test.index.min().date()} → {X_test.index.max().date()})"
)
print(f"Features: {X.shape[1]}")

# ---

# ── 5.1  Tune XGBoost ────────────────────────────────────────────────────────
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def xgb_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 1500),
        learning_rate=trial.suggest_float("lr", 0.005, 0.08, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 9),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample", 0.5, 0.9),
        min_child_weight=trial.suggest_int("min_cw", 1, 10),
        reg_alpha=trial.suggest_float("alpha", 1e-4, 2.0, log=True),
        reg_lambda=trial.suggest_float("lambda", 1e-4, 2.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 0.5),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=40,
        eval_metric="rmse",
    )
    m = xgb.XGBRegressor(**params)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return np.sqrt(mean_squared_error(y_val, m.predict(X_val)))


study_xgb = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study_xgb.optimize(xgb_objective, n_trials=40, show_progress_bar=True)
print(f"\nBest XGBoost RMSE (val): {study_xgb.best_value:.5f}")
print("Best params:", study_xgb.best_params)

# ---

# ── 5.2  Tune LightGBM ───────────────────────────────────────────────────────
def lgb_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 2000),
        learning_rate=trial.suggest_float("lr", 0.005, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample", 0.5, 0.9),
        reg_alpha=trial.suggest_float("alpha", 1e-4, 2.0, log=True),
        reg_lambda=trial.suggest_float("lambda", 1e-4, 2.0, log=True),
        min_child_samples=trial.suggest_int("min_cs", 5, 50),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        early_stopping_rounds=40,
    )
    m = lgb.LGBMRegressor(**params)
    m.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
    )
    return np.sqrt(mean_squared_error(y_val, m.predict(X_val)))


study_lgb = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study_lgb.optimize(lgb_objective, n_trials=40, show_progress_bar=True)
print(f"\nBest LightGBM RMSE (val): {study_lgb.best_value:.5f}")
print("Best params:", study_lgb.best_params)

# ---

import time

results = []
predictions = {}
fitted_models = {}
val_rmse = {}

# ── XGBoost (best Optuna params) ─────────────────────────────────────────────
xgb_p = study_xgb.best_params.copy()
n_est_xgb = xgb_p.pop("n_estimators")
lr_xgb = xgb_p.pop("lr")
min_cw = xgb_p.pop("min_cw")
colsample = xgb_p.pop("colsample")

xgb_best = xgb.XGBRegressor(
    n_estimators=n_est_xgb,
    learning_rate=lr_xgb,
    min_child_weight=min_cw,
    colsample_bytree=colsample,
    **xgb_p,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)

# ── LightGBM (best Optuna params) ────────────────────────────────────────────
lgb_p = study_lgb.best_params.copy()
n_est_lgb = lgb_p.pop("n_estimators")
lr_lgb = lgb_p.pop("lr")
min_cs = lgb_p.pop("min_cs")
colsample_lgb = lgb_p.pop("colsample")

lgb_best = lgb.LGBMRegressor(
    n_estimators=n_est_lgb,
    learning_rate=lr_lgb,
    min_child_samples=min_cs,
    colsample_bytree=colsample_lgb,
    **lgb_p,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

# ── All models ────────────────────────────────────────────────────────────────
models = {
    "Ridge": Pipeline([("sc", RobustScaler()), ("m", Ridge(alpha=0.5))]),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.4,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": xgb_best,
    "LightGBM": lgb_best,
}

for name, model in models.items():
    t0 = time.time()
    print(f"Training {name}...", end=" ")
    model.fit(X_train, y_train)
    vp = model.predict(X_val)
    val_rmse[name] = np.sqrt(mean_squared_error(y_val, vp))
    model.fit(X_tv, y_tv)
    yp = model.predict(X_test)
    predictions[name] = yp
    fitted_models[name] = model
    r = metrics(name, y_test.values, yp)
    results.append(r)
    print(f'R²={r["R2"]:.4f}  RMSE={r["RMSE"]:.4f}  ({time.time()-t0:.1f}s)')

print("\n✅ All models trained")

# ---

# ── 7.1  Weighted Ensemble (inv-RMSE) ─────────────────────────────────────────
inv = {k: 1.0 / max(v, 1e-9) for k, v in val_rmse.items()}
inv_s = sum(inv.values())
w = {k: v / inv_s for k, v in inv.items()}
print("Weights:")
for k, v in sorted(w.items(), key=lambda x: -x[1]):
    print(f"  {k:20s}: {v:.4f}")

ens_pred = sum(wt * predictions[k] for k, wt in w.items())
results.append(metrics("Weighted Ensemble", y_test.values, ens_pred))
predictions["Weighted Ensemble"] = ens_pred
print(
    f'\nWeighted Ensemble  R²={results[-1]["R2"]:.4f}  RMSE={results[-1]["RMSE"]:.4f}'
)

# ---

# ── 7.2  Optimized blend (scipy minimize on val set) ─────────────────────────
from scipy.optimize import minimize

# Get val predictions for blending
val_preds = {}
for name, model in fitted_models.items():
    # Re-predict on val (already trained on train only initially)
    tmp = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
    try:
        tmp.fit(X_train, y_train)
        val_preds[name] = tmp.predict(X_val)
    except:
        val_preds[name] = model.predict(X_val)

vp_arr = np.column_stack([val_preds[n] for n in fitted_models])
tp_arr = np.column_stack([predictions[n] for n in fitted_models])


def obj(wts):
    wts = np.abs(wts)
    wts /= wts.sum()
    return np.sqrt(mean_squared_error(y_val, vp_arr @ wts))


res = minimize(
    obj,
    np.ones(len(fitted_models)) / len(fitted_models),
    method="Nelder-Mead",
    options={"maxiter": 2000},
)
opt_w = np.abs(res.x)
opt_w /= opt_w.sum()
opt_pred = tp_arr @ opt_w

results.append(metrics("Optimized Blend", y_test.values, opt_pred))
predictions["Optimized Blend"] = opt_pred

print("Optimized weights:")
for name, wt in zip(fitted_models.keys(), opt_w):
    print(f"  {name:20s}: {wt:.4f}")
print(f'\nOptimized Blend  R²={results[-1]["R2"]:.4f}  RMSE={results[-1]["RMSE"]:.4f}')

# ---

# ── 7.3  Stacking with LightGBM meta-learner ─────────────────────────────────
from sklearn.model_selection import TimeSeriesSplit

STACK_MODELS = ["Random Forest", "XGBoost", "LightGBM", "Extra Trees"]
tscv = TimeSeriesSplit(n_splits=5)
oof = np.zeros((len(X_tv), len(STACK_MODELS)))

for fold, (tr_idx, oof_idx) in enumerate(tscv.split(X_tv)):
    Xtr, Xoo = X_tv.iloc[tr_idx], X_tv.iloc[oof_idx]
    ytr, yoo = y_tv.iloc[tr_idx], y_tv.iloc[oof_idx]
    for j, sname in enumerate(STACK_MODELS):
        if sname == "XGBoost":
            tmp = xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                n_jobs=-1,
                verbosity=0,
                random_state=RANDOM_STATE,
            )
        elif sname == "LightGBM":
            tmp = lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                n_jobs=-1,
                verbosity=-1,
                random_state=RANDOM_STATE,
            )
        else:
            tmp = type(fitted_models[sname])(**fitted_models[sname].get_params())
        tmp.fit(Xtr, ytr)
        oof[oof_idx, j] = tmp.predict(Xoo)
    print(f"  Fold {fold+1}/5 done")

# LightGBM meta-learner (better than Ridge for non-linear blending)
meta = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    verbosity=-1,
    random_state=RANDOM_STATE,
)
meta.fit(oof, y_tv)

test_stack = np.column_stack([predictions[n] for n in STACK_MODELS])
stack_pred = meta.predict(test_stack)
results.append(metrics("LightGBM Stacking", y_test.values, stack_pred))
predictions["LightGBM Stacking"] = stack_pred
print(
    f'\nLightGBM Stacking  R²={results[-1]["R2"]:.4f}  RMSE={results[-1]["RMSE"]:.4f}'
)

# ---

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
best_name = results_df.iloc[0]["Model"]
best_pred = predictions[best_name]

disp = results_df.copy()
for c in ["MAE", "RMSE"]:
    disp[c] = disp[c].map("{:.4f}".format)
disp["R2"] = disp["R2"].map("{:.4f}".format)
disp["MAPE"] = disp["MAPE"].map("{:.2f}%".format)
disp["Accuracy"] = disp["Accuracy"].map("{:.2f}%".format)

print("=" * 70)
print("  MODEL COMPARISON — TEST SET")
print("=" * 70)
print(disp[["Model", "R2", "RMSE", "MAE", "MAPE", "Accuracy"]].to_string(index=True))
print(
    f'\n🏆 Best model: {best_name}  R²={results_df.iloc[0]["R2"]:.4f}  '
    f'RMSE={results_df.iloc[0]["RMSE"]:.4f}  '
    f'MAPE={results_df.iloc[0]["MAPE"]:.2f}%'
)

# ---

# ── 9.1  Model comparison ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#0f1117")
fig.suptitle(
    "Model Performance Comparison — Test Set",
    fontsize=14,
    color="#e8eaf6",
    fontweight="bold",
)

for ax, (metric, ascending) in zip(
    axes, [("R2", False), ("RMSE", True), ("MAPE", True)]
):
    df_plot = results_df.sort_values(metric, ascending=ascending)
    colors = [
        PAL[0] if r["Model"] == best_name else "#3a3d52" for _, r in df_plot.iterrows()
    ]
    bars = ax.barh(
        df_plot["Model"], df_plot[metric], color=colors, alpha=0.9, height=0.6
    )
    for bar, v in zip(bars, df_plot[metric]):
        ax.text(
            bar.get_width() * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontsize=8,
            color="#c8cce8",
        )
    ax.set_title(metric, fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(
    "/home/claude/comparison.png", dpi=150, bbox_inches="tight", facecolor="#0f1117"
)
plt.show()

# ---

# ── 9.2  Actual vs Predicted (best model) ────────────────────────────────────
SHOW = min(30 * 24, len(dates_test))  # 30 days
t_ = dates_test[-SHOW:]
act = y_test.values[-SHOW:]
prd = best_pred[-SHOW:]
err = prd - act

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f1117")
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

# Time series
ax = fig.add_subplot(gs[0, :])
ax.plot(t_, act, color=PAL[0], lw=1.0, label="Actual", alpha=0.95)
ax.plot(
    t_, prd, color=PAL[1], lw=0.8, ls="--", label=f"Predicted ({best_name})", alpha=0.9
)
ax.fill_between(t_, act, prd, alpha=0.1, color=PAL[3])
ax.set_title("Actual vs Predicted — Last 30 Days", fontsize=12)
ax.set_ylabel("Power (kW)")
ax.legend(facecolor="#1a1d27", edgecolor="#2d3147")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# Residuals
ax = fig.add_subplot(gs[1, :])
ax.bar(t_, err, color=np.where(err > 0, PAL[1], PAL[2]), width=0.03, alpha=0.7)
ax.axhline(0, color="white", lw=0.8, ls="--")
ax.axhline(err.std(), color=PAL[3], lw=0.6, ls=":", label=f"+1σ={err.std():.3f}")
ax.axhline(-err.std(), color=PAL[3], lw=0.6, ls=":")
ax.set_title("Residuals (Predicted − Actual)")
ax.set_ylabel("Error (kW)")
ax.legend(facecolor="#1a1d27", edgecolor="#2d3147", fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

# Scatter
ax = fig.add_subplot(gs[2, 0])
ax.scatter(act, prd, alpha=0.2, s=5, color=PAL[0])
lim = [min(act.min(), prd.min()) - 0.1, max(act.max(), prd.max()) + 0.1]
ax.plot(lim, lim, "w--", lw=1, alpha=0.6, label="Perfect")
ax.set_xlabel("Actual (kW)")
ax.set_ylabel("Predicted (kW)")
ax.set_title("Scatter: Actual vs Predicted")
ax.legend(facecolor="#1a1d27", edgecolor="#2d3147")
ax.grid(True, alpha=0.3)

# Error distribution
ax = fig.add_subplot(gs[2, 1])
full_err = best_pred - y_test.values
ax.hist(full_err, bins=80, color=PAL[2], alpha=0.8, edgecolor="none")
ax.axvline(
    full_err.mean(), color=PAL[1], lw=1.5, ls="--", label=f"Mean={full_err.mean():.4f}"
)
ax.set_xlabel("Error (kW)")
ax.set_ylabel("Count")
ax.set_title("Error Distribution (full test set)")
ax.legend(facecolor="#1a1d27", edgecolor="#2d3147")
ax.grid(True, alpha=0.3, axis="y")

plt.savefig("/home/claude/avp.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# ---

# ── 9.3  Peak demand ─────────────────────────────────────────────────────────
cdf = pd.DataFrame({"actual": y_test.values, "predicted": best_pred}, index=dates_test)
thr = cdf["actual"].quantile(0.90)
cdf["peak"] = cdf["actual"] > thr

rec = cdf.tail(30 * 24)
fig, ax = plt.subplots(figsize=(18, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")
ax.plot(rec.index, rec["actual"], color=PAL[0], lw=0.9, label="Actual")
ax.plot(rec.index, rec["predicted"], color=PAL[1], lw=0.7, ls="--", label="Predicted")
ax.fill_between(
    rec.index,
    rec["actual"],
    where=rec["peak"],
    color=PAL[3],
    alpha=0.4,
    label=f"Peak (>{thr:.2f} kW)",
)
ax.axhline(thr, color=PAL[3], lw=0.8, ls=":")
ax.set_title(
    "⚡ Peak Demand Detection — Last 30 Days",
    fontsize=13,
    color="#e8eaf6",
    fontweight="bold",
)
ax.set_ylabel("Power (kW)")
ax.legend(facecolor="#1a1d27", edgecolor="#2d3147")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
plt.tight_layout()
plt.savefig("/home/claude/peak.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()

# ---

# ── 9.4  Feature importance ───────────────────────────────────────────────────
fi_name = "LightGBM" if "LightGBM" in fitted_models else "XGBoost"
fi_model = fitted_models[fi_name]
if hasattr(fi_model, "feature_importances_"):
    fi_df = pd.DataFrame({"feature": X.columns, "imp": fi_model.feature_importances_})
    fi_df = fi_df.sort_values("imp", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    colors_fi = [PAL[i % len(PAL)] for i in range(len(fi_df))]
    ax.barh(
        fi_df["feature"][::-1],
        fi_df["imp"][::-1],
        color=colors_fi[::-1],
        alpha=0.85,
        height=0.7,
    )
    ax.set_title(
        f"Top 30 Features — {fi_name}", fontsize=13, color="#e8eaf6", fontweight="bold"
    )
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        "/home/claude/feat_imp.png", dpi=150, bbox_inches="tight", facecolor="#0f1117"
    )
    plt.show()
    print("\nTop 10 features:")
    print(fi_df.head(10).to_string(index=False))

# ---

# ── 9.5  Error by hour & day ──────────────────────────────────────────────────
cdf["hour"] = cdf.index.hour
cdf["dow"] = cdf.index.dayofweek
cdf["abs_err"] = np.abs(cdf["actual"] - cdf["predicted"])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor("#0f1117")
fig.suptitle("Error Analysis", fontsize=13, color="#e8eaf6", fontweight="bold")

for ax, (col, xlabel, xticks) in zip(
    axes,
    [
        ("hour", "Hour of Day", range(0, 24)),
        ("dow", "Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    ],
):
    grp = cdf.groupby(col)["abs_err"].mean()
    ax.bar(range(len(grp)), grp.values, color=PAL[0], alpha=0.85, width=0.7)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(xticks, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE (kW)")
    ax.set_title(f"MAE by {xlabel}", color="#e8eaf6")
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    "/home/claude/err_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0f1117"
)
plt.show()

# ---

best = results_df.iloc[0]
print("=" * 68)
print("  SMART POWER USAGE PREDICTION — FINAL RESULTS (v2)")
print("=" * 68)
print(f"  Dataset   : UCI Individual Household Electric Power Consumption")
print(f"  Resolution: Hourly (resampled from 1-min)")
print(f"  Target    : Mean power over next 6 hours (strictly future)")
print(f"  Features  : {X.shape[1]} (lags, rolling stats, cyclical, Fourier, blocks)")
print(f"  Test set  : {len(y_test):,} samples")
print()
print(f'  🏆 Best Model : {best["Model"]}')
print(f'     R²        : {best["R2"]:.4f}')
print(f'     RMSE      : {best["RMSE"]:.4f} kW')
print(f'     MAE       : {best["MAE"]:.4f} kW')
print(f'     MAPE      : {best["MAPE"]:.2f}%')
print(f'     Accuracy  : {best["Accuracy"]:.2f}%')
print()
print("  All Models:")
print(
    results_df[["Model", "R2", "RMSE", "MAE", "MAPE", "Accuracy"]].to_string(
        index=False
    )
)
print("=" * 68)

# ---

