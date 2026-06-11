#!/usr/bin/env python3
"""Build a PDF report with notebook screenshots and explanations."""

from __future__ import annotations

import base64
import io
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "electric_power_ml.ipynb"
OUTPUT = ROOT / "reports" / "electric_power_ml_explained.pdf"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cell_stream(nb: dict, idx: int) -> str:
    parts: list[str] = []
    for out in nb["cells"][idx].get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out.get("text", [])))
    return "\n".join(parts).strip()


def cell_image(nb: dict, idx: int, out_idx: int = 0) -> Image.Image | None:
    outputs = nb["cells"][idx].get("outputs", [])
    img_outputs = [
        o for o in outputs if o.get("data", {}).get("image/png")
    ]
    if not img_outputs or out_idx >= len(img_outputs):
        return None
    raw = base64.b64decode(img_outputs[out_idx]["data"]["image/png"])
    return Image.open(io.BytesIO(raw))


def wrap(text: str, width: int = 95) -> str:
    paragraphs = text.strip().split("\n\n")
    wrapped: list[str] = []
    for para in paragraphs:
        lines = para.splitlines()
        if len(lines) == 1 and not para.startswith("|"):
            wrapped.append(textwrap.fill(para, width=width))
        else:
            wrapped.append(para)
    return "\n\n".join(wrapped)


def text_page(
    pdf: PdfPages,
    title: str,
    body: str,
    *,
    subtitle: str | None = None,
    fontsize: int = 10,
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    y = 0.94
    fig.text(0.08, y, title, fontsize=16, fontweight="bold", va="top")
    y -= 0.04
    if subtitle:
        fig.text(0.08, y, subtitle, fontsize=11, color="#444444", va="top")
        y -= 0.05
    fig.text(
        0.08,
        y,
        wrap(body),
        fontsize=fontsize,
        va="top",
        family="sans-serif",
        linespacing=1.45,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_page(
    pdf: PdfPages,
    title: str,
    image: Image.Image,
    caption: str,
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.96, title, fontsize=14, fontweight="bold", va="top")

    ax = fig.add_axes([0.06, 0.22, 0.88, 0.70])
    ax.imshow(image)
    ax.axis("off")

    fig.text(
        0.08,
        0.16,
        wrap(caption, width=100),
        fontsize=9,
        va="top",
        color="#333333",
        linespacing=1.4,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> Path:
    nb = load_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    peak_stats = cell_stream(nb, 26)
    model_summary = cell_stream(nb, 23)
    feat_stats = cell_stream(nb, 12)
    split_stats = cell_stream(nb, 14)
    clean_stats = cell_stream(nb, 8)

    with PdfPages(OUTPUT) as pdf:
        text_page(
            pdf,
            "Individual Household Electric Power Consumption",
            (
                "This document explains the electric_power_ml.ipynb notebook, which trains "
                "machine-learning models on the UCI Household Power Consumption dataset "
                "(dataset #235).\n\n"
                "Goal: predict Global_active_power — household electricity consumption in kilowatts.\n\n"
                "Pipeline overview:\n"
                "  1. Load raw minute-level data from UCI (~2M rows, 2006–2010)\n"
                "  2. Clean missing values and build a datetime index\n"
                "  3. Engineer 57 predictive features (temporal, lag, rolling, physics)\n"
                "  4. Train multiple regressors and compare them on a chronological test set\n"
                "  5. Detect peak-demand periods and predict next-day peak time-of-day\n\n"
                "Screenshots in this report are taken directly from notebook outputs."
            ),
            subtitle="Notebook walkthrough — feature engineering & peak-period prediction",
        )

        text_page(
            pdf,
            "1. Data Loading & Cleaning",
            (
                "The notebook downloads the UCI dataset via ucimlrepo. Each row is a one-minute "
                "measurement with Date, Time, Global_active_power, Global_reactive_power, Voltage, "
                "Global_intensity, and three sub-metering channels.\n\n"
                "Cleaning steps:\n"
                "  • Replace '?' placeholders with NaN and cast numeric columns\n"
                "  • Parse Date + Time into a datetime index (day-first format)\n"
                "  • Drop rows with missing target (Global_active_power)\n"
                "  • Forward-fill then median-fill remaining gaps in numeric columns\n\n"
                f"Notebook output:\n{clean_stats}"
            ),
        )

        dist_img = cell_image(nb, 9)
        if dist_img:
            image_page(
                pdf,
                "1.1 Target Variable Distribution",
                dist_img,
                "Distribution of Global_active_power after cleaning. Most readings are low "
                "(typical standby/appliance use), with a long right tail for high-load periods.",
            )

        text_page(
            pdf,
            "2. Feature Engineering (Detailed)",
            (
                "The engineer_features() function expands 7 raw sensor columns into 57 model "
                "inputs. Rows with insufficient lag history are dropped (~10,080 minutes for "
                "the one-week lag).\n\n"
                "A. Temporal features\n"
                "  • Calendar: hour, minute, day-of-week, day-of-year, month, year, quarter\n"
                "  • is_weekend flag and season (Winter/Spring/Summer/Autumn)\n"
                "  • time_of_day bucket: Night (0–5h), Morning (6–11h), Afternoon (12–17h), "
                "Evening (18–23h)\n"
                "  • Cyclical sin/cos encodings for hour, day-of-week, month, and day-of-year "
                "so 23:59 is close to 00:00 in feature space\n\n"
                "B. Physics-derived features\n"
                "  • apparent_power = Voltage × Global_intensity / 1000\n"
                "  • power_factor = Global_active_power / apparent_power (clipped 0–1)\n"
                "  • sub_total = sum of Sub_metering_1/2/3\n"
                "  • unmeasured = active power converted to Wh/min minus sub-metered usage\n"
                "  • reactive_ratio = Global_reactive_power / Global_active_power\n\n"
                "C. Lag features (minutes)\n"
                "  • lag_1, lag_2, lag_3, lag_5, lag_10, lag_30, lag_60, lag_1440 (1 day), "
                "lag_10080 (1 week)\n\n"
                "D. Rolling statistics (windows: 5, 15, 60, 1440 minutes)\n"
                "  • roll_mean, roll_std, roll_min, roll_max for each window\n\n"
                "E. Interaction features\n"
                "  • hour_x_weekend, hour_x_season, voltage_x_hour\n\n"
                f"Notebook output:\n{feat_stats}"
            ),
        )

        corr_img = cell_image(nb, 31)
        if corr_img:
            image_page(
                pdf,
                "2.1 Feature Correlation Heatmap",
                corr_img,
                "Correlation matrix of engineered features. Strong correlations among lags and "
                "rolling means are expected; tree models (XGBoost, LightGBM) handle this well.",
            )

        feat_imp_img = cell_image(nb, 28)
        if feat_imp_img:
            image_page(
                pdf,
                "2.2 Top Feature Importances (LightGBM)",
                feat_imp_img,
                "LightGBM feature importances for the top 25 inputs. Recent lags and short-window "
                "rolling statistics dominate — past consumption is the strongest predictor of "
                "next-minute usage.",
            )

        text_page(
            pdf,
            "3. Train / Test Split & Model Training",
            (
                "The notebook uses a chronological 80/20 split (no shuffling) so the last 20% "
                "of time simulates real forecasting on unseen future data.\n\n"
                f"{split_stats}\n\n"
                "Models trained:\n"
                "  • Ridge Regression (scaled features) — linear baseline\n"
                "  • Random Forest Regressor — bagged trees, limited depth\n"
                "  • XGBoost — gradient boosted trees with early stopping on test set\n"
                "  • LightGBM — fast gradient boosting, also with early stopping\n"
                "  • Stacking Ensemble — simple average of RF + XGBoost + LightGBM\n\n"
                f"{model_summary}\n\n"
                "LightGBM achieves the best balance of accuracy among the tree models. Ridge "
                "shows near-perfect R² because lag features make the problem nearly "
                "autoregressive at minute resolution."
            ),
        )

        r2_img = cell_image(nb, 24)
        if r2_img:
            image_page(
                pdf,
                "3.1 Model Comparison (R²)",
                r2_img,
                "Bar chart comparing R² across all trained models on the held-out test period.",
            )

        scatter_img = cell_image(nb, 25)
        if scatter_img:
            image_page(
                pdf,
                "3.2 Actual vs Predicted (Best Model)",
                scatter_img,
                "Scatter plot of actual vs predicted Global_active_power for the best-performing "
                "model. Points along the diagonal indicate accurate forecasts.",
            )

        ts_img = cell_image(nb, 27)
        if ts_img:
            image_page(
                pdf,
                "3.3 Time-Series Forecast (Last 7 Days)",
                ts_img,
                "Seven-day overlay of actual and predicted power. The model tracks daily cycles "
                "and spikes reasonably well on unseen data.",
            )

        text_page(
            pdf,
            "4. Peak-Period Prediction (How It Works)",
            (
                "Peak analysis happens in two related stages:\n\n"
                "Stage A — Minute-level peak detection\n"
                "  After forecasting power on the test set, the notebook marks a minute as a "
                "'peak period' when actual consumption ≥ the 90th percentile of actual test "
                "values. This flags the top 10% highest-demand minutes for backup-power planning.\n\n"
                "Stage B — Next-day peak time-of-day classification\n"
                "Instead of predicting an exact hour (24 classes, hard to learn), the notebook:\n"
                "  1. Resamples cleaned data to hourly averages\n"
                "  2. For each calendar day, computes day_mean, day_max, day_std, and peak_hour "
                "(hour with maximum hourly consumption)\n"
                "  3. Maps peak_hour to one of four time-of-day buckets:\n"
                "       0 = Night   (00:00–05:59)\n"
                "       1 = Morning (06:00–11:59)\n"
                "       2 = Afternoon (12:00–17:59)\n"
                "       3 = Evening (18:00–23:59)\n"
                "  4. Adds calendar features (day_of_week, month, is_weekend) and previous-day "
                "stats (prev_day_mean, prev_day_max, prev_day_std)\n"
                "  5. Trains a RandomForestClassifier (350 trees) on 80% of days, tests on 20%\n"
                "  6. Predicts tomorrow's peak time-of-day from the latest day's features\n\n"
                "The recommended backup window is derived from the predicted bucket, e.g. "
                "Morning → 06:00–11:59.\n\n"
                f"Notebook output:\n{peak_stats}"
            ),
        )

        peak_img = cell_image(nb, 26)
        if peak_img:
            image_page(
                pdf,
                "4.1 Peak Periods in Test Window",
                peak_img,
                "Red markers show detected peak minutes (top 10% of actual consumption). The "
                "dashed line is the 90th-percentile threshold. This visualization links "
                "high-demand events to the forecasting model's test period.",
            )

        text_page(
            pdf,
            "5. Artifacts & Deployment",
            (
                "The final notebook cell saves:\n"
                "  • models/power_forecast.joblib — trained regressors, scaler, feature list\n"
                "  • reports/model_comparison.csv — metrics table\n"
                "  • reports/peak_periods.csv — detected peak timestamps\n\n"
                "The same logic is mirrored in src/electric_power_ml.py for programmatic "
                "retraining. The Streamlit app loads the saved artifact for live predictions "
                "and displays the predicted backup-power window.\n\n"
                "Key takeaway: rich feature engineering (especially lags and rolling stats) "
                "drives minute-level accuracy, while daily aggregation plus a 4-class "
                "time-of-day classifier provides a stable, actionable peak-period forecast."
            ),
        )

        d = pdf.infodict()
        d["Title"] = "Electric Power ML Notebook Explained"
        d["Author"] = "End-to-End Smart Power"
        d["Subject"] = "Feature engineering and peak-period prediction"

    print(f"Wrote {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    main()
