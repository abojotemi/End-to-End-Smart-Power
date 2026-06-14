from __future__ import annotations

from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import Any

from src.config import artifact_path_for_interval
from src.pipeline import load_artifacts
from src.train import run_training_with_options

app = FastAPI(title="Smart Power Forecast API", version="1.0.0")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/train")
def train(
    max_rows: int | None = None,
) -> dict[str, Any]:
    try:
        all_artifacts = run_training_with_options(
            max_rows=max_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Training complete for all intervals",
        "intervals": list(all_artifacts.keys())
    }

@app.get("/forecast")
def forecast(interval: str = "1hr") -> dict:
    """Return the latest artifact forecast summary for a given interval."""
    artifact_path = artifact_path_for_interval(interval)
    try:
        artifact = load_artifacts(artifact_path)
    except FileNotFoundError:
        all_artifacts = run_training_with_options(max_rows=180_000)
        artifact = all_artifacts.get(interval)
        if not artifact:
            raise HTTPException(status_code=400, detail=f"Invalid interval: {interval}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "interval": interval,
        "predicted_peak_hour_next_day": artifact.get("predicted_peak_hour_next_day"),
        "backup_power_time_window": artifact.get("backup_power_time_window"),
        "predicted_peak": artifact.get("predicted_peak", False),
        "artifact_path": str(artifact_path),
    }

@app.get("/metrics")
def metrics(interval: str = "1hr") -> Any:
    artifact_path = artifact_path_for_interval(interval)
    try:
        artifact = load_artifacts(artifact_path)
    except FileNotFoundError:
        all_artifacts = run_training_with_options(max_rows=180_000)
        artifact = all_artifacts.get(interval)
        if not artifact:
            raise HTTPException(status_code=400, detail=f"Invalid interval: {interval}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = artifact["results_df"].copy()
    if "best_metrics" in artifact:
        best_row = pd.DataFrame([artifact["best_metrics"]])
        payload = pd.concat([payload, best_row], ignore_index=True, sort=False)
        payload = payload.drop_duplicates(subset=["Model"], keep="first")

    return payload.to_dict(orient="records")
