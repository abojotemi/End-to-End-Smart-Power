from __future__ import annotations


from fastapi import FastAPI, HTTPException
import pandas as pd

from src.config import ARTIFACT_PATH
from src.pipeline import load_artifacts
from src.train import run_training, run_training_with_options

app = FastAPI(title="Smart Power Forecast API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train(
    data_path: str | None = None,
    model_profile: str = "balanced",
    max_rows: int | None = None,
) -> dict[str, object]:
    try:
        artifact = run_training_with_options(
            data_path=data_path,
            model_profile=model_profile,
            max_rows=max_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Training complete",
        "best_model": artifact["best_model_name"],
        "best_metrics": artifact.get("best_metrics", {}),
        "model_paths": artifact.get("model_paths", {}),
        "artifact_path": str(ARTIFACT_PATH),
    }


@app.get("/forecast")
def forecast() -> dict:
    """Return the latest artifact forecast summary (peak hour and backup window).

    This endpoint replaces the older next-6h-specific endpoints and surfaces
    the artifact's summary fields.
    """
    try:
        artifact = load_artifacts(ARTIFACT_PATH)
    except FileNotFoundError:
        artifact = run_training_with_options(model_profile="fast", max_rows=180_000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "predicted_peak_hour_next_day": artifact.get("predicted_peak_hour_next_day"),
        "backup_power_time_window": artifact.get("backup_power_time_window"),
        "predicted_peak": artifact.get("predicted_peak", False),
        "artifact_path": str(ARTIFACT_PATH),
    }


@app.get("/metrics")
def metrics() -> list[dict[str, object]]:
    try:
        artifact = load_artifacts(ARTIFACT_PATH)
    except FileNotFoundError:
        artifact = run_training_with_options(model_profile="fast", max_rows=180_000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = artifact["results_df"].copy()
    if "best_metrics" in artifact:
        best_row = pd.DataFrame([artifact["best_metrics"]])
        payload = pd.concat([payload, best_row], ignore_index=True, sort=False)
        payload = payload.drop_duplicates(subset=["Model"], keep="first")

    return payload.to_dict(orient="records")
