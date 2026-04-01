from __future__ import annotations


from fastapi import FastAPI, HTTPException

from src.config import ARTIFACT_PATH
from src.pipeline import load_artifacts, predict_next_6_hours
from src.train import run_training

app = FastAPI(title="Smart Power Forecast API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train() -> dict[str, str]:
    try:
        artifact = run_training()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Training complete",
        "best_model": artifact["best_model_name"],
        "artifact_path": str(ARTIFACT_PATH),
    }


@app.get("/forecast/next")
def forecast_next() -> dict:
    try:
        artifact = load_artifacts(ARTIFACT_PATH)
    except FileNotFoundError:
        artifact = run_training()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return predict_next_6_hours(artifact)


@app.get("/forecast/next6h")
def forecast_next_6h() -> dict:
    try:
        artifact = load_artifacts(ARTIFACT_PATH)
    except FileNotFoundError:
        artifact = run_training()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return predict_next_6_hours(artifact)


@app.get("/metrics")
def metrics() -> list[dict]:
    try:
        artifact = load_artifacts(ARTIFACT_PATH)
    except FileNotFoundError:
        artifact = run_training()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return artifact["results_df"].to_dict(orient="records")
