from __future__ import annotations


from .config import (
    ARTIFACT_PATH,
    COMPARISON_PATH,
    METRICS_PATH,
    PEAK_PERIODS_PATH,
)
from .electric_power_ml import train_electric_power_models
from .pipeline import load_artifacts, predict_next_6_hours, save_artifacts


def run_training() -> dict:
    return run_training_with_options()


def run_training_with_options(
    model_profile: str = "balanced",  # kept for API compatibility; unused
    max_rows: int | None = None,
) -> dict:
    del model_profile  # electric_power_ml.ipynb does not use note.ipynb profiles

    trained = train_electric_power_models(max_rows=max_rows)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained["results_df"].to_csv(METRICS_PATH, index=False)
    trained["comparison_df"].to_csv(COMPARISON_PATH)
    trained["peak_periods"].to_csv(PEAK_PERIODS_PATH)

    save_artifacts(ARTIFACT_PATH, trained)
    return trained


if __name__ == "__main__":
    output = run_training()
    next_6h = predict_next_6_hours(output)

    print("Training complete.")
    print(f"Best model: {output['best_model_name']}")
    print("Top metrics:")
    print(output["results_df"].head(5).to_string(index=False))
    print("Saved:")
    print(f"- {ARTIFACT_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {COMPARISON_PATH}")
    print(f"- {PEAK_PERIODS_PATH}")
    if "model_paths" in output:
        for name, path in output["model_paths"].items():
            print(f"- {name}: {path}")
    print("Forecast summary:")
    print(next_6h)
