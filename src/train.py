from __future__ import annotations

from .config import (
    artifact_path_for_interval,
    comparison_path_for_interval,
    metrics_path_for_interval,
    peak_periods_path_for_interval,
    SUPPORTED_INTERVALS,
)
from .electric_power_ml_multi import train_all_intervals
from .pipeline import load_artifacts, predict_next_6_hours, save_artifacts

def run_training() -> dict:
    return run_training_with_options()

def run_training_with_options(
    model_profile: str = "balanced",  # kept for API compatibility; unused
    max_rows: int | None = None,
) -> dict:
    del model_profile  # electric_power_ml.ipynb does not use note.ipynb profiles

    all_trained = train_all_intervals(max_rows=max_rows)

    for interval_key, trained in all_trained.items():
        metrics_path = metrics_path_for_interval(interval_key)
        comparison_path = comparison_path_for_interval(interval_key)
        peak_periods_path = peak_periods_path_for_interval(interval_key)
        artifact_path = artifact_path_for_interval(interval_key)

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        trained["results_df"].to_csv(metrics_path, index=False)
        trained["comparison_df"].to_csv(comparison_path)
        trained["peak_periods"].to_csv(peak_periods_path)

        save_artifacts(artifact_path, trained)
        
    return all_trained

if __name__ == "__main__":
    outputs = run_training()
    for interval_key, output in outputs.items():
        print(f"\n{'='*60}")
        print(f"Interval: {interval_key}")
        print(f"{'='*60}")
        next_6h = predict_next_6_hours(output)
        print(f"Best model: {output['best_model_name']}")
        print("Top metrics:")
        print(output["results_df"].head(5).to_string(index=False))
        print("Saved Paths:")
        print(f"- {artifact_path_for_interval(interval_key)}")
        print(f"- {metrics_path_for_interval(interval_key)}")
        print(f"- {comparison_path_for_interval(interval_key)}")
        print(f"- {peak_periods_path_for_interval(interval_key)}")
        if "model_paths" in output:
            for name, path in output["model_paths"].items():
                print(f"- {name}: {path}")
        print("Forecast summary:")
        print(next_6h)
