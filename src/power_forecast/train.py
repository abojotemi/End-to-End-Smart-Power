from __future__ import annotations


from .config import (
    ARTIFACT_PATH,
    COMPARISON_PATH,
    METRICS_PATH,
    PEAK_PERIODS_PATH,
)
from .pipeline import (
    build_model_frame,
    load_raw_data,
    predict_next_hour,
    preprocess_hourly,
    save_artifacts,
    train_and_evaluate,
)


def run_training() -> dict:
    raw_df = load_raw_data()
    df_hourly = preprocess_hourly(raw_df)
    model_df = build_model_frame(df_hourly)

    trained = train_and_evaluate(model_df)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained["results_df"].to_csv(METRICS_PATH, index=False)
    trained["comparison_df"].to_csv(COMPARISON_PATH)
    trained["peak_periods"].to_csv(PEAK_PERIODS_PATH)

    save_artifacts(ARTIFACT_PATH, trained)
    return trained


if __name__ == "__main__":
    output = run_training()
    next_hour = predict_next_hour(output)

    print("Training complete.")
    print(f"Best model: {output['best_model_name']}")
    print("Saved:")
    print(f"- {ARTIFACT_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {COMPARISON_PATH}")
    print(f"- {PEAK_PERIODS_PATH}")
    print("Next-hour forecast:")
    print(next_hour)
