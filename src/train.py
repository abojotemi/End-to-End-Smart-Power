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
    predict_next_6_hours,
    preprocess_hourly,
    save_artifacts,
    train_and_evaluate,
)


def run_training() -> dict:
    raw_df = load_raw_data()
    df_hourly = preprocess_hourly(raw_df)
    model_df = build_model_frame(df_hourly)

    trained = train_and_evaluate(model_df, df_hourly)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained["results_df"].to_csv(METRICS_PATH, index=False)
    trained["comparison_df"].to_csv(COMPARISON_PATH)
    trained["peak_periods"].to_csv(PEAK_PERIODS_PATH)

    save_artifacts(ARTIFACT_PATH, trained)
    return trained


def run_training_with_options(
    data_path: str | None = None,
    model_profile: str = "balanced",
    max_rows: int | None = None,
) -> dict:
    raw_df = load_raw_data(data_path)
    if max_rows is not None and max_rows > 0 and len(raw_df) > max_rows:
        raw_df = raw_df.tail(max_rows).copy()

    df_hourly = preprocess_hourly(raw_df)
    model_df = build_model_frame(df_hourly)

    trained = train_and_evaluate(
        model_df,
        df_hourly,
        model_profile=model_profile,
    )

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
    print("Saved:")
    print(f"- {ARTIFACT_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {COMPARISON_PATH}")
    print(f"- {PEAK_PERIODS_PATH}")
    print("Next-6-hour forecast and peak-hour recommendation:")
    print(next_6h)
