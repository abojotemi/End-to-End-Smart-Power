#!/usr/bin/env python
"""Measure current model performance on real data."""

from __future__ import annotations

import argparse

from src.pipeline import (
    DEFAULT_DATASET_PATH,
    build_model_frame,
    load_raw_data,
    preprocess_hourly,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark smart-power models")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to local CSV dataset.",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "full"],
        default="fast",
        help="Training profile. Use fast for quick iteration.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=180_000,
        help="Use only the most recent N rows for quicker runs. <=0 uses all rows.",
    )
    return parser.parse_args()


args = parse_args()

# Load and train on real data
raw_df = load_raw_data(args.data_path)
if args.max_rows and args.max_rows > 0 and len(raw_df) > args.max_rows:
    raw_df = raw_df.tail(args.max_rows).copy()

print(f"Raw data shape: {raw_df.shape}")

df_hourly = preprocess_hourly(raw_df)
print(f"Hourly data shape: {df_hourly.shape}")

model_df = build_model_frame(df_hourly)
print(f"Model data shape: {model_df.shape}")

result = train_and_evaluate(model_df, df_hourly, model_profile=args.profile)
print("\n=== Model Performance ===")
print(result["results_df"].to_string())
print(f'\nBest model: {result["best_model_name"]}')
print(f'Daily peak hour accuracy: {result["daily_peak_accuracy"]:.4f}')

# Identify which metrics need improvement
best_rae = result["results_df"].loc[0, "RAE"]
best_acc = result["results_df"].loc[0, "Relative_Accuracy"]
print("\n=== Current Best Performance ===")
print(f"RAE: {best_rae:.4f} (Target: 0.1 or lower for 0.9 accuracy)")
print(f"Relative Accuracy: {best_acc:.4f} (Target: 0.9 or higher)")
print(f"\nGap to achieve: {0.9 - best_acc:.4f}")
