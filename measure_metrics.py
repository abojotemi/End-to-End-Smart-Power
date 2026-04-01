#!/usr/bin/env python
"""Measure current model performance on real data."""

from src.pipeline import load_raw_data, preprocess_hourly, build_model_frame, train_and_evaluate
import pandas as pd

# Load and train on real data
raw_df = load_raw_data()
print(f'Raw data shape: {raw_df.shape}')

df_hourly = preprocess_hourly(raw_df)
print(f'Hourly data shape: {df_hourly.shape}')

model_df = build_model_frame(df_hourly)
print(f'Model data shape: {model_df.shape}')

result = train_and_evaluate(model_df, df_hourly)
print('\n=== Model Performance ===')
print(result['results_df'].to_string())
print(f'\nBest model: {result["best_model_name"]}')
print(f'Daily peak hour accuracy: {result["daily_peak_accuracy"]:.4f}')

# Identify which metrics need improvement
best_rae = result['results_df'].loc[0, 'RAE']
best_acc = result['results_df'].loc[0, 'Relative_Accuracy']
print(f'\n=== Current Best Performance ===')
print(f'RAE: {best_rae:.4f} (Target: 0.1 or lower for 0.9 accuracy)')
print(f'Relative Accuracy: {best_acc:.4f} (Target: 0.9 or higher)')
print(f'\nGap to achieve: {0.9 - best_acc:.4f}')
