# Peak Power Period Prediction: Technical Overview

The Smart Power Usage Forecasting system doesn't just predict the average power consumption; it also attempts to predict *when* the highest demand (the peak power period) will occur during the next day. This capability is critical for proactive load management, scheduling battery backups, or avoiding peak time-of-use tariffs.

This document outlines the detailed mechanics of how the Peak Power Period Prediction works within the application.

## 1. Data Aggregation and Daily Profiling
The peak prediction model operates on a daily timeline. To prepare the data, the system first resamples the cleaned data into a consistent 1-hour interval frequency (`1h`). 

From this hourly dataset, the system groups the data by **date** and calculates daily statistical profiles. For each day, it extracts:
- `day_mean`: The average power consumption for the day.
- `day_max`: The absolute maximum power consumption recorded for the day.
- `day_std`: The standard deviation (volatility) of power consumption throughout the day.
- `peak_hour`: The specific hour (0-23) at which the `day_max` occurred.

## 2. Target Variable Generation: "Time of Day"
Predicting an exact hour (out of 24 possible classes) is highly prone to variance and noise. Instead, the application simplifies the problem into predicting a broader **Time of Day** block. 

The exact `peak_hour` is mapped into one of four distinct categories:
- **0: Night** (00:00 - 05:59)
- **1: Morning** (06:00 - 11:59)
- **2: Afternoon** (12:00 - 17:59)
- **3: Evening** (18:00 - 23:59)

This mapping creates a more robust classification target (`peak_time_of_day`) that the machine learning model can predict more reliably.

## 3. Feature Engineering for the Classifier
To predict *tomorrow's* peak time of day, the model needs information about the current day. The system engineers the following features for each day:
- **Temporal Features:** `day_of_week`, `month`, and an `is_weekend` flag. These are crucial because power usage patterns typically shift significantly on weekends and across different seasons.
- **Historical Lags:** The system shifts the `day_mean`, `day_max`, and `day_std` from the *previous* day (`prev_day_mean`, `prev_day_max`, `prev_day_std`). This allows the model to learn autoregressive patterns (e.g., if yesterday was highly volatile with a high peak, today might follow a similar pattern).
- **The exact `peak_hour` of the previous day.**

*Note: The target variable (`peak_time_of_day`) is excluded from the feature set to prevent data leakage.*

## 4. Model Training (Random Forest Classifier)
The dataset is chronologically split into an 80% training set and a 20% testing set. 

A **Random Forest Classifier** is trained on the engineered daily features to predict the `peak_time_of_day` category. 
- **Hyperparameters used:** `n_estimators=350` (number of trees), `max_depth=12`, and `min_samples_leaf=2`.
- Random Forests are well-suited for this task because they naturally handle non-linear relationships (like the cyclical nature of days and months) without needing complex feature scaling.

## 5. Inference and Operational Output
When generating a prediction for the "next day," the model takes the very last known daily feature vector (today's stats) and passes it through the trained Random Forest Classifier.

The model outputs a predicted class (0, 1, 2, or 3). The application then translates this prediction back into human-readable and operational insights:
1. **Predicted Peak Period Label:** Converted back to "Morning", "Afternoon", "Evening", or "Night".
2. **Representative Hour:** A single hour representing the middle of the predicted block (e.g., if Morning is predicted, it outputs `09:00` as the representative hour).
3. **Backup Power Window:** The system defines a window for potential battery discharge or grid avoidance (e.g., `06:00 - 11:59`).

## 6. Short-Term Peak Detection vs. Daily Peak Prediction
It is important to distinguish between the two types of peak analysis in the dashboard:
- **Daily Peak-Hour Model:** (Described above). A Random Forest Classifier predicting the general time-of-day block for tomorrow's highest usage.
- **Short-Term Peak Threshold Warning:** The regression models predicting the *actual power (kW)* for the next hours compare their predicted value against a static `peak_threshold` (calculated as the 90th percentile of historical power usage). If the predicted kW exceeds this threshold, the dashboard flashes a "Yes 🔴" warning for an imminent peak period. 

Together, these two systems provide both a long-term daily forecast of *when* the peak will happen, and a short-term warning if the immediate power draw is about to cross into extreme territory.
