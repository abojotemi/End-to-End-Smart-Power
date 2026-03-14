# End-to-End Smart Power Usage Forecasting

This project upgrades your notebook prototype into a complete ML workflow with:

- **Data preprocessing + feature engineering**
- **Model training and comparison** (Linear Regression, Random Forest, Gradient Boosting)
- **Saved artifacts** for reuse
- **Streamlit dashboard** for interactive analytics
- **FastAPI service** for programmatic forecasts
- **Basic test coverage**

## Project Structure

- `src/power_forecast/`: reusable ML pipeline and training logic
- `app/streamlit_app.py`: interactive dashboard
- `app/api.py`: API endpoints for train/forecast/metrics
- `models/`: saved trained artifact (`.joblib`)
- `reports/`: generated metrics and evaluation outputs
- `tests/`: unit test(s)

## Quick Start

1. Create a Python environment and install dependencies from `requirements.txt`.
2. Launch Streamlit app:
   - `streamlit run app/streamlit_app.py`
3. Launch API:
   - `uvicorn app.api:app --reload`

## API Endpoints
- `GET /health`
- `POST /train?data_path=...`
- `GET /forecast/next`
- `GET /metrics`

## Notes

- The model predicts **next-hour global active power (kW)**.
- Peak-demand detection uses the **90th percentile** of actual values in the test window.
