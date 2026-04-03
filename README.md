# End-to-End Smart Power Usage Forecasting

This project upgrades your notebook prototype into a complete ML workflow with:

- **Data preprocessing + feature engineering**
- **Model training and comparison** (Deep MLP models + weighted ensemble + baselines)
- **Saved artifacts** for reuse
- **Streamlit dashboard** for interactive analytics
- **FastAPI service** for programmatic forecasts
- **Basic test coverage**

## Project Structure

- `src/`: reusable ML pipeline and training logic
- `app/streamlit_app.py`: interactive dashboard
- `app/api.py`: API endpoints for train/forecast/metrics
- `models/`: saved trained artifact (`.joblib`)
- `reports/`: generated metrics and evaluation outputs
- `tests/`: unit test(s)

## Quick Start

### Requirements

- Python **3.10+**
- `pip`

### 1. Create and activate a virtual environment

From the project root, create a virtual environment:

- Linux/macOS:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
- Windows (PowerShell):
   - `python -m venv .venv`
   - `.venv\Scripts\Activate.ps1`

After activation, you should see your shell prompt prefixed with `(.venv)`.

### 2. Install the required packages

Install the project dependencies with:

- `pip install --upgrade pip`
- `pip install -r requirements.txt`

### 3. Run the Streamlit app

Start the interactive dashboard with:

- `streamlit run app/streamlit_app.py`

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`.

### 4. Run the FastAPI service (optional)

If you want to use the API instead of, or alongside, the dashboard, run:

- `uvicorn app.api:app --reload`

The API will usually be available at `http://127.0.0.1:8000`, and the interactive docs at `http://127.0.0.1:8000/docs`.

## API Endpoints
- `GET /health`
- `POST /train?data_path=...&model_profile=fast|max_rows=180000`
- `GET /forecast/next`
- `GET /forecast/next6h`
- `GET /metrics`

## Faster local iteration (recommended during development)

- Use the local CSV in `data/household_power_consumption.csv` (default behavior when available).
- Use `model_profile=fast` and a `max_rows` cap while experimenting.
- Run full profile only for final model refresh/deployment checks.

Example (optional):

- `python measure_metrics.py --profile fast --max-rows 180000`
- `python measure_metrics.py --profile full --max-rows 0`

## Notes

- The main regressor predicts **next 6-hour average global active power (kW)**.
- Peak-demand detection uses the **90th percentile** of actual values in the test window.
- A daily peak-hour model predicts the **most likely highest-load hour for the next day** (useful for backup power scheduling).
