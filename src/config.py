from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

ARTIFACT_PATH = MODELS_DIR / "power_forecast.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
COMPARISON_PATH = REPORTS_DIR / "test_comparison.csv"
PEAK_PERIODS_PATH = REPORTS_DIR / "peak_periods.csv"

RANDOM_STATE = 42
