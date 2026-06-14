from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_EXPORT_DIR = MODELS_DIR / "saved_models"
REPORTS_DIR = ROOT_DIR / "reports"

ARTIFACT_PATH = MODELS_DIR / "power_forecast.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
COMPARISON_PATH = REPORTS_DIR / "test_comparison.csv"
PEAK_PERIODS_PATH = REPORTS_DIR / "peak_periods.csv"

RANDOM_STATE = 42

# Bump this whenever you change the training/payload schema.
# Streamlit/API will refuse to load older artifacts and will retrain instead.
ARTIFACT_SCHEMA_VERSION = "electric_power_ml_v1"

# ── Multi-interval support ────────────────────────────────────────────────────
# Supported time intervals for resampling
SUPPORTED_INTERVALS = ["30min", "1hr", "2hr", "4hr", "6hr"]


def artifact_path_for_interval(interval_key: str) -> Path:
    """Return the artifact path for a given interval."""
    return MODELS_DIR / f"power_forecast_{interval_key}.joblib"


def metrics_path_for_interval(interval_key: str) -> Path:
    """Return the metrics CSV path for a given interval."""
    return REPORTS_DIR / f"model_metrics_{interval_key}.csv"


def comparison_path_for_interval(interval_key: str) -> Path:
    """Return the test-comparison CSV path for a given interval."""
    return REPORTS_DIR / f"test_comparison_{interval_key}.csv"


def peak_periods_path_for_interval(interval_key: str) -> Path:
    """Return the peak-periods CSV path for a given interval."""
    return REPORTS_DIR / f"peak_periods_{interval_key}.csv"
