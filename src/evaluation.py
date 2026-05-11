"""Model evaluation helpers for Marketing Mix Modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calculate_r2(y_true, y_pred) -> float:
    """Calculate R-squared."""
    return float(r2_score(y_true, y_pred))


def calculate_rmse(y_true, y_pred) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mape(y_true, y_pred) -> float:
    """Calculate mean absolute percentage error."""
    return float(mean_absolute_percentage_error(y_true, y_pred))


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the core regression metrics used in the MMM notebook."""
    return {
        "R2": calculate_r2(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }


def metrics_to_frame(metrics: dict[str, float]) -> pd.DataFrame:
    """Convert a metrics dictionary into a report-friendly DataFrame."""
    return pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }
    )
