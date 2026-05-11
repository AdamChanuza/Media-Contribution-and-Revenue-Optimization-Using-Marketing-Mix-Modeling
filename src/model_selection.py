"""Model selection helpers for the MMM Ridge Regression workflow."""

from __future__ import annotations

from itertools import product

import pandas as pd
from sklearn.linear_model import Ridge

try:
    from .evaluation import calculate_metrics
    from .transformations import add_mmm_transformations
except ImportError:  # Allows direct use from the src directory.
    from evaluation import calculate_metrics
    from transformations import add_mmm_transformations


def time_aware_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
):
    """Split data chronologically, using the final period as the test set."""
    split_idx = int(len(X) * (1 - test_size))

    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


def fit_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
) -> Ridge:
    """Fit a Ridge Regression model."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def run_parameter_grid_search(
    df: pd.DataFrame,
    target_col: str,
    media_spend_cols: list[str],
    control_cols: list[str],
    decay_values: list[float],
    alpha_values: list[float],
    gamma_values: list[float],
    ridge_alpha: float = 1.0,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Evaluate adstock and saturation parameters with time-aware validation."""
    results = []

    for decay, saturation_alpha, gamma in product(
        decay_values,
        alpha_values,
        gamma_values,
    ):
        transformed_df = add_mmm_transformations(
            df=df,
            media_columns=media_spend_cols,
            decay=decay,
            alpha=saturation_alpha,
            gamma=gamma,
        )
        feature_cols = [
            f"{col}_adstock_saturated" for col in media_spend_cols
        ] + control_cols

        X = transformed_df[feature_cols]
        y = transformed_df[target_col]
        X_train, X_test, y_train, y_test = time_aware_train_test_split(
            X,
            y,
            test_size=test_size,
        )

        model = fit_ridge_model(X_train, y_train, alpha=ridge_alpha)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        results.append(
            {
                "decay": decay,
                "alpha": saturation_alpha,
                "gamma": gamma,
                "r2": metrics["R2"],
                "rmse": metrics["RMSE"],
                "mape": metrics["MAPE"],
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )
