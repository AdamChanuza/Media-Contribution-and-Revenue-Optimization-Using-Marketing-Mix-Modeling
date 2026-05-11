"""Media transformation utilities for Marketing Mix Modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_adstock(values, decay: float) -> np.ndarray:
    """Apply geometric adstock to model media carryover effects."""
    x = np.asarray(values, dtype=float)
    result = np.zeros(len(x), dtype=float)

    for idx, value in enumerate(x):
        if idx == 0:
            result[idx] = value
        else:
            result[idx] = value + decay * result[idx - 1]

    return result


def apply_hill_saturation(values, alpha: float, gamma: float) -> np.ndarray:
    """Apply Hill saturation to model diminishing marginal returns."""
    x = np.asarray(values, dtype=float)
    max_value = np.max(x)

    if max_value == 0:
        return x

    x_scaled = x / max_value
    numerator = np.power(x_scaled, alpha)
    denominator = numerator + np.power(gamma, alpha)

    return numerator / denominator


def add_mmm_transformations(
    df: pd.DataFrame,
    media_columns: list[str],
    decay: float,
    alpha: float,
    gamma: float,
) -> pd.DataFrame:
    """Add adstocked and saturated media features to a DataFrame."""
    df_transformed = df.copy()

    for col in media_columns:
        adstock_col = f"{col}_adstock"
        saturated_col = f"{col}_adstock_saturated"

        df_transformed[adstock_col] = apply_adstock(
            df_transformed[col].fillna(0).to_numpy(),
            decay=decay,
        )
        df_transformed[saturated_col] = apply_hill_saturation(
            df_transformed[adstock_col].to_numpy(),
            alpha=alpha,
            gamma=gamma,
        )

    return df_transformed
