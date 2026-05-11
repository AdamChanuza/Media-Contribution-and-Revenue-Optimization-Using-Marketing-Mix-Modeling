"""Reusable plotting functions for the MMM project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


CHANNEL_LABELS = {
    "tv_S": "TV",
    "ooh_S": "Out-of-Home",
    "print_S": "Print",
    "facebook_S": "Facebook",
    "search_S": "Paid Search",
}


def _save_figure(fig, save_path=None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_revenue_trend(df, date_col: str, target_col: str, save_path=None):
    """Plot weekly revenue trend and seasonality."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[date_col], df[target_col], linewidth=2)
    ax.set_title("Weekly Revenue Trend and Seasonality")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_media_spend_by_channel(df, date_col: str, media_spend_cols, output_dir=None):
    """Plot individual spend trends for each paid media channel."""
    figures = {}

    for col in media_spend_cols:
        fig, ax = plt.subplots(figsize=(12, 4))
        channel_name = CHANNEL_LABELS.get(col, col)
        ax.plot(df[date_col], df[col], linewidth=1.8)
        ax.set_title(f"{channel_name} Spend Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Spend")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        save_path = None
        if output_dir is not None:
            save_path = Path(output_dir) / f"{col}_spend.png"
        _save_figure(fig, save_path)
        figures[col] = (fig, ax)

    return figures


def plot_correlation_heatmap(corr, save_path=None):
    """Plot a correlation heatmap for numeric MMM variables."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Revenue and Marketing Variables")
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_actual_vs_predicted(results_plot, save_path=None):
    """Plot actual and predicted revenue for the validation period."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        results_plot["Date"],
        results_plot["Actual Revenue"],
        label="Actual Revenue",
        linewidth=2,
    )
    ax.plot(
        results_plot["Date"],
        results_plot["Predicted Revenue"],
        label="Predicted Revenue",
        linewidth=2,
    )
    ax.set_title("Actual vs Predicted Revenue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_channel_contribution(coefficients, save_path=None):
    """Plot model-based channel contribution coefficients."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(coefficients["Feature"], coefficients["Coefficient"])
    ax.set_title("Marketing Channel Contribution")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax


def plot_channel_efficiency_proxy(roi_proxy, save_path=None):
    """Plot the coefficient-to-spend efficiency proxy by channel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(roi_proxy["Channel"], roi_proxy["Efficiency Proxy"])
    ax.set_title("Channel Efficiency Proxy")
    ax.set_xlabel("Coefficient / Total Spend")
    ax.set_ylabel("Channel")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, ax
