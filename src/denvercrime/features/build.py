"""Leakage-safe features for the (cell x week) panel.

Every feature for week t is computed only from counts in weeks <= t - horizon. The panel is
complete and sorted (cell, week), so each count column reshapes to a (cells x weeks) matrix
and all lags/rolling windows are plain array shifts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from denvercrime.config import FeatureConfig
from denvercrime.features.panel import count_column

CALENDAR_FEATURES = ("week_sin", "week_cos", "month")
CELL_FEATURES = ("cell_lat", "cell_lon")


def _shift(m: np.ndarray, k: int) -> np.ndarray:
    """Value from k weeks earlier; NaN where that week is before the panel starts."""
    out = np.full_like(m, np.nan, dtype=np.float64)
    if k < m.shape[1]:
        out[:, k:] = m[:, : m.shape[1] - k]
    return out


def _rolling_mean(m: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over `window` columns ending at each column.

    NaN unless all `window` values are present (the leading NaNs left by a shift propagate).
    """
    out = np.full_like(m, np.nan, dtype=np.float64)
    if window > m.shape[1]:
        return out
    valid = ~np.isnan(m)
    pad = np.zeros((m.shape[0], 1))
    csum = np.cumsum(np.concatenate([pad, np.where(valid, m, 0.0)], axis=1), axis=1)
    ccount = np.cumsum(np.concatenate([pad, valid], axis=1), axis=1)
    sums = csum[:, window:] - csum[:, :-window]
    counts = ccount[:, window:] - ccount[:, :-window]
    out[:, window - 1 :] = np.where(counts == window, sums / window, np.nan)
    return out


def _expanding_mean(m: np.ndarray) -> np.ndarray:
    """Mean of all non-NaN values up to and including each column."""
    valid = ~np.isnan(m)
    total = np.cumsum(np.where(valid, m, 0.0), axis=1)
    count = np.cumsum(valid, axis=1)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def check_panel(panel: pd.DataFrame) -> tuple[list[str], pd.DatetimeIndex]:
    """Assert the panel is complete and sorted by (cell, week); return its cells and weeks."""
    cells = list(dict.fromkeys(panel["cell"]))
    weeks = pd.DatetimeIndex(panel["week"].iloc[: len(panel) // max(len(cells), 1)])
    expected = pd.MultiIndex.from_product([cells, weeks])
    actual = pd.MultiIndex.from_arrays([panel["cell"], panel["week"]])
    if len(panel) != len(cells) * len(weeks) or not actual.equals(expected):
        raise ValueError("panel must contain every (cell, week) pair, sorted by cell then week")
    if len(weeks) > 1 and not (np.diff(weeks.values) == np.timedelta64(7, "D")).all():
        raise ValueError("panel weeks must be consecutive")
    return cells, weeks


def build_features(
    panel: pd.DataFrame,
    groups: list[str],
    cfg: FeatureConfig,
    neighbors: np.ndarray,
    cell_info: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the panel with feature columns added, and the list of feature names.

    neighbors: row-normalised (cells x cells) matrix aligned with the panel's cell order.
    cell_info: indexed by cell, with at least `cell_lat` and `cell_lon`.
    """
    cells, weeks = check_panel(panel)
    n_cells, n_weeks = len(cells), len(weeks)
    if neighbors.shape != (n_cells, n_cells):
        raise ValueError("neighbor matrix does not match the panel's cells")
    h = cfg.horizon
    new: dict[str, np.ndarray] = {}

    for group in groups:
        y = panel[count_column(group)].to_numpy(dtype=np.float64).reshape(n_cells, n_weeks)
        base = _shift(y, h)  # the most recent week the forecaster is allowed to see
        for k in cfg.lags:
            new[f"{group}_lag{k}"] = _shift(y, k)
        for w in cfg.rolling_windows:
            new[f"{group}_mean{w}"] = _rolling_mean(base, w)
        new[f"{group}_expanding_mean"] = _expanding_mean(base)

        # Spatial lags: average over neighbouring cells (NaN until the history exists).
        nbr_lag = neighbors @ np.nan_to_num(base, nan=0.0)
        nbr_lag[:, :h] = np.nan
        nbr_mean4 = neighbors @ np.nan_to_num(_rolling_mean(base, 4), nan=0.0)
        nbr_mean4[:, : h + 3] = np.nan
        new[f"{group}_nbr_lag{h}"] = nbr_lag
        new[f"{group}_nbr_mean4"] = nbr_mean4

        city = _shift(y.sum(axis=0, keepdims=True), h)
        new[f"{group}_city_lag{h}"] = np.repeat(city, n_cells, axis=0)

    features = pd.DataFrame({name: m.reshape(-1) for name, m in new.items()}, index=panel.index)

    week_of_year = panel["week"].dt.isocalendar().week.astype("float64")
    features["week_sin"] = np.sin(2 * np.pi * week_of_year / 52.1775)
    features["week_cos"] = np.cos(2 * np.pi * week_of_year / 52.1775)
    features["month"] = panel["week"].dt.month.astype("int8")
    info = cell_info.reindex(panel["cell"])
    for col in CELL_FEATURES:
        features[col] = info[col].to_numpy()

    out = pd.concat([panel, features], axis=1)
    return out, list(features.columns)
