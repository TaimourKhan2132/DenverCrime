"""Leakage-safe features for the (cell x week) panel.

Every feature for week t is computed only from counts in weeks <= t - horizon (calendar
features describe week t itself, which is known in advance). The panel is complete and sorted
(cell, week), so each count column reshapes to a (cells x weeks) matrix and all lags, rolling
windows and spatial averages are plain array operations.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from denvercrime.config import FeatureConfig
from denvercrime.features.panel import count_column

CALENDAR_FEATURES = ("week_sin", "week_cos", "month")
CELL_FEATURES = ("cell_lat", "cell_lon")
TREND_SMOOTHING = 0.1  # added to both sides of a ratio so empty cells give 1, not 0/0

# Feature group of a per-crime-group feature, by the part after "<group>_".
_SUFFIX_GROUPS = [
    (re.compile(r"^(lag\d+|mean\d+|expanding_mean)$"), "history"),
    (re.compile(r"^ewm\d+$"), "ewm"),
    (re.compile(r"^trend\d+$"), "trend"),
    (re.compile(r"^nbr_"), "spatial"),
    (re.compile(r"^nbr2_"), "outer_spatial"),
    (re.compile(r"^city_"), "city"),
]
ABLATION_GROUPS = ("cross_group", "ewm", "trend", "spatial", "outer_spatial", "city",
                   "calendar", "holidays", "location")


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


def _ewm(m: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted mean along weeks, starting at each row's first non-NaN value."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    out = np.full_like(m, np.nan, dtype=np.float64)
    state = np.full(m.shape[0], np.nan)
    for t in range(m.shape[1]):
        x = m[:, t]
        fresh = np.isnan(state) & ~np.isnan(x)
        state = np.where(fresh, x, state)
        update = ~fresh & ~np.isnan(x)
        state = np.where(update, alpha * x + (1 - alpha) * state, state)
        out[:, t] = state
    return out


def holiday_counts(weeks: pd.DatetimeIndex) -> np.ndarray:
    """Number of US federal holidays in each Monday-start week."""
    holidays = USFederalHolidayCalendar().holidays(weeks.min(), weeks.max() + pd.Timedelta(days=6))
    starts = holidays - pd.to_timedelta(holidays.weekday, unit="D")
    counts = pd.Series(1, index=starts).groupby(level=0).sum()
    return counts.reindex(weeks, fill_value=0).to_numpy(dtype=np.float64)


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
    outer: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the panel with feature columns added, and the list of feature names.

    neighbors: row-normalised (cells x cells) matrix for the inner ring, in panel cell order.
    outer: row-normalised matrix for the outer annulus (used when cfg.outer_ring > 0).
    cell_info: indexed by cell, with at least `cell_lat` and `cell_lon`.
    """
    cells, weeks = check_panel(panel)
    n_cells, n_weeks = len(cells), len(weeks)
    if neighbors.shape != (n_cells, n_cells):
        raise ValueError("neighbor matrix does not match the panel's cells")
    if cfg.outer_ring and (outer is None or outer.shape != (n_cells, n_cells)):
        raise ValueError("outer_ring is enabled but no matching outer neighbour matrix was given")
    h = cfg.horizon
    new: dict[str, np.ndarray] = {}

    def spatial(matrix: np.ndarray, values: np.ndarray, warmup: int) -> np.ndarray:
        out = matrix @ np.nan_to_num(values, nan=0.0)
        out[:, :warmup] = np.nan  # not enough history yet
        return out

    for group in groups:
        y = panel[count_column(group)].to_numpy(dtype=np.float64).reshape(n_cells, n_weeks)
        base = _shift(y, h)  # the most recent week the forecaster is allowed to see
        mean = {w: _rolling_mean(base, w) for w in {*cfg.rolling_windows, 4, 13, 52}}

        for k in cfg.lags:
            new[f"{group}_lag{k}"] = _shift(y, k)
        for w in cfg.rolling_windows:
            new[f"{group}_mean{w}"] = mean[w]
        new[f"{group}_expanding_mean"] = _expanding_mean(base)
        for hl in cfg.ewm_halflives:
            new[f"{group}_ewm{hl:g}"] = _ewm(base, hl)
        if cfg.trend_ratios:
            for w in (4, 13):
                new[f"{group}_trend{w}"] = (mean[w] + TREND_SMOOTHING) / (mean[52] + TREND_SMOOTHING)

        new[f"{group}_nbr_lag{h}"] = spatial(neighbors, base, h)
        new[f"{group}_nbr_mean4"] = spatial(neighbors, mean[4], h + 3)
        if cfg.outer_ring:
            new[f"{group}_nbr2_mean4"] = spatial(outer, mean[4], h + 3)
            new[f"{group}_nbr2_mean13"] = spatial(outer, mean[13], h + 12)

        city = _shift(y.sum(axis=0, keepdims=True), h)
        new[f"{group}_city_lag{h}"] = np.repeat(city, n_cells, axis=0)

    features = pd.DataFrame({name: m.reshape(-1) for name, m in new.items()}, index=panel.index)

    week_of_year = panel["week"].dt.isocalendar().week.astype("float64")
    features["week_sin"] = np.sin(2 * np.pi * week_of_year / 52.1775)
    features["week_cos"] = np.cos(2 * np.pi * week_of_year / 52.1775)
    features["month"] = panel["week"].dt.month.astype("int8")
    if cfg.holidays:
        features["holidays"] = np.tile(holiday_counts(weeks), n_cells)
    info = cell_info.reindex(panel["cell"])
    for col in CELL_FEATURES:
        features[col] = info[col].to_numpy()

    out = pd.concat([panel, features], axis=1)
    return out, list(features.columns)


def feature_group(name: str, groups: list[str]) -> tuple[str | None, str]:
    """(crime group or None, feature group) for a feature name."""
    if name in CALENDAR_FEATURES:
        return None, "calendar"
    if name == "holidays":
        return None, "holidays"
    if name in CELL_FEATURES:
        return None, "location"
    for group in sorted(groups, key=len, reverse=True):
        if name.startswith(f"{group}_"):
            suffix = name[len(group) + 1 :]
            for pattern, kind in _SUFFIX_GROUPS:
                if pattern.search(suffix):
                    return group, kind
    raise ValueError(f"cannot classify feature {name!r}")


def select_features(features: list[str], target: str, groups: list[str], drop: tuple[str, ...] = ()) -> list[str]:
    """Features for one target after dropping whole feature groups.

    "cross_group" drops every feature built from the other crime groups' counts.
    """
    unknown = set(drop) - set(ABLATION_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups {sorted(unknown)}; choose from {ABLATION_GROUPS}")
    kept = []
    for name in features:
        crime_group, kind = feature_group(name, groups)
        if kind in drop:
            continue
        if "cross_group" in drop and crime_group not in (None, target):
            continue
        kept.append(name)
    return kept
