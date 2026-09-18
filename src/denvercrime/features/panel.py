"""Aggregate incidents into a complete (cell x week) panel of counts."""

from __future__ import annotations

import pandas as pd

from denvercrime.features.spatial import to_cells


def week_start(ts: pd.Series) -> pd.Series:
    """Monday 00:00 of the week containing each timestamp."""
    day = ts.dt.normalize()
    return day - pd.to_timedelta(day.dt.weekday, unit="D")


def count_column(group: str) -> str:
    return f"y_{group}"


def assign_cells(incidents: pd.DataFrame, resolution: int) -> pd.DataFrame:
    return incidents.assign(
        cell=to_cells(incidents["lat"], incidents["lon"], resolution),
        week=week_start(incidents["occurred_at"]),
    )


def select_cells(incidents: pd.DataFrame, train_end: str, min_incidents: int) -> list[str]:
    """Cells with at least `min_incidents` incidents (any group) up to `train_end`.

    Using only the training period keeps the cell universe free of future information.
    """
    train = incidents[incidents["week"] <= pd.Timestamp(train_end)]
    counts = train.groupby("cell").size()
    return sorted(counts[counts >= min_incidents].index)


def complete_weeks(first: pd.Timestamp, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    """Mondays from the week of `first` through the last week that ends on or before `cutoff`.

    Data are kept only for timestamps < cutoff, so the week containing the cutoff is partial.
    """
    start = first.normalize() - pd.Timedelta(days=first.weekday())
    last = cutoff.normalize() - pd.Timedelta(days=cutoff.weekday() + 7)
    return pd.date_range(start, last, freq="7D")


def build_panel(
    incidents: pd.DataFrame,
    groups: list[str],
    cells: list[str],
    weeks: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Rows = every (cell, week) pair, sorted by cell then week; one count column per group.

    Pairs with no incidents are explicit zeros; forecasting models need the zeros.
    """
    data = incidents[incidents["cell"].isin(cells) & incidents["week"].isin(weeks)]
    counts = (
        data.groupby(["cell", "week", "group"]).size()
        .unstack("group", fill_value=0)
        .reindex(columns=groups, fill_value=0)
    )
    counts.columns = [count_column(g) for g in groups]
    full = pd.MultiIndex.from_product([cells, weeks], names=["cell", "week"])
    panel = counts.reindex(full, fill_value=0).reset_index()
    panel["cell"] = panel["cell"].astype(str)
    return panel.astype({count_column(g): "int32" for g in groups})
