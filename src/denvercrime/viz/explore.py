"""Descriptive analysis of the cleaned panel: trends, seasonality, concentration, stability.

Each `plot_*` function returns an open matplotlib Figure so a notebook can display it, and
saves it first when `out_path` is given. `explore()` runs all of them and writes the stats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from denvercrime.features.panel import count_column
from denvercrime.viz.maps import draw_hexagons, plt
from denvercrime.viz import finish_figure

log = logging.getLogger(__name__)

COLORS = {"property": "#1f77b4", "person": "#d62728", "society": "#7f7f7f", "other": "#bcbd22"}
TOP_SHARE = 0.05
MONTH_INITIALS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


# --- tables ------------------------------------------------------------------------------

def weekly_totals(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """City-wide incidents per week, one column per crime group."""
    weekly = frame.groupby("week")[[count_column(g) for g in groups]].sum()
    weekly.columns = groups
    return weekly


def level_by_year(weekly: pd.DataFrame) -> pd.DataFrame:
    """Mean incidents per week within each year, so a partial current year stays comparable."""
    return weekly.groupby(weekly.index.year).mean()


def seasonality_index(weekly: pd.DataFrame) -> pd.DataFrame:
    """Month-of-year index: a month's weekly average divided by its own year's average."""
    relative = weekly / weekly.groupby(weekly.index.year).transform("mean")
    return relative.groupby(relative.index.month).mean()


def cell_area(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("cell")["area_km2"].first()


def concentration(counts: pd.Series, area: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of area vs cumulative share of incidents, busiest cells first."""
    order = counts.sort_values(ascending=False).index
    x = np.concatenate([[0], np.cumsum(area[order]) / area.sum()])
    y = np.concatenate([[0], np.cumsum(counts[order]) / counts.sum()])
    return x, y


def area_for_share(x: np.ndarray, y: np.ndarray, share: float) -> float:
    return float(np.interp(share, y, x))


def concentration_table(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """How little of the city holds half the incidents, and how much lands in the busiest 5%."""
    area = cell_area(frame)
    rows = []
    for target in targets:
        x, y = concentration(frame.groupby("cell")[count_column(target)].sum(), area)
        rows.append(
            {
                "target": target,
                "area_share_holding_50pct": area_for_share(x, y, 0.5),
                "incident_share_in_top_5pct_area": float(np.interp(TOP_SHARE, x, y)),
            }
        )
    return pd.DataFrame(rows).set_index("target")


def hotspot_stability(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Year-over-year Spearman correlation of cell counts and overlap of the top 5% of cells."""
    label = count_column(target)
    yearly = frame.assign(year=frame["week"].dt.year).groupby(["year", "cell"])[label].sum().unstack("year")
    full_years = [y for y in yearly.columns if frame.loc[frame["week"].dt.year == y, "week"].nunique() >= 50]
    rows = []
    for a, b in zip(full_years, full_years[1:]):
        k = max(1, int(round(TOP_SHARE * len(yearly))))
        top_a, top_b = set(yearly[a].nlargest(k).index), set(yearly[b].nlargest(k).index)
        rows.append({"years": f"{a}->{b}", "spearman": yearly[a].corr(yearly[b], method="spearman"),
                     "top5pct_overlap": len(top_a & top_b) / k})
    return pd.DataFrame(rows)


def stability_table(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """`hotspot_stability` for several targets, stacked."""
    return pd.concat([hotspot_stability(frame, t).assign(target=t) for t in targets], ignore_index=True)


# --- figures -----------------------------------------------------------------------------

def plot_trend_by_year(weekly: pd.DataFrame, groups: list[str], out_path: Path | None = None):
    by_year = level_by_year(weekly)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(groups)
    for i, group in enumerate(groups):
        ax.bar(by_year.index + (i - (len(groups) - 1) / 2) * width, by_year[group], width,
               label=group, color=COLORS.get(group))
    ax.set_ylabel("incidents per week (city-wide)")
    ax.set_title("Weekly incident level by year")
    ax.legend(frameon=False, ncol=len(groups))
    ax.grid(axis="y", alpha=0.3)
    return finish_figure(fig, out_path)


def plot_seasonality(weekly: pd.DataFrame, targets: list[str], out_path: Path | None = None):
    season = seasonality_index(weekly)
    fig, ax = plt.subplots(figsize=(8, 4))
    for target in targets:
        ax.plot(season.index, season[target], marker="o", label=target, color=COLORS.get(target))
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, 13), MONTH_INITIALS)
    ax.set_ylabel("weekly incidents / yearly average")
    ax.set_title("Seasonality")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    return finish_figure(fig, out_path)


def plot_concentration(frame: pd.DataFrame, targets: list[str], out_path: Path | None = None):
    area = cell_area(frame)
    fig, ax = plt.subplots(figsize=(6, 5))
    for target in targets:
        x, y = concentration(frame.groupby("cell")[count_column(target)].sum(), area)
        half = area_for_share(x, y, 0.5)
        ax.plot(x * 100, y * 100, label=f"{target}: 50% of incidents in {half:.0%} of area",
                color=COLORS.get(target))
    ax.plot([0, 100], [0, 100], color="grey", linestyle="--", label="evenly spread")
    ax.set_xlabel("% of city area (busiest hexagons first)")
    ax.set_ylabel("% of incidents")
    ax.set_title("Crime concentration")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    return finish_figure(fig, out_path)


def plot_hotspot_stability(frame: pd.DataFrame, targets: list[str], out_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for target in targets:
        stability = hotspot_stability(frame, target)
        axes[0].plot(stability["years"], stability["spearman"], marker="o", label=target, color=COLORS.get(target))
        axes[1].plot(stability["years"], stability["top5pct_overlap"], marker="o", label=target, color=COLORS.get(target))
    axes[0].set_title("Rank correlation of hexagon counts, year to year")
    axes[1].set_title("Share of top-5% hexagons still top-5% next year")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    return finish_figure(fig, out_path)


def plot_density_map(frame: pd.DataFrame, targets: list[str], out_path: Path | None = None):
    area = cell_area(frame)
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5.2))
    for ax, target in zip(np.atleast_1d(axes), targets):
        density = frame.groupby("cell")[count_column(target)].mean() / area
        draw_hexagons(ax, density.index.to_series(), density.values,
                      vmax=float(density.quantile(0.99)), label="incidents / km² / week")
        ax.set_title(f"{target} crime density")
    return finish_figure(fig, out_path)


# --- orchestration -----------------------------------------------------------------------

def explore(frame: pd.DataFrame, groups: list[str], targets: list[str], out_dir: Path) -> dict:
    """Write every exploration figure to `out_dir` and return the statistics behind them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly = weekly_totals(frame, groups)
    concentrated = concentration_table(frame, targets)

    stats = {
        "mean_weekly_incidents_by_year": level_by_year(weekly).round(1).to_dict(),
        "seasonality_index_by_month": seasonality_index(weekly)[targets].round(3).to_dict(),
        "concentration": {t: concentrated.loc[t].to_dict() for t in targets},
        "stability": {t: hotspot_stability(frame, t).round(3).to_dict("records") for t in targets},
    }

    for fig in (
        plot_trend_by_year(weekly, groups, out_dir / "trend_by_year.png"),
        plot_seasonality(weekly, targets, out_dir / "seasonality.png"),
        plot_concentration(frame, targets, out_dir / "concentration.png"),
        plot_hotspot_stability(frame, targets, out_dir / "hotspot_stability.png"),
        plot_density_map(frame, targets, out_dir / "density_map.png"),
    ):
        plt.close(fig)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("Exploration figures and stats written to %s", out_dir)
    return stats
