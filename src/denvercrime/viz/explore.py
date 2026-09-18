"""Descriptive analysis of the cleaned panel: trends, seasonality, concentration, stability."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from denvercrime.features.panel import count_column
from denvercrime.viz.maps import draw_hexagons, plt

log = logging.getLogger(__name__)

COLORS = {"property": "#1f77b4", "person": "#d62728", "society": "#7f7f7f", "other": "#bcbd22"}
TOP_SHARE = 0.05


def concentration(counts: pd.Series, area: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of area vs cumulative share of incidents, busiest cells first."""
    order = counts.sort_values(ascending=False).index
    x = np.concatenate([[0], np.cumsum(area[order]) / area.sum()])
    y = np.concatenate([[0], np.cumsum(counts[order]) / counts.sum()])
    return x, y


def area_for_share(x: np.ndarray, y: np.ndarray, share: float) -> float:
    return float(np.interp(share, y, x))


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


def explore(frame: pd.DataFrame, groups: list[str], targets: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    weekly = frame.groupby("week")[[count_column(g) for g in groups]].sum()
    weekly.columns = groups

    # 1. Level by year (mean incidents per week, so the partial current year is comparable)
    by_year = weekly.groupby(weekly.index.year).mean()
    stats["mean_weekly_incidents_by_year"] = by_year.round(1).to_dict()
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(groups)
    for i, g in enumerate(groups):
        ax.bar(by_year.index + (i - (len(groups) - 1) / 2) * width, by_year[g], width, label=g, color=COLORS.get(g))
    ax.set_ylabel("incidents per week (city-wide)")
    ax.set_title("Weekly incident level by year")
    ax.legend(frameon=False, ncol=len(groups))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "trend_by_year.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. Seasonality: month-of-year index relative to each year's mean
    rel = weekly / weekly.groupby(weekly.index.year).transform("mean")
    season = rel.groupby(rel.index.month).mean()
    stats["seasonality_index_by_month"] = season[targets].round(3).to_dict()
    fig, ax = plt.subplots(figsize=(8, 4))
    for g in targets:
        ax.plot(season.index, season[g], marker="o", label=g, color=COLORS.get(g))
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("weekly incidents / yearly average")
    ax.set_title("Seasonality")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "seasonality.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3. Concentration: how much of the city holds how much of the crime
    area = frame.groupby("cell")["area_km2"].first()
    fig, ax = plt.subplots(figsize=(6, 5))
    stats["concentration"] = {}
    for g in targets:
        x, y = concentration(frame.groupby("cell")[count_column(g)].sum(), area)
        half, top5 = area_for_share(x, y, 0.5), float(np.interp(TOP_SHARE, x, y))
        stats["concentration"][g] = {"area_share_holding_50pct": half, "incident_share_in_top_5pct_area": top5}
        ax.plot(x * 100, y * 100, label=f"{g}: 50% of incidents in {half:.0%} of area", color=COLORS.get(g))
    ax.plot([0, 100], [0, 100], color="grey", linestyle="--", label="evenly spread")
    ax.set_xlabel("% of city area (busiest hexagons first)")
    ax.set_ylabel("% of incidents")
    ax.set_title("Crime concentration")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "concentration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 4. Stability: do hotspots stay put from one year to the next?
    stats["stability"] = {g: hotspot_stability(frame, g).round(3).to_dict("records") for g in targets}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for g in targets:
        st = hotspot_stability(frame, g)
        axes[0].plot(st["years"], st["spearman"], marker="o", label=g, color=COLORS.get(g))
        axes[1].plot(st["years"], st["top5pct_overlap"], marker="o", label=g, color=COLORS.get(g))
    axes[0].set_title("Rank correlation of hexagon counts, year to year")
    axes[1].set_title("Share of top-5% hexagons still top-5% next year")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "hotspot_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 5. Where: mean weekly incidents per km², per target
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5.2))
    for ax, g in zip(np.atleast_1d(axes), targets):
        density = frame.groupby("cell")[count_column(g)].mean() / area
        draw_hexagons(ax, density.index.to_series(), density.values, vmax=float(density.quantile(0.99)),
                      label="incidents / km² / week")
        ax.set_title(f"{g} crime density")
    fig.tight_layout()
    fig.savefig(out_dir / "density_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("Exploration figures and stats written to %s", out_dir)
    return stats
