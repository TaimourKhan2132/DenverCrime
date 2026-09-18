"""Folium hexagon maps and matplotlib diagnostics for a backtest run."""

from __future__ import annotations

from pathlib import Path

import branca.colormap as cm
import folium
import matplotlib
import numpy as np
import pandas as pd

from denvercrime.evaluation.metrics import hotspot_metrics
from denvercrime.features.spatial import cell_polygon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DENVER_CENTER = (39.7392, -104.9903)


def hex_map(week_frame: pd.DataFrame, layers: dict[str, str], title: str, out_path: Path) -> Path:
    """One toggleable choropleth layer per entry of `layers` ({layer name: column})."""
    vmax = float(max(week_frame[col].max() for col in layers.values())) or 1.0
    colormap = cm.linear.YlOrRd_09.scale(0, vmax)
    colormap.caption = title

    fmap = folium.Map(location=DENVER_CENTER, zoom_start=11, tiles="OpenStreetMap")
    for i, (name, col) in enumerate(layers.items()):
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [cell_polygon(row.cell)]},
                "properties": {"cell": row.cell, **{c: round(float(getattr(row, c)), 3) for c in layers.values()}},
            }
            for row in week_frame.itertuples(index=False)
        ]
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            name=name,
            show=i == 0,
            style_function=lambda f, col=col: {
                "fillColor": colormap(f["properties"][col]),
                "color": "#555555",
                "weight": 0.3,
                "fillOpacity": 0.65,
            },
            tooltip=folium.GeoJsonTooltip(fields=["cell", *layers.values()]),
        ).add_to(fmap)
    colormap.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def draw_hexagons(ax, cells: pd.Series, values: pd.Series, cmap: str = "YlOrRd", vmax: float | None = None,
                  label: str = "") -> None:
    """Filled H3 hexagons on a matplotlib axis (lon/lat, aspect corrected for Denver's latitude)."""
    from matplotlib.collections import PolyCollection

    polys = [[(lon, lat) for lon, lat in cell_polygon(c)] for c in cells]
    coll = PolyCollection(polys, array=np.asarray(values, dtype=float), cmap=cmap,
                          edgecolors="white", linewidths=0.15)
    coll.set_clim(0, vmax if vmax is not None else float(np.nanmax(values)) or 1.0)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(DENVER_CENTER[0])))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    plt.colorbar(coll, ax=ax, shrink=0.7, label=label)


def plot_forecast_map(week_frame: pd.DataFrame, pred_col: str, y_col: str, title: str, out_path: Path) -> Path:
    """Side-by-side static maps: forecast vs actual for one week."""
    vmax = float(max(week_frame[pred_col].max(), week_frame[y_col].max())) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    draw_hexagons(axes[0], week_frame["cell"], week_frame[pred_col], vmax=vmax, label="expected incidents")
    axes[0].set_title("Forecast (LightGBM)")
    draw_hexagons(axes[1], week_frame["cell"], week_frame[y_col], vmax=vmax, label="incidents")
    axes[1].set_title("What happened")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_evaluation_timeline(split: dict, folds: list[dict], out_path: Path) -> Path:
    """Train / validation / rolling test windows, with the data each window was retrained on.

    split: `Split.describe()` output; folds: the per-window records written by the backtest.
    """
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    week = pd.Timedelta(days=7)
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "history": "#c9d6e3", "test": "#2ca02c"}
    start = pd.Timestamp(split["train"]["first_week"])
    train_end = pd.Timestamp(split["train"]["last_week"]) + week
    val_start, val_end = pd.Timestamp(split["val"]["first_week"]), pd.Timestamp(split["val"]["last_week"]) + week

    def bar(y, a, b, color, text=None):
        left, width = mdates.date2num(a), mdates.date2num(b) - mdates.date2num(a)
        ax.barh(y, width, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.8)
        if text:
            ax.text(left + width / 2, y, text, ha="center", va="center", fontsize=9.5, color="white", weight="bold")

    n = len(folds)
    fig, ax = plt.subplots(figsize=(11, 1.4 + 0.5 * (n + 1)))
    bar(n, start, train_end, colors["train"], f"Train  ({split['train']['weeks']} weeks)")
    bar(n, val_start, val_end, colors["val"], f"Validation ({split['val']['weeks']} wk)")
    labels = ["Model selection"]
    for i, fold in enumerate(folds):
        y = n - 1 - i
        w_start, w_end = pd.Timestamp(fold["first_week"]), pd.Timestamp(fold["last_week"]) + week
        bar(y, start, pd.Timestamp(fold["trained_through"]) + week, colors["history"])
        bar(y, w_start, w_end, colors["test"])
        labels.append(f"Test window {fold['fold']}: {w_start:%b %Y} – {pd.Timestamp(fold['last_week']):%b %Y}")
    for boundary in (val_start, pd.Timestamp(folds[0]["first_week"])):
        ax.axvline(mdates.date2num(boundary), color="grey", linestyle="--", linewidth=0.8)

    ax.set_yticks(range(n, -1, -1), labels)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-0.6, n + 0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Patch(color=colors["train"], label="training data"),
            Patch(color=colors["val"], label="validation year: early stopping, ablation, tuning"),
            Patch(color=colors["history"], label="data the model is retrained on"),
            Patch(color=colors["test"], label="weeks it then predicts (never used for any choice)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=9,
    )
    ax.set_title("Evaluation protocol: choose on the validation year, then retrain before each test window")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_weekly_totals(preds: pd.DataFrame, y_col: str, pred_cols: list[str], out_path: Path) -> Path:
    weekly = preds.groupby("week")[[y_col, *pred_cols]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(weekly.index, weekly[y_col], color="black", linewidth=2, label="actual")
    for col in pred_cols:
        ax.plot(weekly.index, weekly[col], linewidth=1.2, label=col)
    ax.set_ylabel("incidents per week (all cells)")
    ax.set_title("City-wide weekly incidents: actual vs forecast (test period)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hotspot_curve(preds: pd.DataFrame, y_col: str, pred_cols: list[str], out_path: Path,
                       shares: np.ndarray | None = None) -> Path:
    shares = shares if shares is not None else np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for col in pred_cols:
        hits = [hotspot_metrics(preds, col, y_col, s)["hit_rate"] for s in shares]
        ax.plot(shares * 100, np.array(hits) * 100, marker="o", label=col)
    ax.plot(shares * 100, shares * 100, color="grey", linestyle="--", label="random")
    ax.set_xlabel("% of city area flagged")
    ax.set_ylabel("% of incidents captured (mean over weeks)")
    ax.set_title("Hotspot capture curve (test period)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
