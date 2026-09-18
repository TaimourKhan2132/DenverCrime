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
    fig.savefig(out_path, dpi=160)
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
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
