"""Charts for the data-quality story: the cleaning funnel and the timestamp shift.

Each function returns an open matplotlib Figure (so a notebook can display it) and saves
it first when `out_path` is given.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from denvercrime.viz import finish_figure
from denvercrime.viz.maps import plt

KEPT_COLOR = "#1f77b4"
REMOVED_COLOR = "#d62728"


def plot_cleaning_funnel(funnel: pd.DataFrame, out_path: Path | None = None):
    """Horizontal bars of the rows surviving each cleaning step (`profile.funnel_table`)."""
    steps = funnel.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(funnel) + 1.2))
    ax.barh(steps["step"], steps["rows"], color=KEPT_COLOR, height=0.62)
    span = steps["rows"].max() or 1
    for y, (rows, removed) in enumerate(zip(steps["rows"], steps["removed"])):
        label = f"{rows:,}" + (f"   −{removed:,}" if removed else "")
        ax.text(rows + 0.01 * span, y, label, va="center", fontsize=9)
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel("offense rows remaining")
    ax.set_title("Cleaning funnel")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return finish_figure(fig, out_path)


def plot_timestamp_violations(summary: pd.DataFrame, out_path: Path | None = None):
    """Bars of 'offenses that end before they start' per candidate shift (`order_violation_summary`)."""
    labels = ["file as supplied" if s == 0 else f"{s:+g} h correction" for s in summary["shift_hours"]]
    colors = [REMOVED_COLOR if v else KEPT_COLOR for v in summary["violations"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(labels, summary["violations"], color=colors, width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}\n({p:.1f}%)" for v, p in zip(summary["violations"], summary["violation_pct"])],
                 padding=3, fontsize=9)
    ax.set_ylim(0, max(summary["violations"].max() * 1.25, 1))
    ax.set_ylabel("offenses ending before they start")
    ax.set_title("Impossible timestamps, before and after correction")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return finish_figure(fig, out_path)
