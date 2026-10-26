"""The figure helpers return a usable Figure, and save one when asked."""

import h3
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from denvercrime.data.clean import clean_offenses
from denvercrime.data.load import apply_time_shift, order_violation_summary
from denvercrime.data.profile import funnel_table
from denvercrime.viz import explore as ex
from denvercrime.viz.quality import plot_cleaning_funnel, plot_timestamp_violations

GROUPS = ["property", "person"]
WEEKS = pd.date_range("2022-01-03", periods=160, freq="7D")
CELLS = [h3.latlng_to_cell(39.74 + 0.01 * i, -104.99, 8) for i in range(6)]


@pytest.fixture
def panel():
    """A small (cell x week) panel shaped like the real one."""
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product([CELLS, WEEKS], names=["cell", "week"])
    frame = pd.DataFrame(index=index).reset_index()
    busy = frame["cell"].map({c: i for i, c in enumerate(CELLS)})
    frame["y_property"] = rng.poisson(1 + busy)
    frame["y_person"] = rng.poisson(0.2 + 0.1 * busy)
    frame["area_km2"] = 0.74
    return frame


def test_weekly_tables(panel):
    weekly = ex.weekly_totals(panel, GROUPS)
    assert list(weekly.columns) == GROUPS
    assert weekly["property"].sum() == panel["y_property"].sum()
    assert len(weekly) == len(WEEKS)

    by_year = ex.level_by_year(weekly)
    assert set(by_year.index) == set(WEEKS.year.unique())

    season = ex.seasonality_index(weekly)
    assert len(season) == 12 and season["property"].between(0.5, 1.5).all()


def test_concentration_and_stability_tables(panel):
    concentrated = ex.concentration_table(panel, GROUPS)
    # The synthetic panel's busiest cells hold more than an even share.
    assert (concentrated["area_share_holding_50pct"] < 0.5).all()
    assert (concentrated["incident_share_in_top_5pct_area"] > 0).all()

    stability = ex.stability_table(panel, GROUPS)
    assert set(stability["target"]) == set(GROUPS)
    assert stability["top5pct_overlap"].between(0, 1).all()


@pytest.mark.parametrize("plot", [ex.plot_concentration, ex.plot_hotspot_stability, ex.plot_density_map])
def test_panel_figures_render_and_save(panel, tmp_path, plot):
    out = tmp_path / f"{plot.__name__}.png"
    fig = plot(panel, GROUPS, out)
    assert isinstance(fig, Figure) and out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("plot", [ex.plot_trend_by_year, ex.plot_seasonality])
def test_weekly_figures_render(panel, plot):
    assert isinstance(plot(ex.weekly_totals(panel, GROUPS), GROUPS), Figure)


def test_quality_figures_render_and_save(raw_df, cfg, tmp_path):
    _, report = clean_offenses(raw_df, cfg)
    funnel_png = tmp_path / "funnel.png"
    assert isinstance(plot_cleaning_funnel(funnel_table(report), funnel_png), Figure)
    assert funnel_png.exists()

    summary = order_violation_summary(apply_time_shift(raw_df, 7))
    assert isinstance(plot_timestamp_violations(summary), Figure)
