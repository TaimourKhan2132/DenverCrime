import pandas as pd

from denvercrime.data.clean import clean_offenses, to_incidents
from denvercrime.features.panel import assign_cells, build_panel, complete_weeks, select_cells, week_start


def _incidents(raw_df, cfg):
    offenses, report = clean_offenses(raw_df, cfg)
    return assign_cells(to_incidents(offenses), cfg.panel.h3_resolution), report


def test_week_start_is_monday():
    ts = pd.Series(pd.to_datetime(["2024-05-19 23:59", "2024-05-20 00:00", "2024-05-26 12:00"]))
    assert list(week_start(ts)) == list(pd.to_datetime(["2024-05-13", "2024-05-20", "2024-05-20"]))


def test_complete_weeks_stop_before_the_cutoff_week():
    weeks = complete_weeks(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-02-14"))  # a Wednesday
    assert weeks[0] == pd.Timestamp("2024-01-01")
    assert weeks[-1] + pd.Timedelta(days=7) <= pd.Timestamp("2024-02-14")
    assert weeks[-1] == pd.Timestamp("2024-02-05")
    monday = complete_weeks(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-12"))
    assert monday[-1] == pd.Timestamp("2024-02-05")


def test_panel_is_complete_and_counts_match(raw_df, cfg):
    incidents, report = _incidents(raw_df, cfg)
    cells = select_cells(incidents, cfg.split.train_end, cfg.panel.min_train_incidents)
    weeks = complete_weeks(pd.Timestamp(cfg.data.start_date), pd.Timestamp(report.cutoff))
    groups = list(cfg.groups)
    panel = build_panel(incidents, groups, cells, weeks)

    assert len(panel) == len(cells) * len(weeks)
    assert not panel.duplicated(["cell", "week"]).any()
    assert (panel.filter(like="y_") >= 0).all().all()
    kept = incidents[incidents["cell"].isin(cells) & incidents["week"].isin(weeks)]
    for g in groups:
        assert panel[f"y_{g}"].sum() == (kept["group"] == g).sum()


def test_cell_selection_ignores_the_future(raw_df, cfg):
    incidents, _ = _incidents(raw_df, cfg)
    future_only = incidents.iloc[[0]].assign(cell="88268cdb97fffff", week=pd.Timestamp("2023-08-07"))
    cells = select_cells(pd.concat([incidents, *[future_only] * 20]), cfg.split.train_end, 1)
    assert "88268cdb97fffff" not in cells
