import pandas as pd

from denvercrime.data.clean import clean_offenses, to_incidents
from tests.conftest import _row, add_dirty_rows


def test_dirty_rows_are_dropped_and_reported(raw_df, cfg):
    offenses, report = clean_offenses(add_dirty_rows(raw_df), cfg)
    assert not offenses["INCIDENT_ID"].str.startswith("BAD").any()
    assert report.unmapped_categories == {"mystery-category": 1}
    steps = dict(report.steps)
    assert steps["raw offenses"] == len(raw_df) + 5


def test_recent_rows_after_cutoff_are_dropped(raw_df, cfg):
    offenses, report = clean_offenses(raw_df, cfg)
    cutoff = pd.Timestamp(report.cutoff)
    assert offenses["FIRST_OCCURRENCE_DATE"].max() < cutoff
    assert cutoff == raw_df["REPORTED_DATE"].max().normalize() - pd.Timedelta(days=30)


def test_incidents_count_once_per_group(cfg):
    t = pd.Timestamp("2022-06-01 10:00")
    rows = [
        _row("A", 0, "larceny", t, pd.NaT, t, 39.74, -104.99),
        _row("A", 1, "burglary", t + pd.Timedelta(hours=1), pd.NaT, t, 39.75, -104.98),  # same group
        _row("B", 0, "larceny", t, pd.NaT, t, 39.74, -104.99),
        _row("B", 1, "robbery", t, pd.NaT, t, 39.74, -104.99),  # second group
    ]
    offenses, _ = clean_offenses(pd.DataFrame(rows).assign(REPORTED_DATE=pd.Timestamp("2023-01-01")), cfg)
    incidents = to_incidents(offenses)
    a = incidents[incidents["INCIDENT_ID"] == "A"]
    assert len(a) == 1 and a["n_offenses"].item() == 2
    assert a["lat"].item() == 39.74  # taken from the earliest offense
    assert sorted(incidents.loc[incidents["INCIDENT_ID"] == "B", "group"]) == ["person", "property"]
