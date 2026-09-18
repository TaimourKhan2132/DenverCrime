import pandas as pd

from denvercrime.data.clean import clean_offenses, to_incidents
from tests.conftest import _row, add_dirty_rows, make_config


def test_dirty_rows_are_dropped_and_reported(raw_df, cfg):
    offenses, report = clean_offenses(add_dirty_rows(raw_df), cfg)
    assert not offenses["INCIDENT_ID"].str.startswith("BAD").any()
    assert report.unmapped_categories == {"mystery-category": 1}
    steps = dict(report.steps)
    assert steps["raw offenses"] == len(raw_df) + 7


def test_airport_and_institutions_are_excluded_and_counted(raw_df, cfg):
    _, report = clean_offenses(add_dirty_rows(raw_df), cfg)
    assert report.excluded["neighborhood:dia"] == 1
    assert report.excluded["address:490 W COLFAX AVE"] == 1  # matched despite case and spacing
    assert report.excluded["address:1331 N CHEROKEE ST"] == 0


def test_exclusions_can_be_switched_off(raw_df, tmp_path):
    cfg = make_config(tmp_path, cleaning__exclude_neighborhoods=[], cleaning__exclude_addresses=[])
    offenses, report = clean_offenses(add_dirty_rows(raw_df), cfg)
    assert {"BAD6", "BAD7"} <= set(offenses["INCIDENT_ID"])
    assert report.excluded == {}


def test_long_occurrence_windows_are_counted(raw_df, cfg):
    t = pd.Timestamp("2022-06-01 10:00")
    long_row = _row("LONG", 0, "burglary", t, t + pd.Timedelta(days=20), t + pd.Timedelta(days=21), 39.74, -104.99)
    _, report = clean_offenses(pd.concat([raw_df, pd.DataFrame([long_row])], ignore_index=True), cfg)
    assert report.long_window_offenses >= 1


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
