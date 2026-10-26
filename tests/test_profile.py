import numpy as np
import pandas as pd
import pytest

from denvercrime.data.clean import clean_offenses
from denvercrime.data.profile import excluded_places_table, funnel_table, missing_value_table
from tests.conftest import add_dirty_rows


def test_missing_value_table_counts_and_orders(raw_df):
    table = missing_value_table(raw_df)
    assert list(table.columns) == ["dtype", "missing", "missing_pct"]
    assert table["missing"].is_monotonic_decreasing
    assert table.loc["LAST_OCCURRENCE_DATE", "missing"] == int(raw_df["LAST_OCCURRENCE_DATE"].isna().sum())
    assert table.loc["INCIDENT_ID", "missing"] == 0
    expected_pct = round(100 * table.loc["LAST_OCCURRENCE_DATE", "missing"] / len(raw_df), 2)
    assert table.loc["LAST_OCCURRENCE_DATE", "missing_pct"] == pytest.approx(expected_pct)


def test_missing_value_table_handles_an_empty_frame():
    table = missing_value_table(pd.DataFrame({"a": pd.Series(dtype="float64")}))
    assert table.loc["a", "missing"] == 0 and table.loc["a", "missing_pct"] == 0


def test_funnel_table_reports_what_each_step_removed(raw_df, cfg):
    _, report = clean_offenses(add_dirty_rows(raw_df), cfg)
    funnel = funnel_table(report)
    assert funnel["step"].iloc[0] == "raw offenses"
    assert funnel["removed"].iloc[0] == 0  # nothing removed before the first row
    assert (funnel["removed"].iloc[1:] == -funnel["rows"].diff().iloc[1:]).all()
    assert funnel["rows"].is_monotonic_decreasing
    assert funnel["pct_of_raw"].iloc[0] == 100.0
    assert funnel["pct_of_raw"].iloc[-1] < 100.0


def test_excluded_places_table_joins_the_configured_reasons(raw_df, cfg):
    _, report = clean_offenses(add_dirty_rows(raw_df), cfg)
    table = excluded_places_table(cfg, report)
    assert set(table["kind"]) == {"neighbourhood", "address"}
    assert table["offense_rows_removed"].is_monotonic_decreasing
    airport = table[table["place"] == "dia"].iloc[0]
    assert airport["kind"] == "neighbourhood" and "Airport" in airport["reason"]
    jail = table[table["place"] == "490 W COLFAX AVE"].iloc[0]
    assert jail["offense_rows_removed"] == 1 and "Detention" in jail["reason"]
    assert (table["reason"].str.len() > 0).all()  # every configured place explains itself
