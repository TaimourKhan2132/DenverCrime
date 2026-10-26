import pandas as pd
import pytest

from denvercrime.data.load import (
    apply_time_shift,
    detect_time_shift,
    load_raw,
    order_violation_rate,
    order_violation_summary,
    order_violations,
)
from denvercrime.schema import HUB_CSV_DATE_FORMAT


def _to_hub_csv(df: pd.DataFrame, path, shift_hours: float = 0.0) -> None:
    """Write `df` the way the Hub exports it, optionally with the +7 h FIRST/REPORTED shift."""
    out = df.copy()
    for col in ("FIRST_OCCURRENCE_DATE", "REPORTED_DATE"):
        out[col] = out[col] + pd.Timedelta(hours=shift_hours)
    for col in ("FIRST_OCCURRENCE_DATE", "LAST_OCCURRENCE_DATE", "REPORTED_DATE"):
        out[col] = out[col].dt.strftime(HUB_CSV_DATE_FORMAT)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def test_parquet_round_trip(raw_df, tmp_path):
    path = tmp_path / "crime.parquet"
    raw_df.to_parquet(path, index=False)
    df = load_raw(path)
    assert len(df) == len(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(df["FIRST_OCCURRENCE_DATE"])
    assert df["IS_CRIME"].dtype == "Int64"


def test_hub_csv_shift_is_detected_and_undone(raw_df, tmp_path):
    path = tmp_path / "crime.csv"
    _to_hub_csv(raw_df, path, shift_hours=7)
    df = load_raw(path, "auto")
    assert df.attrs["time_shift_hours"] == -7
    pd.testing.assert_series_equal(
        df["FIRST_OCCURRENCE_DATE"].reset_index(drop=True),
        raw_df["FIRST_OCCURRENCE_DATE"].dt.floor("s").reset_index(drop=True),
        check_names=False, check_dtype=False,
    )


def test_unshifted_csv_is_left_alone(raw_df, tmp_path):
    path = tmp_path / "crime.csv"
    _to_hub_csv(raw_df, path, shift_hours=0)
    assert load_raw(path, "auto").attrs["time_shift_hours"] == 0


def test_explicit_shift_overrides_detection(raw_df, tmp_path):
    path = tmp_path / "crime.csv"
    _to_hub_csv(raw_df, path, shift_hours=7)
    assert load_raw(path, -7).attrs["time_shift_hours"] == -7
    assert load_raw(path, 0).attrs["time_shift_hours"] == 0


def test_violation_rate_explains_the_shift(raw_df):
    shifted = raw_df.assign(FIRST_OCCURRENCE_DATE=raw_df["FIRST_OCCURRENCE_DATE"] + pd.Timedelta(hours=7))
    assert order_violation_rate(raw_df) == 0
    assert order_violation_rate(shifted) > 0.3
    assert detect_time_shift(shifted) == -7


def test_order_violation_summary_tells_the_shift_story(raw_df):
    """The +7 h Hub CSV shift makes offenses end before they start; the correction undoes it."""
    shifted = apply_time_shift(raw_df, 7)
    summary = order_violation_summary(shifted)
    as_supplied = summary.loc[summary["shift_hours"] == 0].iloc[0]
    corrected = summary.loc[summary["shift_hours"] == -7].iloc[0]

    assert as_supplied["violations"] > 0 and corrected["violations"] == 0
    assert as_supplied["violations"] == order_violations(shifted)
    assert order_violations(raw_df) == 0  # the API snapshot is already consistent
    assert (summary["comparable_rows"] == raw_df["LAST_OCCURRENCE_DATE"].notna().sum()).all()
    assert as_supplied["violation_pct"] == pytest.approx(
        100 * as_supplied["violations"] / as_supplied["comparable_rows"], abs=0.01
    )


def test_order_violations_ignores_rows_without_an_end_time():
    first = pd.Timestamp("2024-05-01 10:00")
    df = pd.DataFrame(
        {
            "FIRST_OCCURRENCE_DATE": [first, first, first],
            "LAST_OCCURRENCE_DATE": [pd.NaT, first - pd.Timedelta(hours=1), first + pd.Timedelta(hours=1)],
        }
    )
    assert order_violations(df) == 1


def test_missing_required_column(raw_df, tmp_path):
    path = tmp_path / "crime.parquet"
    raw_df.drop(columns="GEO_LAT").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="GEO_LAT"):
        load_raw(path)
