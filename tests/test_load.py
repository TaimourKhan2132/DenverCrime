import pandas as pd
import pytest

from denvercrime.data.load import detect_time_shift, load_raw, order_violation_rate
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


def test_missing_required_column(raw_df, tmp_path):
    path = tmp_path / "crime.parquet"
    raw_df.drop(columns="GEO_LAT").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="GEO_LAT"):
        load_raw(path)
