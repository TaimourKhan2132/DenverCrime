"""Read a raw snapshot (API Parquet or Hub CSV) into a frame with a consistent schema."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from denvercrime.schema import (
    DATE_COLUMNS,
    FLOAT_COLUMNS,
    HUB_CSV_DATE_FORMAT,
    INT_COLUMNS,
    REQUIRED_COLUMNS,
    STRING_COLUMNS,
)

log = logging.getLogger(__name__)

# Shifts (hours) tried when auto-detecting the Hub CSV timestamp offset.
SHIFT_CANDIDATES = (0.0, -6.0, -7.0)
SHIFTED_COLUMNS = ("FIRST_OCCURRENCE_DATE", "REPORTED_DATE")


def find_latest_raw(raw_dir: Path) -> Path:
    """Newest Parquet snapshot in `raw_dir`, falling back to the newest CSV."""
    for pattern in ("*.parquet", "*.csv"):
        files = sorted(raw_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if files:
            return files[-1]
    raise FileNotFoundError(
        f"No raw data in {raw_dir}. Run `python -m denvercrime download` or place the Hub CSV there."
    )


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in STRING_COLUMNS:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    for col in FLOAT_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in DATE_COLUMNS:
        if col in df and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def order_violation_rate(df: pd.DataFrame, shift_hours: float = 0.0) -> float:
    """Share of rows with a LAST_OCCURRENCE_DATE where LAST < FIRST after shifting FIRST."""
    has_last = df["LAST_OCCURRENCE_DATE"].notna() & df["FIRST_OCCURRENCE_DATE"].notna()
    if not has_last.any():
        return 0.0
    first = df.loc[has_last, "FIRST_OCCURRENCE_DATE"] + pd.Timedelta(hours=shift_hours)
    return float((df.loc[has_last, "LAST_OCCURRENCE_DATE"] < first).mean())


def detect_time_shift(df: pd.DataFrame, candidates: tuple[float, ...] = SHIFT_CANDIDATES) -> float:
    """Pick the shift that makes FIRST <= LAST hold most often; 0 unless another is clearly better."""
    rates = {s: order_violation_rate(df, s) for s in candidates}
    best = min(rates, key=rates.get)
    if best != 0.0 and rates[best] < 0.5 * rates[0.0] and rates[0.0] > 0.01:
        log.info("Detected a %+.0f h timestamp shift (FIRST>LAST violations %.1f%% -> %.1f%%)",
                 best, 100 * rates[0.0], 100 * rates[best])
        return best
    return 0.0


def apply_time_shift(df: pd.DataFrame, hours: float) -> pd.DataFrame:
    if hours == 0:
        return df
    df = df.copy()
    for col in SHIFTED_COLUMNS:
        df[col] = df[col] + pd.Timedelta(hours=hours)
    return df


def read_hub_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    for col in DATE_COLUMNS:
        if col in df:
            parsed = pd.to_datetime(df[col], format=HUB_CSV_DATE_FORMAT, errors="coerce")
            if parsed.isna().sum() > df[col].isna().sum():  # a different export format
                parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed
    return df


def load_raw(path: Path, csv_time_shift_hours: str | float = "auto") -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = read_hub_csv(path)
    else:
        raise ValueError(f"Unsupported raw file type: {path.suffix}")

    validate_columns(df)
    df = normalize_dtypes(df)

    if path.suffix == ".csv":
        shift = detect_time_shift(df) if csv_time_shift_hours == "auto" else float(csv_time_shift_hours)
        df = apply_time_shift(df, shift)
        df.attrs["time_shift_hours"] = shift
    log.info("Loaded %s offense rows from %s", f"{len(df):,}", path.name)
    return df
