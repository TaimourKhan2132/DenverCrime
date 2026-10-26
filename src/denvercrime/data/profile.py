"""Small tables describing a snapshot and what cleaning did to it.

These summarise work done elsewhere (`denvercrime.data.clean`) for reports and for the
walkthrough notebook; they contain no filtering logic of their own.
"""

from __future__ import annotations

import pandas as pd

from denvercrime.config import Config
from denvercrime.data.clean import CleaningReport

NEIGHBORHOOD_PREFIX = "neighborhood"


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, how many values are missing, and the share, worst first."""
    missing = df.isna().sum()
    share = 100 * missing / len(df) if len(df) else missing * 0.0
    out = pd.DataFrame(
        {"dtype": df.dtypes.astype(str), "missing": missing.astype(int), "missing_pct": share.round(2)}
    )
    out.index.name = "column"
    return out.sort_values(["missing", "column"], ascending=[False, True])


def funnel_table(report: CleaningReport) -> pd.DataFrame:
    """Rows surviving each cleaning step, with how many that step removed."""
    steps = pd.DataFrame(report.steps, columns=["step", "rows"])
    if steps.empty:
        return steps.assign(removed=[], pct_of_raw=[])
    steps["removed"] = (steps["rows"].shift(1) - steps["rows"]).fillna(0).astype(int)
    steps["pct_of_raw"] = (100 * steps["rows"] / steps["rows"].iloc[0]).round(1)
    return steps


def excluded_places_table(cfg: Config, report: CleaningReport) -> pd.DataFrame:
    """Every configured place exclusion: what it is, why, and how many offense rows it removed."""
    rows = []
    for key, removed in report.excluded.items():
        kind, _, name = key.partition(":")
        is_hood = kind == NEIGHBORHOOD_PREFIX
        rows.append(
            {
                "place": name,
                "kind": "neighbourhood" if is_hood else "address",
                "offense_rows_removed": int(removed),
                "reason": cfg.cleaning.exclude_reasons.get(key if is_hood else name, ""),
            }
        )
    out = pd.DataFrame(rows, columns=["place", "kind", "offense_rows_removed", "reason"])
    return out.sort_values("offense_rows_removed", ascending=False, ignore_index=True)
