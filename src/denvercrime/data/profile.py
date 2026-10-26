"""Small tables describing a snapshot and what cleaning did to it.

These summarise work done elsewhere (`denvercrime.data.clean`) for reports and for the
walkthrough notebook; they contain no filtering logic of their own.
"""

from __future__ import annotations

import re

import pandas as pd

from denvercrime.config import Config
from denvercrime.data.clean import CleaningReport

NEIGHBORHOOD_PREFIX = "neighborhood"

# Plain-English names for the cleaner's step labels, for charts and reports written for
# readers outside the project. The technical label is always kept alongside.
FRIENDLY_STEPS = {
    "raw offenses": "Every offense in the file",
    "IS_CRIME = 1 and IS_TRAFFIC = 0": "Crimes only, traffic cases removed",
    "valid coordinates inside Denver bbox": "Has a usable map location in Denver",
    "excluded neighbourhoods": "Airport removed",
    "excluded institutional addresses": "Police HQ, jails and hospital removed",
    "category mapped to a group": "Crime type recognised",
}
DATE_RANGE_STEP = re.compile(r"^occurred in \[(\S+), (\S+)\)$")


def friendly_step(step: str) -> str:
    """A step label a non-technical reader can follow; the original if there is no plain name."""
    if step in FRIENDLY_STEPS:
        return FRIENDLY_STEPS[step]
    dates = DATE_RANGE_STEP.match(step)
    if dates:
        return f"Occurred between {dates.group(1)} and {dates.group(2)}"
    return step


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
    """Rows surviving each cleaning step, with a plain-English label and how many it removed."""
    steps = pd.DataFrame(report.steps, columns=["step", "rows"])
    if steps.empty:
        return steps.assign(label=[], removed=[], pct_of_raw=[])
    steps["label"] = steps["step"].map(friendly_step)
    steps["removed"] = (steps["rows"].shift(1) - steps["rows"]).fillna(0).astype(int)
    steps["pct_of_raw"] = (100 * steps["rows"] / steps["rows"].iloc[0]).round(1)
    return steps[["label", "rows", "removed", "pct_of_raw", "step"]]


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
