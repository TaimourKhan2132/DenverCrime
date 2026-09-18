"""Filter offense rows and collapse them into one row per (incident, group)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from denvercrime.config import Config

log = logging.getLogger(__name__)

UNMAPPED = "unmapped"


@dataclass
class CleaningReport:
    steps: list[tuple[str, int]] = field(default_factory=list)
    unmapped_categories: dict[str, int] = field(default_factory=dict)
    as_of: str = ""
    cutoff: str = ""

    def record(self, step: str, df: pd.DataFrame) -> None:
        self.steps.append((step, len(df)))
        log.info("  %-44s %10s rows", step, f"{len(df):,}")

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "cutoff": self.cutoff,
            "steps": [{"step": s, "rows": n} for s, n in self.steps],
            "unmapped_categories": self.unmapped_categories,
        }


def data_as_of(df: pd.DataFrame) -> pd.Timestamp:
    """The snapshot date, taken as the most recent REPORTED_DATE in the data."""
    return df["REPORTED_DATE"].max().normalize()


def clean_offenses(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply row filters and attach the forecasting group of each offense."""
    report = CleaningReport()
    as_of = data_as_of(df)
    cutoff = as_of - pd.Timedelta(days=cfg.data.drop_recent_days)
    report.as_of, report.cutoff = str(as_of.date()), str(cutoff.date())
    report.record("raw offenses", df)

    df = df[(df["IS_CRIME"] == 1) & (df["IS_TRAFFIC"] == 0)]
    report.record("IS_CRIME = 1 and IS_TRAFFIC = 0", df)

    b = cfg.data.bbox
    in_box = df["GEO_LAT"].between(b.min_lat, b.max_lat) & df["GEO_LON"].between(b.min_lon, b.max_lon)
    df = df[in_box]  # also drops missing (NaN) and zeroed coordinates
    report.record("valid coordinates inside Denver bbox", df)

    start = pd.Timestamp(cfg.data.start_date)
    occurred = df["FIRST_OCCURRENCE_DATE"]
    df = df[(occurred >= start) & (occurred < cutoff)]
    report.record(f"occurred in [{start.date()}, {cutoff.date()})", df)

    groups = df["OFFENSE_CATEGORY_ID"].map(cfg.category_to_group).fillna(UNMAPPED)
    unmapped = df.loc[groups == UNMAPPED, "OFFENSE_CATEGORY_ID"].value_counts(dropna=False)
    if len(unmapped):
        report.unmapped_categories = {str(k): int(v) for k, v in unmapped.items()}
        log.warning("Categories missing from [groups] (excluded): %s", report.unmapped_categories)
    df = df.assign(group=groups)
    df = df[df["group"] != UNMAPPED]
    report.record("category mapped to a group", df)
    return df, report


def to_incidents(offenses: pd.DataFrame) -> pd.DataFrame:
    """One row per (INCIDENT_ID, group).

    A single incident often has several offense rows; counting offenses would let one event
    count two or three times. An incident with offenses in two groups counts once in each.
    Time and place come from the earliest-occurring offense of the incident.
    """
    ordered = offenses.sort_values(["INCIDENT_ID", "FIRST_OCCURRENCE_DATE", "OFFENSE_ID"])
    aggs = {
        "occurred_at": ("FIRST_OCCURRENCE_DATE", "first"),
        "reported_at": ("REPORTED_DATE", "first"),
        "lat": ("GEO_LAT", "first"),
        "lon": ("GEO_LON", "first"),
        "n_offenses": ("OFFENSE_ID", "size"),
    }
    if "NEIGHBORHOOD_ID" in ordered:
        aggs["neighborhood"] = ("NEIGHBORHOOD_ID", "first")
    return ordered.groupby(["INCIDENT_ID", "group"], sort=False).agg(**aggs).reset_index()
