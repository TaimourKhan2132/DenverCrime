"""Shared fixtures: a synthetic raw snapshot shaped like the real FeatureServer layer."""

from __future__ import annotations

import copy
import tomllib

import numpy as np
import pandas as pd
import pytest

from denvercrime.config import DEFAULT_CONFIG, Config

# (lat, lon, mean incidents per week) — a few Denver-area hotspots of different intensity.
CENTERS = [
    (39.7392, -104.9903, 9.0),   # downtown
    (39.7530, -105.0000, 5.0),   # LoDo
    (39.6780, -104.9620, 3.0),   # south
    (39.7810, -104.8860, 2.0),   # northeast
    (39.7040, -105.0500, 1.0),   # west
]
CATEGORIES = [
    "theft-from-motor-vehicle", "larceny", "auto-theft", "burglary", "arson",
    "aggravated-assault", "robbery", "murder", "other-crimes-against-persons",
    "drug-alcohol", "public-disorder", "all-other-crimes", "white-collar-crime",
]
CATEGORY_P = np.array([18, 16, 15, 7, 0.3, 4, 2, 0.1, 4, 5, 15, 12, 2], dtype=float)
CATEGORY_P /= CATEGORY_P.sum()
START = pd.Timestamp("2022-01-03")
N_WEEKS = 110


def make_raw(seed: int = 0, n_weeks: int = N_WEEKS) -> pd.DataFrame:
    """Offense-level rows with API dtypes (dates already parsed as naive local time)."""
    rng = np.random.default_rng(seed)
    rows = []
    incident_no = 0
    for w in range(n_weeks):
        season = 1.0 + 0.3 * np.sin(2 * np.pi * w / 52.0)
        for lat0, lon0, rate in CENTERS:
            for _ in range(rng.poisson(rate * season)):
                incident_no += 1
                inc_id = f"DP{2022000000 + incident_no}"
                lat, lon = lat0 + rng.normal(0, 0.004), lon0 + rng.normal(0, 0.005)
                first = START + pd.Timedelta(weeks=w) + pd.Timedelta(minutes=int(rng.integers(0, 7 * 24 * 60)))
                last = first + pd.Timedelta(minutes=int(rng.integers(0, 12 * 60))) if rng.random() < 0.6 else pd.NaT
                reported = (last if pd.notna(last) else first) + pd.Timedelta(minutes=int(rng.integers(5, 48 * 60)))
                n_off = 2 if rng.random() < 0.15 else 1
                for k in range(n_off):
                    cat = CATEGORIES[rng.choice(len(CATEGORIES), p=CATEGORY_P)]
                    rows.append(_row(inc_id, k, cat, first, last, reported, lat, lon))
    df = pd.DataFrame(rows)
    df["OBJECTID"] = np.arange(1, len(df) + 1)
    return df


def _row(inc_id, k, cat, first, last, reported, lat, lon, is_crime=1, is_traffic=0):
    return {
        "INCIDENT_ID": inc_id,
        "OFFENSE_ID": f"{inc_id}{k:04d}",
        "OFFENSE_CODE": "2399",
        "OFFENSE_CODE_EXTENSION": 0,
        "OFFENSE_TYPE_ID": f"{cat}-type",
        "OFFENSE_CATEGORY_ID": cat,
        "FIRST_OCCURRENCE_DATE": first,
        "LAST_OCCURRENCE_DATE": last,
        "REPORTED_DATE": reported,
        "INCIDENT_ADDRESS": "100 N TEST ST",
        "GEO_X": 3140000,
        "GEO_Y": 1690000,
        "GEO_LON": lon,
        "GEO_LAT": lat,
        "DISTRICT_ID": "6",
        "PRECINCT_ID": "611",
        "NEIGHBORHOOD_ID": "capitol-hill",
        "IS_CRIME": is_crime,
        "IS_TRAFFIC": is_traffic,
        "VICTIM_COUNT": 1.0,
    }


def add_dirty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the cleaner must drop: bad coordinates, traffic, an unmapped category, the airport
    neighbourhood and an institutional address."""
    t = START + pd.Timedelta(weeks=30)
    bad = [
        _row("BAD1", 0, "larceny", t, pd.NaT, t, np.nan, np.nan),
        _row("BAD2", 0, "larceny", t, pd.NaT, t, 0.0, 0.0),
        _row("BAD3", 0, "larceny", t, pd.NaT, t, 40.5, -104.99),          # north of the bbox
        _row("BAD4", 0, "larceny", t, pd.NaT, t, 39.74, -104.99, is_traffic=1),
        _row("BAD5", 0, "mystery-category", t, pd.NaT, t, 39.74, -104.99),
        {**_row("BAD6", 0, "auto-theft", t, pd.NaT, t, 39.85, -104.67), "NEIGHBORHOOD_ID": "dia"},
        {**_row("BAD7", 0, "aggravated-assault", t, pd.NaT, t, 39.73, -104.99), "INCIDENT_ADDRESS": "490  w colfax ave"},
    ]
    out = pd.concat([df, pd.DataFrame(bad)], ignore_index=True)
    out["OBJECTID"] = np.arange(1, len(out) + 1)
    return out


def make_config(tmp_path, **overrides) -> Config:
    with open(DEFAULT_CONFIG, "rb") as f:
        raw = tomllib.load(f)
    raw = copy.deepcopy(raw)
    raw["data"]["start_date"] = str(START.date())
    raw["split"].update(train_end="2023-03-31", val_end="2023-09-30")
    raw["panel"]["min_train_incidents"] = 3
    raw["model"].update(num_boost_round=60, early_stopping_rounds=15)
    raw["model"]["lightgbm"].update(min_data_in_leaf=20, learning_rate=0.1)
    raw["tuning"]["n_trials"] = 3
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        raw[section][key] = value
    return Config.from_dict(raw, root=tmp_path)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def cfg(tmp_path) -> Config:
    return make_config(tmp_path)
