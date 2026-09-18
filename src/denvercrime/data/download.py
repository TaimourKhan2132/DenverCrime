"""Download the Denver "Crime" layer from its ArcGIS FeatureServer into a Parquet snapshot.

The service caps each response at `maxRecordCount` (2,000) rows, so records are paged by
OBJECTID (keyset pagination). Keyset paging stays correct even if DPD appends or edits rows
while the download is running, which offset paging does not.

Date fields arrive as epoch milliseconds. Read as naive timestamps they give the local
wall-clock time recorded by DPD (FIRST <= LAST <= REPORTED holds), so no timezone conversion
is applied.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from denvercrime.config import Config
from denvercrime.schema import DATE_COLUMNS

log = logging.getLogger(__name__)

TIMEOUT_S = 60


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "denvercrime-forecasting/0.1 (research)"
    return session


def _get_json(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:  # ArcGIS reports errors with HTTP 200
        raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
    return payload


def fetch_count(session: requests.Session, layer_url: str) -> int:
    payload = _get_json(session, f"{layer_url}/query", {"where": "1=1", "returnCountOnly": "true", "f": "json"})
    return int(payload["count"])


def iter_pages(session: requests.Session, layer_url: str, page_size: int) -> Iterator[list[dict]]:
    """Yield lists of attribute dicts, ordered by OBJECTID."""
    last_id = -1
    while True:
        payload = _get_json(
            session,
            f"{layer_url}/query",
            {
                "where": f"OBJECTID > {last_id}",
                "outFields": "*",
                "orderByFields": "OBJECTID ASC",
                "resultRecordCount": page_size,
                "returnGeometry": "false",
                "f": "json",
            },
        )
        rows = [feature["attributes"] for feature in payload.get("features", [])]
        if not rows:
            return
        yield rows
        last_id = max(row["OBJECTID"] for row in rows)


def records_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(rows)
    for col in DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(pd.to_numeric(df[col], errors="coerce"), unit="ms")
    return df


def download(cfg: Config, session: requests.Session | None = None) -> Path:
    """Download the full layer and write `crime_<UTC timestamp>.parquet` plus a metadata JSON."""
    session = session or make_session()
    layer_url = cfg.data.feature_server.rstrip("/")
    expected = fetch_count(session, layer_url)
    log.info("Server reports %s offense records", f"{expected:,}")

    frames, n = [], 0
    for i, rows in enumerate(iter_pages(session, layer_url, cfg.data.page_size), start=1):
        frames.append(records_to_frame(rows))
        n += len(rows)
        if i % 25 == 0:
            log.info("  %s / %s rows", f"{n:,}", f"{expected:,}")
    if not frames:
        raise RuntimeError("The service returned no records")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("OBJECTID")

    if len(df) != expected:
        log.warning("Downloaded %s rows but the server reported %s (the layer may have been "
                    "updated during the download)", f"{len(df):,}", f"{expected:,}")

    downloaded_at = datetime.now(timezone.utc)
    cfg.data.raw_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.data.raw_dir / f"crime_{downloaded_at:%Y%m%dT%H%M%SZ}.parquet"
    df.to_parquet(out, index=False)

    meta = {
        "source": layer_url,
        "downloaded_at_utc": downloaded_at.isoformat(),
        "rows": len(df),
        "server_count": expected,
        "first_occurrence_min": str(df["FIRST_OCCURRENCE_DATE"].min()),
        "first_occurrence_max": str(df["FIRST_OCCURRENCE_DATE"].max()),
        "reported_max": str(df["REPORTED_DATE"].max()),
        "time_convention": "naive local wall-clock (epoch ms read without timezone conversion)",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s (%s rows)", out, f"{len(df):,}")
    return out
