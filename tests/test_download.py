import json
import re

import pandas as pd

from denvercrime.data.download import download, iter_pages, records_to_frame


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Mimics the ArcGIS query endpoint over an in-memory table."""

    def __init__(self, rows, reported_count=None):
        self.rows = sorted(rows, key=lambda r: r["OBJECTID"])
        self.reported_count = reported_count if reported_count is not None else len(rows)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        if params.get("returnCountOnly") == "true":
            return FakeResponse({"count": self.reported_count})
        last = int(re.match(r"OBJECTID > (-?\d+)", params["where"]).group(1))
        page = [r for r in self.rows if r["OBJECTID"] > last][: params["resultRecordCount"]]
        return FakeResponse({"features": [{"attributes": r} for r in page]})


def _rows(n):
    base_ms = int(pd.Timestamp("2024-05-23 23:55").timestamp() * 1000)
    return [
        {"OBJECTID": 10 + i * 3, "INCIDENT_ID": f"I{i}", "OFFENSE_ID": f"O{i}",
         "FIRST_OCCURRENCE_DATE": base_ms, "LAST_OCCURRENCE_DATE": None, "REPORTED_DATE": base_ms + 3_600_000}
        for i in range(n)
    ]


def test_keyset_pagination_returns_every_row_once():
    session = FakeSession(_rows(25))
    pages = list(iter_pages(session, "https://example/324", page_size=10))
    assert [len(p) for p in pages] == [10, 10, 5]
    ids = [r["OBJECTID"] for p in pages for r in p]
    assert ids == sorted(set(ids)) and len(ids) == 25


def test_epoch_ms_become_naive_wall_clock():
    df = records_to_frame(_rows(1))
    assert df.loc[0, "FIRST_OCCURRENCE_DATE"] == pd.Timestamp("2024-05-23 23:55")
    assert pd.isna(df.loc[0, "LAST_OCCURRENCE_DATE"])
    assert df["FIRST_OCCURRENCE_DATE"].dt.tz is None


def test_download_writes_parquet_and_metadata(cfg):
    session = FakeSession(_rows(7), reported_count=8)  # count mismatch only warns
    out = download(cfg, session=session)
    df = pd.read_parquet(out)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert len(df) == 7 and meta["rows"] == 7 and meta["server_count"] == 8
    assert out.parent == cfg.data.raw_dir
