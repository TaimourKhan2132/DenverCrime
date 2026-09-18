# Denver Crime Forecasting

Weekly crime-count forecasts for every ~0.74 km² H3 hexagon in Denver, built from the city's
open NIBRS offense data. Forecasts are evaluated on a held-out future period against naive
baselines, using both count errors and hotspot metrics (hit rate, PAI, PEI).

> **Status:** work in progress. The data layer (download → load → clean) is in place and
> tested. The panel, features, models and evaluation are next; see [Roadmap](#roadmap).

## Data

**Source:** Denver Open Data Catalog, [Crime](https://opendata-geospatialdenver.hub.arcgis.com/datasets/geospatialDenver::crime/about)
(ArcGIS FeatureServer `ODC_CRIME_OFFENSES_P`, layer 324). NIBRS offenses for the previous five
calendar years plus the current year, updated Monday–Friday.

Checked against the live service on 2026-09-18: **379,705 offense rows**, first occurrence
2021-01-01 through 2026-09-14, 13 offense categories.

What the data layer handles:

- **Paged download.** The service returns at most 2,000 rows per request. The downloader pages
  by `OBJECTID` (keyset pagination), which stays correct even if the layer is updated while the
  download is running. It retries on transient errors and saves a Parquet snapshot plus a
  metadata JSON.
- **Timestamps in Hub CSV exports are shifted.** The CSV download reports `FIRST_OCCURRENCE_DATE`
  and `REPORTED_DATE` exactly **+7 h** later than the API, while `LAST_OCCURRENCE_DATE` is not
  shifted, which makes many incidents end before they start. The API values satisfy
  FIRST ≤ LAST ≤ REPORTED. The loader detects the shift and undoes it (`csv_time_shift_hours = "auto"`).
- **One incident, several offenses.** Rows are offenses; in a 2020–2025 extract, 11% of rows
  belonged to multi-offense incidents. Cleaning collapses rows to one per incident and
  forecasting group, so one event is counted once.
- **Recent weeks are incomplete.** DPD notes that records are most accurate once they are at
  least 30 days old, so the last 30 days before the snapshot are excluded (`drop_recent_days`).

## Quickstart (Windows / PowerShell)

```powershell
py -3.11 -m venv $HOME\.venvs\denvercrime          # keep the environment outside OneDrive
& $HOME\.venvs\denvercrime\Scripts\python.exe -m pip install -e ".[dev]"
& $HOME\.venvs\denvercrime\Scripts\python.exe -m pytest   # synthetic data, no download needed
```

Settings (offense groups, date range, bounding box, split dates, model parameters) are in
[`configs/default.toml`](configs/default.toml).

## Roadmap

- [x] Downloader, loader (with the CSV timestamp fix), cleaning and incident deduplication
- [ ] H3 cell × week panel with explicit zeros; lagged, spatial and calendar features
- [ ] LightGBM (Poisson) and naive baselines, with a strict temporal train/validation/test split
- [ ] Evaluation: MAE, Poisson deviance, hit rate, PAI, PEI, and a week-by-week paired comparison
- [ ] Hexagon maps, diagnostic plots and a command-line interface
