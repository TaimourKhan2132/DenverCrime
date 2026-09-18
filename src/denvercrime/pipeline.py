"""End-to-end steps: prepare, explore, ablate, tune, backtest and maps."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from denvercrime.config import Config
from denvercrime.data.clean import clean_offenses, to_incidents
from denvercrime.data.load import find_latest_raw, load_raw
from denvercrime.evaluation.backtest import MODEL_NAME, run_backtest, temporal_split
from denvercrime.features.build import build_features
from denvercrime.features.panel import assign_cells, build_panel, complete_weeks, count_column, select_cells
from denvercrime.features.spatial import cell_table, neighbor_matrix
from denvercrime.viz.maps import hex_map, plot_forecast_map, plot_hotspot_curve, plot_weekly_totals

log = logging.getLogger(__name__)

FEATURES_FILE = "features.parquet"
FEATURE_LIST_FILE = "feature_columns.json"
INCIDENTS_FILE = "incidents.parquet"
REPORT_FILE = "prepare_report.json"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def prepare(cfg: Config, raw_path: Path | None = None) -> dict:
    """Raw snapshot -> incidents -> (cell x week) panel -> feature table on disk."""
    raw_path = Path(raw_path) if raw_path else find_latest_raw(cfg.data.raw_dir)
    raw = load_raw(raw_path, cfg.data.csv_time_shift_hours)

    log.info("Cleaning")
    offenses, cleaning = clean_offenses(raw, cfg)
    incidents = assign_cells(to_incidents(offenses), cfg.panel.h3_resolution)
    groups = list(cfg.groups)

    cells = select_cells(incidents, cfg.split.train_end, cfg.panel.min_train_incidents)
    if not cells:
        raise ValueError("no cells meet min_train_incidents in the training period")
    weeks = complete_weeks(pd.Timestamp(cfg.data.start_date), pd.Timestamp(cleaning.cutoff))
    panel = build_panel(incidents, groups, cells, weeks)
    info = cell_table(cells)
    panel["area_km2"] = panel["cell"].map(info["area_km2"]).to_numpy()

    log.info("Building features for %d cells x %d weeks", len(cells), len(weeks))
    fcfg = cfg.features
    neighbors = neighbor_matrix(cells, fcfg.neighbor_ring)
    outer = neighbor_matrix(cells, fcfg.outer_ring, annulus=True) if fcfg.outer_ring else None
    frame, feature_cols = build_features(panel, groups, fcfg, neighbors, info, outer)

    in_window = incidents[incidents["week"].isin(weeks)]
    report = {
        "raw_file": str(raw_path),
        "time_shift_hours_applied": raw.attrs.get("time_shift_hours", 0.0),
        "cleaning": cleaning.to_dict(),
        "incidents_by_group": in_window["group"].value_counts().to_dict(),
        "h3_resolution": cfg.panel.h3_resolution,
        "cells": len(cells),
        "weeks": len(weeks),
        "first_week": str(weeks[0].date()),
        "last_week": str(weeks[-1].date()),
        "incident_coverage_of_selected_cells": float(in_window["cell"].isin(cells).mean()),
        "panel_rows": len(frame),
        "n_features": len(feature_cols),
    }

    out = cfg.data.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    incidents.to_parquet(out / INCIDENTS_FILE, index=False)
    frame.to_parquet(out / FEATURES_FILE, index=False)
    (out / FEATURE_LIST_FILE).write_text(json.dumps(feature_cols, indent=2))
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2, default=str))
    log.info("Prepared %s panel rows, %d features -> %s", f"{len(frame):,}", len(feature_cols), out)
    return report


def load_features(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    out = cfg.data.processed_dir
    if not (out / FEATURES_FILE).exists():
        raise FileNotFoundError(f"{out / FEATURES_FILE} not found; run `python -m denvercrime prepare` first")
    frame = pd.read_parquet(out / FEATURES_FILE)
    features = json.loads((out / FEATURE_LIST_FILE).read_text())
    return frame, features


def explore(cfg: Config, out_dir: Path | None = None) -> dict:
    from denvercrime.viz.explore import explore as run_explore

    frame, _ = load_features(cfg)
    return run_explore(frame, list(cfg.groups), list(cfg.model.targets), out_dir or cfg.runs_dir.parent / "explore")


def ablate(cfg: Config) -> Path:
    from denvercrime.evaluation.selection import ablate as run_ablate

    frame, features = load_features(cfg)
    out_dir = cfg.runs_dir.parent / "ablation" / _timestamp()
    run_ablate(temporal_split(frame, cfg), features, cfg, out_dir)
    log.info("Ablation written to %s", out_dir)
    return out_dir


def tune(cfg: Config) -> Path:
    from denvercrime.evaluation.selection import tune as run_tune

    frame, features = load_features(cfg)
    out_dir = cfg.runs_dir.parent / "tuning" / _timestamp()
    run_tune(temporal_split(frame, cfg), features, cfg, out_dir)
    log.info("Tuning written to %s; copy overrides.toml into the config to use it", out_dir)
    return out_dir


def backtest(cfg: Config, config_path: Path | None = None) -> Path:
    frame, features = load_features(cfg)
    run_dir = cfg.runs_dir / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copy(config_path, run_dir / "config.toml")
    run_backtest(frame, features, cfg, run_dir)
    log.info("Backtest written to %s", run_dir)
    return run_dir


def latest_run(cfg: Config) -> Path:
    runs = sorted(p for p in cfg.runs_dir.glob("*") if (p / "summary.json").exists())
    if not runs:
        raise FileNotFoundError(f"no completed runs in {cfg.runs_dir}; run `python -m denvercrime backtest`")
    return runs[-1]


def make_maps(cfg: Config, run_dir: Path | None = None, week: str | None = None) -> list[Path]:
    """Interactive and static maps for one test week plus weekly-total and hotspot plots, per target."""
    run_dir = run_dir or latest_run(cfg)
    outputs = []
    for target in cfg.model.targets:
        preds = pd.read_parquet(run_dir / f"predictions_{target}.parquet")
        y_col = count_column(target)
        chosen = pd.Timestamp(week) if week else preds["week"].max()
        week_frame = preds[preds["week"] == chosen]
        if week_frame.empty:
            raise ValueError(f"week {chosen.date()} is not in the test period of {run_dir.name}")
        layers = {"forecast (LightGBM)": MODEL_NAME, "actual": y_col}
        if "moving_average_52" in week_frame:
            layers["52-week average"] = "moving_average_52"
        title = f"{target} incidents, week of {chosen.date()}"
        outputs.append(hex_map(week_frame, layers, title, run_dir / f"map_{target}_{chosen.date()}.html"))
        outputs.append(plot_forecast_map(week_frame, MODEL_NAME, y_col, title, run_dir / f"forecast_map_{target}.png"))
        compare = [MODEL_NAME] + [b for b in ("moving_average_52", "last_week") if b in preds]
        outputs.append(plot_weekly_totals(preds, y_col, compare, run_dir / f"weekly_totals_{target}.png"))
        outputs.append(plot_hotspot_curve(preds, y_col, compare, run_dir / f"hotspot_curve_{target}.png"))
    for p in outputs:
        log.info("Wrote %s", p)
    return outputs
