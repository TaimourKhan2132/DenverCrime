"""End-to-end smoke tests on the synthetic snapshot: prepare -> explore/ablate/tune -> backtest -> maps."""

import json

import pandas as pd
import pytest

from denvercrime import pipeline
from denvercrime.evaluation.backtest import MODEL_NAME, split_test_windows, temporal_split
from tests.conftest import add_dirty_rows, make_raw


@pytest.fixture
def prepared(cfg):
    cfg.data.raw_dir.mkdir(parents=True)
    add_dirty_rows(make_raw()).to_parquet(cfg.data.raw_dir / "crime_test.parquet", index=False)
    report = pipeline.prepare(cfg)
    return cfg, report


def test_prepare_writes_features(prepared):
    cfg, report = prepared
    frame, features = pipeline.load_features(cfg)
    assert report["cells"] > 5 and report["panel_rows"] == len(frame)
    assert report["cleaning"]["unmapped_categories"] == {"mystery-category": 1}
    assert report["cleaning"]["excluded_places"]["neighborhood:dia"] == 1
    assert set(features) <= set(frame.columns)
    assert not any(f.startswith("y_") for f in features)  # targets never leak in as features


def test_temporal_split_and_test_windows(prepared):
    cfg, _ = prepared
    frame, _ = pipeline.load_features(cfg)
    split = temporal_split(frame, cfg)
    assert split.train["week"].max() < split.val["week"].min()
    assert split.val["week"].max() < split.test["week"].min()
    assert len(split.train) + len(split.val) + len(split.test) == len(split.usable) < len(frame)
    windows = split_test_windows(split.test, cfg.split.test_folds)
    flat = [w for block in windows for w in block]
    assert len(windows) == cfg.split.test_folds
    assert flat == sorted(flat) and len(flat) == len(set(flat)) == split.test["week"].nunique()


def test_backtest_and_maps(prepared):
    cfg, _ = prepared
    run_dir = pipeline.backtest(cfg)
    summary = json.loads((run_dir / "summary.json").read_text())
    for target in cfg.model.targets:
        result = summary["targets"][target]
        metrics = result["metrics"]
        assert set(metrics) == {MODEL_NAME, *cfg.evaluation.baselines}
        # The synthetic data has stable hotspots, so the model must beat "same as last week".
        assert metrics[MODEL_NAME]["poisson_deviance"] < metrics["last_week"]["poisson_deviance"]
        assert 0 <= metrics[MODEL_NAME]["occ_f1"] <= 1
        # Each test window is predicted by a model trained only on earlier weeks.
        for fold in result["folds"]:
            assert fold["trained_through"] < fold["first_week"]
        paired = result["paired_vs_baselines"]["last_week"]
        assert paired["ci95_low"] <= paired["mean_diff"] <= paired["ci95_high"]
        preds = pd.read_parquet(run_dir / f"predictions_{target}.parquet")
        assert (preds[MODEL_NAME] >= 0).all() and preds["fold"].nunique() == cfg.split.test_folds

    outputs = pipeline.make_maps(cfg, run_dir)
    assert all(p.exists() for p in outputs)
    assert (run_dir / "metrics.md").read_text().startswith("# Backtest results")


def test_explore_ablate_and_tune(prepared):
    cfg, _ = prepared
    stats = pipeline.explore(cfg)
    assert 0 < stats["concentration"]["property"]["area_share_holding_50pct"] < 1

    ablation = pd.read_csv(pipeline.ablate(cfg) / "ablation.csv")
    assert {"(none)", "cross_group", "spatial"} <= set(ablation["dropped"])

    best = json.loads((pipeline.tune(cfg) / "best_params.json").read_text())
    for target in cfg.model.targets:
        # The defaults are always the first trial, so tuning can only match or improve them.
        assert best[target]["val_deviance"] <= best[target]["default_val_deviance"] + 1e-12
