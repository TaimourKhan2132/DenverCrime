"""End-to-end smoke test on the synthetic snapshot: prepare -> backtest -> maps."""

import json

import pandas as pd
import pytest

from denvercrime import pipeline
from denvercrime.evaluation.backtest import MODEL_NAME, temporal_split
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
    assert set(features) <= set(frame.columns)
    assert not any(f.startswith("y_") for f in features)  # targets never leak in as features


def test_temporal_split_is_ordered_and_disjoint(prepared):
    cfg, _ = prepared
    frame, _ = pipeline.load_features(cfg)
    split = temporal_split(frame, cfg)
    assert split.train["week"].max() < split.val["week"].min()
    assert split.val["week"].max() < split.test["week"].min()
    assert len(split.train) + len(split.val) + len(split.test) < len(frame)  # warm-up weeks dropped


def test_backtest_and_maps(prepared):
    cfg, _ = prepared
    run_dir = pipeline.backtest(cfg)
    summary = json.loads((run_dir / "summary.json").read_text())
    for target in cfg.model.targets:
        metrics = summary["targets"][target]["metrics"]
        assert set(metrics) == {MODEL_NAME, *cfg.evaluation.baselines}
        # The synthetic data has stable hotspots, so the model must beat "same as last week".
        assert metrics[MODEL_NAME]["poisson_deviance"] < metrics["last_week"]["poisson_deviance"]
        paired = summary["targets"][target]["paired_vs_baselines"]
        assert set(paired) == set(cfg.evaluation.baselines)
        assert paired["last_week"]["ci95_low"] <= paired["last_week"]["mean_diff"] <= paired["last_week"]["ci95_high"]
        preds = pd.read_parquet(run_dir / f"predictions_{target}.parquet")
        assert (preds[MODEL_NAME] >= 0).all()
        assert (run_dir / f"model_{target}.txt").exists()

    outputs = pipeline.make_maps(cfg, run_dir)
    assert all(p.exists() for p in outputs)
    assert (run_dir / "metrics.md").read_text().startswith("# Backtest results")
