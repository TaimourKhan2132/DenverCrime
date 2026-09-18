import numpy as np
import pandas as pd
import pytest

from denvercrime.evaluation.metrics import (
    evaluate,
    hotspot_metrics,
    occurrence_metrics,
    paired_weekly_deviance,
    poisson_deviance,
    roc_auc,
    select_top_area,
)


def test_poisson_deviance():
    y = np.array([0, 1, 3, 0])
    assert poisson_deviance(y, y.astype(float)) == pytest.approx(0, abs=1e-6)
    assert poisson_deviance(y, np.full(4, 1.0)) > 0
    assert np.isfinite(poisson_deviance(np.array([0, 0]), np.array([0.0, 0.0])))


def test_select_top_area_reaches_the_budget():
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    area = np.ones(5)
    rng = np.random.default_rng(0)
    assert select_top_area(scores, area, 0.2, rng).tolist() == [True, False, False, False, False]
    assert select_top_area(scores, area, 0.3, rng).sum() == 2  # smallest set covering >= 30%


def _week(pred, y, week="2024-01-01"):
    n = len(y)
    return pd.DataFrame({"week": pd.Timestamp(week), "cell": [f"c{i}" for i in range(n)],
                         "area_km2": 1.0, "pred": pred, "y": y})


def test_hotspot_metrics_perfect_and_worst_predictors():
    y = np.array([6, 3, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    perfect = hotspot_metrics(_week(y, y), "pred", "y", 0.1)
    assert perfect["hit_rate"] == pytest.approx(0.6)
    assert perfect["pai"] == pytest.approx(6.0)
    assert perfect["pei"] == pytest.approx(1.0)
    worst = hotspot_metrics(_week(-y, y), "pred", "y", 0.1)
    assert worst["hit_rate"] == 0 and worst["pei"] == 0


def test_weeks_without_incidents_are_skipped():
    frame = pd.concat([_week(np.ones(4), np.zeros(4), "2024-01-01"),
                       _week(np.arange(4.0), np.array([0, 0, 1, 3.0]), "2024-01-08")])
    assert hotspot_metrics(frame, "pred", "y", 0.25)["weeks"] == 1


def test_paired_weekly_deviance_favours_the_better_predictor():
    rng = np.random.default_rng(1)
    frames = []
    for i, week in enumerate(pd.date_range("2024-01-01", periods=20, freq="7D")):
        mu = rng.uniform(0.5, 3, 30)
        frames.append(pd.DataFrame({"week": week, "y": rng.poisson(mu), "good": mu, "bad": mu[::-1]}))
    out = paired_weekly_deviance(pd.concat(frames), "good", "bad", "y")
    assert out["mean_diff"] < 0 and out["ci95_high"] < 0
    assert out["weeks"] == 20 and out["weeks_model_better"] >= 18


def test_roc_auc_and_occurrence_metrics():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert roc_auc(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)  # ties count half
    y = np.array([0, 0, 1, 3])
    pred = np.array([0.1, 1.0, 2.0, 0.2])  # P(>=1) = 0.10, 0.63, 0.86, 0.18 -> called: F T T F
    out = occurrence_metrics(y, pred)
    assert out["occ_accuracy"] == pytest.approx(0.5)
    assert out["occ_precision"] == pytest.approx(0.5) and out["occ_recall"] == pytest.approx(0.5)
    assert out["occ_f1"] == pytest.approx(0.5)


def test_evaluate_reports_all_metrics():
    frame = _week(np.array([1.0, 2.0, 0.5, 0.0]), np.array([1.0, 3.0, 0.0, 0.0]))
    out = evaluate(frame, "pred", "y", (0.25, 0.5))
    assert {"mae", "rmse", "poisson_deviance", "pai@25pct", "pei@50pct", "hit_rate@25pct", "occ_f1", "occ_auc"} <= set(out)
    assert out["mae"] == pytest.approx(np.mean([0, 1, 0.5, 0]))
