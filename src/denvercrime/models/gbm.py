"""LightGBM count model (Poisson objective by default)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    booster: lgb.Booster
    best_iteration: int
    val_scores: dict[str, float]


def _dataset(frame: pd.DataFrame, features: list[str], label: str, reference=None) -> lgb.Dataset:
    return lgb.Dataset(frame[features], label=frame[label], reference=reference, free_raw_data=False)


def fit_with_early_stopping(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    label: str,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> FitResult:
    dtrain = _dataset(train, features, label)
    dval = _dataset(val, features, label, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    best = booster.best_iteration or num_boost_round
    scores = {k: float(v) for k, v in booster.best_score.get("val", {}).items()}
    log.info("    early stopping at %d rounds, val %s", best, scores)
    return FitResult(booster, best, scores)


def refit(frame: pd.DataFrame, features: list[str], label: str, params: dict, num_rounds: int) -> lgb.Booster:
    """Retrain on train + validation with the round count chosen by early stopping."""
    return lgb.train(params, _dataset(frame, features, label), num_boost_round=max(num_rounds, 1))


def predict(booster: lgb.Booster, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return booster.predict(frame[features], num_iteration=booster.best_iteration or None)


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
