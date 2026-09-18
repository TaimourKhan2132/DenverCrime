"""Temporal train/validation/test backtest for every target group."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from denvercrime.config import Config
from denvercrime.evaluation.metrics import evaluate, paired_weekly_deviance
from denvercrime.features.panel import count_column
from denvercrime.models import gbm
from denvercrime.models.baselines import baseline_prediction

log = logging.getLogger(__name__)

MODEL_NAME = "lightgbm"


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict:
        def span(df: pd.DataFrame) -> dict:
            if df.empty:
                return {"rows": 0}
            return {"rows": len(df), "first_week": str(df["week"].min().date()),
                    "last_week": str(df["week"].max().date()), "weeks": int(df["week"].nunique())}
        return {"train": span(self.train), "val": span(self.val), "test": span(self.test)}


def temporal_split(frame: pd.DataFrame, cfg: Config) -> Split:
    """Split by week: train <= train_end < val <= val_end < test.

    The first `min_history_weeks` weeks are dropped because most history features are empty.
    Every row of a week lands in exactly one split, and all features are lagged, so no split
    sees counts from a later one.
    """
    weeks = np.sort(frame["week"].unique())
    if len(weeks) <= cfg.split.min_history_weeks:
        raise ValueError("not enough weeks for the configured min_history_weeks")
    frame = frame[frame["week"] >= weeks[cfg.split.min_history_weeks]]
    train_end, val_end = pd.Timestamp(cfg.split.train_end), pd.Timestamp(cfg.split.val_end)
    split = Split(
        train=frame[frame["week"] <= train_end],
        val=frame[(frame["week"] > train_end) & (frame["week"] <= val_end)],
        test=frame[frame["week"] > val_end],
    )
    for name in ("train", "val", "test"):
        if getattr(split, name).empty:
            raise ValueError(f"the {name} split is empty; check [split] dates against the data range")
    return split


def run_target(split: Split, features: list[str], target: str, cfg: Config) -> tuple[pd.DataFrame, dict, pd.DataFrame, object]:
    label = count_column(target)
    params = dict(cfg.model.lightgbm)

    log.info("  [%s] fitting with early stopping on validation", target)
    fit = gbm.fit_with_early_stopping(
        split.train, split.val, features, label, params,
        cfg.model.num_boost_round, cfg.model.early_stopping_rounds,
    )
    history = pd.concat([split.train, split.val])
    log.info("  [%s] refitting on train + validation (%d rounds)", target, fit.best_iteration)
    booster = gbm.refit(history, features, label, params, fit.best_iteration)

    test = split.test[["cell", "week", "area_km2", label]].copy()
    test[MODEL_NAME] = gbm.predict(booster, split.test, features)
    for name in cfg.evaluation.baselines:
        test[name] = baseline_prediction(name, split.test, target, cfg.features.horizon, history=history)

    predictors = [MODEL_NAME, *cfg.evaluation.baselines]
    metrics = {p: evaluate(test, p, label, cfg.evaluation.hotspot_k) for p in predictors}
    paired = {b: paired_weekly_deviance(test, MODEL_NAME, b, label) for b in cfg.evaluation.baselines}
    info = {"best_iteration": fit.best_iteration, "validation": fit.val_scores}
    result = {"metrics": metrics, "paired_vs_baselines": paired, "model": info}
    return test, result, gbm.feature_importance(booster), booster


def metrics_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    table = pd.DataFrame(metrics).T
    table.index.name = "predictor"
    return table


def to_markdown(table: pd.DataFrame, digits: int = 4) -> str:
    cols = [table.index.name or "", *map(str, table.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in table.iterrows():
        cells = [
            str(int(v)) if isinstance(v, float) and v.is_integer() else
            f"{v:.{digits}f}" if isinstance(v, float) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def run_backtest(frame: pd.DataFrame, features: list[str], cfg: Config, run_dir: Path) -> dict:
    """Fit, predict and score every target; write artefacts to `run_dir`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    split = temporal_split(frame, cfg)
    summary = {"split": split.describe(), "features": features, "targets": {}}
    log.info("Split: %s", summary["split"])

    report_lines = ["# Backtest results", ""]
    for target in cfg.model.targets:
        preds, result, importance, booster = run_target(split, features, target, cfg)
        preds.to_parquet(run_dir / f"predictions_{target}.parquet", index=False)
        importance.to_csv(run_dir / f"feature_importance_{target}.csv", index=False)
        booster.save_model(str(run_dir / f"model_{target}.txt"))
        summary["targets"][target] = result

        table = metrics_table(result["metrics"])
        table.to_csv(run_dir / f"metrics_{target}.csv")
        paired = metrics_table(result["paired_vs_baselines"])
        paired.index.name = f"{MODEL_NAME} vs"
        report_lines += [
            f"## {target}", "", to_markdown(table), "",
            "Weekly Poisson deviance, LightGBM minus baseline (negative = LightGBM better; "
            "95% CI bootstrapped over weeks):", "", to_markdown(paired), "",
        ]
        log.info("\n%s\n%s\n%s", target, table.round(4).to_string(), paired.round(4).to_string())

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "metrics.md").write_text("\n".join(report_lines))
    return summary
