"""Validation-year model selection, then a rolling-origin test over several windows.

1. Validation: fit on train (<= train_end) with early stopping on the validation year.
   This fixes the number of boosting rounds; tuning and ablation also use only this stage.
2. Test: the weeks after val_end are split into `test_folds` consecutive windows. Before
   each window the model is refitted on every earlier week and then predicts that window,
   the way a deployed model would be retrained each quarter. The test weeks are never used
   for any choice.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from denvercrime.config import Config
from denvercrime.evaluation.metrics import evaluate, paired_weekly_deviance
from denvercrime.features.build import select_features
from denvercrime.features.panel import count_column
from denvercrime.models import gbm
from denvercrime.models.baselines import baseline_prediction

log = logging.getLogger(__name__)

MODEL_NAME = "lightgbm"
FOLD_COLUMNS = ("poisson_deviance", "mae", "occ_f1", "hit_rate@5pct")


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    usable: pd.DataFrame  # every row after the warm-up weeks (train + val + test)

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
    usable = frame[frame["week"] >= weeks[cfg.split.min_history_weeks]]
    train_end, val_end = pd.Timestamp(cfg.split.train_end), pd.Timestamp(cfg.split.val_end)
    split = Split(
        train=usable[usable["week"] <= train_end],
        val=usable[(usable["week"] > train_end) & (usable["week"] <= val_end)],
        test=usable[usable["week"] > val_end],
        usable=usable,
    )
    for name in ("train", "val", "test"):
        if getattr(split, name).empty:
            raise ValueError(f"the {name} split is empty; check [split] dates against the data range")
    return split


def split_test_windows(test: pd.DataFrame, n_folds: int) -> list[np.ndarray]:
    """Consecutive, non-overlapping blocks of test weeks."""
    weeks = np.sort(test["week"].unique())
    if n_folds > len(weeks):
        raise ValueError(f"test_folds={n_folds} but the test period has only {len(weeks)} weeks")
    return [w for w in np.array_split(weeks, n_folds) if len(w)]


def target_features(features: list[str], target: str, cfg: Config) -> list[str]:
    return select_features(features, target, list(cfg.groups), cfg.features.dropped_for(target))


def _predictions(rows: pd.DataFrame, label: str, model_pred: np.ndarray, target: str,
                 history: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = rows[["cell", "week", "area_km2", label]].copy()
    out[MODEL_NAME] = model_pred
    for name in cfg.evaluation.baselines:
        out[name] = baseline_prediction(name, rows, target, cfg.features.horizon, history=history)
    return out


def score_all(preds: pd.DataFrame, label: str, cfg: Config) -> dict[str, dict[str, float]]:
    return {p: evaluate(preds, p, label, cfg.evaluation.hotspot_k) for p in [MODEL_NAME, *cfg.evaluation.baselines]}


def run_target(split: Split, all_features: list[str], target: str, cfg: Config):
    label = count_column(target)
    features = target_features(all_features, target, cfg)
    params = cfg.model.params(target)

    log.info("  [%s] validation: fit on train, early stopping on the validation year (%d features)",
             target, len(features))
    fit = gbm.fit_with_early_stopping(split.train, split.val, features, label, params,
                                      cfg.model.num_boost_round, cfg.model.early_stopping_rounds)
    val_preds = _predictions(split.val, label, gbm.predict(fit.booster, split.val, features), target, split.train, cfg)
    validation = score_all(val_preds, label, cfg)

    frames, folds = [], []
    windows = split_test_windows(split.test, cfg.split.test_folds)
    for i, weeks in enumerate(windows, start=1):
        history = split.usable[split.usable["week"] < weeks[0]]
        rows = split.test[split.test["week"].isin(weeks)]
        booster = gbm.refit(history, features, label, params, fit.best_iteration)
        fold = _predictions(rows, label, gbm.predict(booster, rows, features), target, history, cfg)
        fold["fold"] = i
        frames.append(fold)
        folds.append({
            "fold": i,
            "trained_through": str(history["week"].max().date()),
            "first_week": str(pd.Timestamp(weeks[0]).date()),
            "last_week": str(pd.Timestamp(weeks[-1]).date()),
            "weeks": len(weeks),
        })
        log.info("  [%s] test window %d/%d: trained through %s, predicting %s .. %s", target, i,
                 len(windows), folds[-1]["trained_through"], folds[-1]["first_week"], folds[-1]["last_week"])

    test = pd.concat(frames, ignore_index=True)
    metrics = score_all(test, label, cfg)
    fold_metrics = {
        p: {f["fold"]: {k: v for k, v in evaluate(test[test["fold"] == f["fold"]], p, label,
                                                  cfg.evaluation.hotspot_k).items() if k in FOLD_COLUMNS}
            for f in folds}
        for p in (MODEL_NAME, "moving_average_52", "historical_mean") if p in test
    }
    paired = {b: paired_weekly_deviance(test, MODEL_NAME, b, label) for b in cfg.evaluation.baselines}
    result = {
        "metrics": metrics,
        "paired_vs_baselines": paired,
        "folds": folds,
        "fold_metrics": fold_metrics,
        "validation": validation,
        "model": {"best_iteration": fit.best_iteration, "params": params, "features": features},
    }
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


def fold_table(fold_metrics: dict, column: str) -> pd.DataFrame:
    table = pd.DataFrame({p: {f"window {k}": v[column] for k, v in folds.items()} for p, folds in fold_metrics.items()}).T
    table.index.name = column
    return table


def run_backtest(frame: pd.DataFrame, features: list[str], cfg: Config, run_dir: Path) -> dict:
    """Fit, predict and score every target; write artefacts to `run_dir`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    split = temporal_split(frame, cfg)
    summary = {"split": split.describe(), "targets": {}}
    log.info("Split: %s", summary["split"])

    report = ["# Backtest results", ""]
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
        folds = pd.DataFrame(result["folds"]).set_index("fold")
        per_window = [
            to_markdown(fold_table(result["fold_metrics"], col)) + "\n"
            for col in FOLD_COLUMNS
            if all(col in m for fm in result["fold_metrics"].values() for m in fm.values())
        ]
        report += [
            f"## {target}", "",
            f"### Test period, all {len(result['folds'])} windows pooled", "", to_markdown(table), "",
            "### Week-by-week Poisson deviance, LightGBM minus baseline",
            "(negative = LightGBM better; 95% CI bootstrapped over weeks)", "", to_markdown(paired), "",
            "### Per test window", "", to_markdown(folds), "", *per_window,
            "### Validation year (used for model selection)", "",
            to_markdown(metrics_table(result["validation"])), "",
        ]
        shown = [c for c in ("poisson_deviance", "mae", "occ_f1", "occ_auc", "hit_rate@5pct") if c in table]
        log.info("\n%s\n%s", target, table[shown].round(4).to_string())

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "metrics.md").write_text("\n".join(report))
    return summary
