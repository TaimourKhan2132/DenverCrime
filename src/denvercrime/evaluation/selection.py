"""Model selection on the validation year only: feature-group ablation and hyperparameter tuning.

Neither function looks at the test weeks. Their outputs are copied into the config by hand
(`[features] drop_groups`, `[model.overrides.<target>]`), so every choice is visible and
reviewable before the test backtest is run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import optuna
import pandas as pd

from denvercrime.config import Config
from denvercrime.evaluation.backtest import Split, to_markdown
from denvercrime.evaluation.metrics import poisson_deviance
from denvercrime.features.build import ABLATION_GROUPS, feature_group, select_features
from denvercrime.features.panel import count_column
from denvercrime.models import gbm

log = logging.getLogger(__name__)


def validation_deviance(split: Split, features: list[str], target: str, params: dict, cfg: Config) -> tuple[float, int]:
    """Poisson deviance on the validation year, and the early-stopped number of rounds."""
    label = count_column(target)
    fit = gbm.fit_with_early_stopping(split.train, split.val, features, label, params,
                                      cfg.model.num_boost_round, cfg.model.early_stopping_rounds)
    pred = gbm.predict(fit.booster, split.val, features)
    return poisson_deviance(split.val[label].to_numpy(), pred), fit.best_iteration


def ablate(split: Split, all_features: list[str], cfg: Config, out_dir: Path) -> pd.DataFrame:
    """Drop one feature group at a time and measure the change in validation deviance.

    A positive change means the model got worse without the group, so the group helps.
    Uses the shared parameters without per-target overrides, so ablation and tuning stay separate.
    """
    groups = list(cfg.groups)
    present = {feature_group(f, groups)[1] for f in all_features} | {"cross_group"}
    candidates = [g for g in ABLATION_GROUPS if g in present]
    rows = []
    for target in cfg.model.targets:
        full = select_features(all_features, target, groups)
        base, _ = validation_deviance(split, full, target, dict(cfg.model.lightgbm), cfg)
        rows.append({"target": target, "dropped": "(none)", "n_features": len(full), "val_deviance": base, "change": 0.0})
        log.info("  [%s] all %d features: validation deviance %.4f", target, len(full), base)
        for group in candidates:
            kept = select_features(all_features, target, groups, (group,))
            if len(kept) == len(full):
                continue
            dev, _ = validation_deviance(split, kept, target, dict(cfg.model.lightgbm), cfg)
            rows.append({"target": target, "dropped": group, "n_features": len(kept), "val_deviance": dev, "change": dev - base})
            log.info("  [%s] without %-14s %.4f (%+.4f)", target, group, dev, dev - base)
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "ablation.csv", index=False)
    wide = table[table["dropped"] != "(none)"].pivot(index="dropped", columns="target", values="change")
    wide.index.name = "dropped group"
    (out_dir / "ablation.md").write_text(
        "# Feature-group ablation (validation year)\n\n"
        "Change in validation Poisson deviance when the group is removed "
        "(positive = the group helps).\n\n" + to_markdown(wide) + "\n")
    return table


def search_space(trial: optuna.Trial) -> dict:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        "objective": trial.suggest_categorical("objective", ["poisson", "tweedie"]),
    }
    if params["objective"] == "tweedie":
        params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.6)
    return params


def tune(split: Split, all_features: list[str], cfg: Config, out_dir: Path) -> dict[str, dict]:
    """Optuna search per target, scored by validation Poisson deviance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best: dict[str, dict] = {}
    for target in cfg.model.targets:
        features = select_features(all_features, target, list(cfg.groups), cfg.features.dropped_for(target))
        default_dev, _ = validation_deviance(split, features, target, dict(cfg.model.lightgbm), cfg)

        def objective(trial: optuna.Trial) -> float:
            params = {**cfg.model.lightgbm, **search_space(trial)}
            dev, rounds = validation_deviance(split, features, target, params, cfg)
            trial.set_user_attr("best_iteration", rounds)
            return dev

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.tuning.seed))
        study.enqueue_trial({k: v for k, v in cfg.model.lightgbm.items() if k in
                             ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
                              "bagging_fraction", "lambda_l2", "objective")})
        study.optimize(objective, n_trials=cfg.tuning.n_trials)
        study.trials_dataframe().to_csv(out_dir / f"trials_{target}.csv", index=False)
        best[target] = {
            "params": study.best_params,
            "val_deviance": study.best_value,
            "default_val_deviance": default_dev,
            "best_iteration": study.best_trial.user_attrs["best_iteration"],
        }
        log.info("  [%s] best validation deviance %.4f (defaults %.4f) with %s",
                 target, study.best_value, default_dev, study.best_params)

    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    snippet = []
    for target, res in best.items():
        snippet.append(f"[model.overrides.{target}]  # validation deviance {res['val_deviance']:.4f} "
                       f"(defaults {res['default_val_deviance']:.4f})")
        for k, v in res["params"].items():
            snippet.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v!r}")
        snippet.append("")
    (out_dir / "overrides.toml").write_text("\n".join(snippet))
    return best
