"""Count-error and hotspot metrics.

Hotspot metrics follow the crime-forecasting literature: each week, flag the highest-ranked
cells until they cover a share `k` of the city's area, then measure what share of that week's
incidents fell inside them.

- hit rate: share of incidents inside the flagged area
- PAI (Predictive Accuracy Index, Chainey et al. 2008): hit rate / area share
- PEI (Predictive Efficiency Index, Hunt 2016): hit rate / best possible hit rate for the same area
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIE_BREAK_SEED = 0


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def poisson_deviance(y: np.ndarray, pred: np.ndarray, eps: float = 1e-9) -> float:
    """Mean Poisson deviance; y log(y / mu) is taken as 0 when y = 0."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(pred, dtype=np.float64), eps, None)
    ratio = np.divide(y, mu, out=np.ones_like(y), where=y > 0)
    return float(2.0 * np.mean(y * np.log(ratio) - (y - mu)))


def select_top_area(scores: np.ndarray, area: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of the highest-scoring cells whose total area first reaches `share` of the total.

    Ties are broken randomly (seeded) so that tied predictions are not ranked by cell id.
    """
    order = np.lexsort((rng.random(len(scores)), -scores))
    cum_area = np.cumsum(area[order])
    budget = share * area.sum()
    n_selected = int(np.searchsorted(cum_area, budget, side="left")) + 1
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[: min(n_selected, len(scores))]] = True
    return mask


def hotspot_metrics(
    frame: pd.DataFrame, pred_col: str, y_col: str, share: float, area_col: str = "area_km2"
) -> dict[str, float]:
    """Hit rate, PAI and PEI averaged over weeks (weeks with no incidents are skipped)."""
    rng = np.random.default_rng(TIE_BREAK_SEED)
    hits, pais, peis = [], [], []
    for _, week in frame.groupby("week", sort=True):
        y = week[y_col].to_numpy(dtype=np.float64)
        total = y.sum()
        if total == 0:
            continue
        area = week[area_col].to_numpy(dtype=np.float64)
        chosen = select_top_area(week[pred_col].to_numpy(dtype=np.float64), area, share, rng)
        best = select_top_area(y, area, share, rng)
        hit = y[chosen].sum() / total
        area_share = area[chosen].sum() / area.sum()
        best_hit = y[best].sum() / total
        hits.append(hit)
        pais.append(hit / area_share)
        peis.append(hit / best_hit if best_hit > 0 else np.nan)
    if not hits:
        return {"hit_rate": np.nan, "pai": np.nan, "pei": np.nan, "weeks": 0}
    return {
        "hit_rate": float(np.mean(hits)),
        "pai": float(np.mean(pais)),
        "pei": float(np.nanmean(peis)),
        "weeks": len(hits),
    }


def paired_weekly_deviance(
    frame: pd.DataFrame, model_col: str, baseline_col: str, y_col: str, n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Compare two predictors week by week on Poisson deviance (model minus baseline).

    Negative = the model is better. The 95% CI is a bootstrap over weeks, which treats
    weeks as the independent unit rather than the thousands of correlated cell-weeks.
    """
    diffs = np.array([
        poisson_deviance(w[y_col].to_numpy(), w[model_col].to_numpy())
        - poisson_deviance(w[y_col].to_numpy(), w[baseline_col].to_numpy())
        for _, w in frame.groupby("week", sort=True)
    ])
    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_diff": float(diffs.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "weeks_model_better": int((diffs < 0).sum()),
        "weeks": len(diffs),
    }


def evaluate(frame: pd.DataFrame, pred_col: str, y_col: str, hotspot_k: tuple[float, ...]) -> dict[str, float]:
    y = frame[y_col].to_numpy(dtype=np.float64)
    pred = frame[pred_col].to_numpy(dtype=np.float64)
    out = {
        "mae": mae(y, pred),
        "rmse": rmse(y, pred),
        "poisson_deviance": poisson_deviance(y, pred),
        "mean_actual": float(y.mean()),
        "mean_pred": float(pred.mean()),
    }
    for k in hotspot_k:
        scores = hotspot_metrics(frame, pred_col, y_col, k)
        tag = f"{k * 100:g}pct"
        out[f"hit_rate@{tag}"] = scores["hit_rate"]
        out[f"pai@{tag}"] = scores["pai"]
        out[f"pei@{tag}"] = scores["pei"]
    return out
