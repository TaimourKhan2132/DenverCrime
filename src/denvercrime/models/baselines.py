"""Naive forecasters the model has to beat. All of them reuse the leakage-safe features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from denvercrime.features.panel import count_column


def _column(frame: pd.DataFrame, name: str, baseline: str) -> pd.Series:
    if name not in frame:
        raise KeyError(f"baseline {baseline!r} needs feature {name!r}; add it to the [features] config")
    return frame[name]


def baseline_prediction(
    name: str,
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    history: pd.DataFrame | None = None,
) -> np.ndarray:
    """Predict the target count for every row of `frame`.

    history: rows used for `historical_mean` (the training data); ignored otherwise.
    Missing history (start of the panel) falls back to 0.
    """
    if name == "last_week":
        pred = _column(frame, f"{target}_lag{horizon}", name)
    elif name.startswith("moving_average_"):
        window = int(name.rsplit("_", 1)[1])
        pred = _column(frame, f"{target}_mean{window}", name)
    elif name == "seasonal_naive":
        pred = _column(frame, f"{target}_lag52", name)
    elif name == "historical_mean":
        if history is None:
            raise ValueError("historical_mean needs the training rows as `history`")
        means = history.groupby("cell")[count_column(target)].mean()
        pred = frame["cell"].map(means)
    else:
        raise ValueError(f"unknown baseline {name!r}")
    return pred.fillna(0.0).to_numpy(dtype=np.float64)
