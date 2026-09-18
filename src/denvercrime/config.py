"""Typed run configuration loaded from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


@dataclass(frozen=True)
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


@dataclass(frozen=True)
class DataConfig:
    feature_server: str
    hub_csv_url: str
    raw_dir: Path
    processed_dir: Path
    page_size: int
    start_date: str
    drop_recent_days: int
    csv_time_shift_hours: str | float
    bbox: BBox


@dataclass(frozen=True)
class PanelConfig:
    h3_resolution: int
    min_train_incidents: int


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    neighbor_ring: int


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    val_end: str
    min_history_weeks: int


@dataclass(frozen=True)
class ModelConfig:
    targets: tuple[str, ...]
    num_boost_round: int
    early_stopping_rounds: int
    lightgbm: dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    hotspot_k: tuple[float, ...]
    baselines: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    data: DataConfig
    groups: dict[str, tuple[str, ...]]
    panel: PanelConfig
    features: FeatureConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    runs_dir: Path
    source: dict[str, Any]

    @property
    def category_to_group(self) -> dict[str, str]:
        return {cat: group for group, cats in self.groups.items() for cat in cats}

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path) -> Config:
        d = raw["data"]
        cfg = cls(
            data=DataConfig(
                feature_server=d["feature_server"],
                hub_csv_url=d["hub_csv_url"],
                raw_dir=(root / d["raw_dir"]).resolve(),
                processed_dir=(root / d["processed_dir"]).resolve(),
                page_size=int(d["page_size"]),
                start_date=str(d["start_date"]),
                drop_recent_days=int(d["drop_recent_days"]),
                csv_time_shift_hours=d["csv_time_shift_hours"],
                bbox=BBox(**{k: float(v) for k, v in d["bbox"].items()}),
            ),
            groups={g: tuple(cats) for g, cats in raw["groups"].items()},
            panel=PanelConfig(**raw["panel"]),
            features=FeatureConfig(
                horizon=int(raw["features"]["horizon"]),
                lags=tuple(sorted(int(x) for x in raw["features"]["lags"])),
                rolling_windows=tuple(sorted(int(x) for x in raw["features"]["rolling_windows"])),
                neighbor_ring=int(raw["features"]["neighbor_ring"]),
            ),
            split=SplitConfig(**raw["split"]),
            model=ModelConfig(
                targets=tuple(raw["model"]["targets"]),
                num_boost_round=int(raw["model"]["num_boost_round"]),
                early_stopping_rounds=int(raw["model"]["early_stopping_rounds"]),
                lightgbm=dict(raw["model"]["lightgbm"]),
            ),
            evaluation=EvaluationConfig(
                hotspot_k=tuple(float(k) for k in raw["evaluation"]["hotspot_k"]),
                baselines=tuple(raw["evaluation"]["baselines"]),
            ),
            runs_dir=(root / raw["output"]["runs_dir"]).resolve(),
            source=raw,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        seen: dict[str, str] = {}
        for group, cats in self.groups.items():
            for cat in cats:
                if cat in seen:
                    raise ValueError(f"category {cat!r} is in both {seen[cat]!r} and {group!r}")
                seen[cat] = group
        unknown = set(self.model.targets) - set(self.groups)
        if unknown:
            raise ValueError(f"model.targets not defined in [groups]: {sorted(unknown)}")
        h = self.features.horizon
        if h < 1:
            raise ValueError("features.horizon must be >= 1")
        if any(lag < h for lag in self.features.lags):
            raise ValueError(f"every lag must be >= horizon ({h}); got {self.features.lags}")
        if self.split.train_end >= self.split.val_end:
            raise ValueError("split.train_end must be before split.val_end")
        if not all(0 < k < 1 for k in self.evaluation.hotspot_k):
            raise ValueError("evaluation.hotspot_k values must be fractions in (0, 1)")


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Load a TOML config. Relative paths resolve against `root` (default: the config's parent's parent)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    root = Path(root) if root else path.resolve().parent.parent
    return Config.from_dict(raw, root)
