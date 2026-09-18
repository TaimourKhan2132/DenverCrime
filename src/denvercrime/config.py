"""Typed run configuration loaded from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
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
class CleaningConfig:
    exclude_neighborhoods: tuple[str, ...] = ()
    exclude_addresses: tuple[str, ...] = ()
    long_window_days: int = 7


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
    ewm_halflives: tuple[float, ...] = ()
    trend_ratios: bool = False
    outer_ring: int = 0
    holidays: bool = False
    drop_groups: tuple[str, ...] = ()
    target_drop_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def dropped_for(self, target: str) -> tuple[str, ...]:
        """Feature groups removed for one target: the shared list plus the target's own."""
        return tuple(dict.fromkeys((*self.drop_groups, *self.target_drop_groups.get(target, ()))))


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    val_end: str
    min_history_weeks: int
    test_folds: int = 1


@dataclass(frozen=True)
class ModelConfig:
    targets: tuple[str, ...]
    num_boost_round: int
    early_stopping_rounds: int
    lightgbm: dict[str, Any]
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def params(self, target: str) -> dict[str, Any]:
        """Shared LightGBM parameters with the target's overrides applied."""
        return {**self.lightgbm, **self.overrides.get(target, {})}


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 40
    seed: int = 42


@dataclass(frozen=True)
class EvaluationConfig:
    hotspot_k: tuple[float, ...]
    baselines: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    data: DataConfig
    cleaning: CleaningConfig
    groups: dict[str, tuple[str, ...]]
    panel: PanelConfig
    features: FeatureConfig
    split: SplitConfig
    model: ModelConfig
    tuning: TuningConfig
    evaluation: EvaluationConfig
    runs_dir: Path
    source: dict[str, Any]

    @property
    def category_to_group(self) -> dict[str, str]:
        return {cat: group for group, cats in self.groups.items() for cat in cats}

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path) -> Config:
        d = raw["data"]
        c = raw.get("cleaning", {})
        f = raw["features"]
        m = raw["model"]
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
            cleaning=CleaningConfig(
                exclude_neighborhoods=tuple(c.get("exclude_neighborhoods", ())),
                exclude_addresses=tuple(normalize_address(e["address"]) for e in c.get("exclude_addresses", ())),
                long_window_days=int(c.get("long_window_days", 7)),
            ),
            groups={g: tuple(cats) for g, cats in raw["groups"].items()},
            panel=PanelConfig(**raw["panel"]),
            features=FeatureConfig(
                horizon=int(f["horizon"]),
                lags=tuple(sorted(int(x) for x in f["lags"])),
                rolling_windows=tuple(sorted(int(x) for x in f["rolling_windows"])),
                neighbor_ring=int(f["neighbor_ring"]),
                ewm_halflives=tuple(float(x) for x in f.get("ewm_halflives", ())),
                trend_ratios=bool(f.get("trend_ratios", False)),
                outer_ring=int(f.get("outer_ring", 0)),
                holidays=bool(f.get("holidays", False)),
                drop_groups=tuple(f.get("drop_groups", ())),
                target_drop_groups={t: tuple(v) for t, v in f.get("target_drop_groups", {}).items()},
            ),
            split=SplitConfig(**raw["split"]),
            model=ModelConfig(
                targets=tuple(m["targets"]),
                num_boost_round=int(m["num_boost_round"]),
                early_stopping_rounds=int(m["early_stopping_rounds"]),
                lightgbm=dict(m["lightgbm"]),
                overrides={t: dict(v) for t, v in m.get("overrides", {}).items()},
            ),
            tuning=TuningConfig(**raw.get("tuning", {})),
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
        stray = set(self.model.overrides) - set(self.model.targets)
        if stray:
            raise ValueError(f"[model.overrides] for targets that are not modelled: {sorted(stray)}")
        stray = set(self.features.target_drop_groups) - set(self.model.targets)
        if stray:
            raise ValueError(f"[features.target_drop_groups] for targets that are not modelled: {sorted(stray)}")
        h = self.features.horizon
        if h < 1:
            raise ValueError("features.horizon must be >= 1")
        if any(lag < h for lag in self.features.lags):
            raise ValueError(f"every lag must be >= horizon ({h}); got {self.features.lags}")
        if self.features.outer_ring and self.features.outer_ring <= self.features.neighbor_ring:
            raise ValueError("features.outer_ring must be larger than neighbor_ring (or 0 to disable)")
        if self.split.train_end >= self.split.val_end:
            raise ValueError("split.train_end must be before split.val_end")
        if self.split.test_folds < 1:
            raise ValueError("split.test_folds must be >= 1")
        if not all(0 < k < 1 for k in self.evaluation.hotspot_k):
            raise ValueError("evaluation.hotspot_k values must be fractions in (0, 1)")


def normalize_address(address: str) -> str:
    return " ".join(str(address).upper().split())


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge tables. Non-table values (including lists) in `override` win, and an
    empty table replaces the base table, so `overrides = {}` clears inherited overrides."""
    out = dict(base)
    for key, value in override.items():
        mergeable = isinstance(value, dict) and value and isinstance(out.get(key), dict)
        out[key] = deep_merge(out[key], value) if mergeable else value
    return out


def read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file, applying `extends = "other.toml"` (relative to this file) if present."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    parent = raw.pop("extends", None)
    return deep_merge(read_toml(path.parent / parent), raw) if parent else raw


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Load a TOML config. Relative paths resolve against `root` (default: the config's parent's parent)."""
    path = Path(path) if path else DEFAULT_CONFIG
    raw = read_toml(path)
    root = Path(root) if root else path.resolve().parent.parent
    return Config.from_dict(raw, root)
