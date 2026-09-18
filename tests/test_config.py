import pytest

from denvercrime.config import load_config
from tests.conftest import CATEGORIES, make_config

# OFFENSE_CATEGORY_ID values present in the live layer on 2026-09-18.
LIVE_CATEGORIES = set(CATEGORIES)


def test_default_config_loads_and_maps_every_live_category():
    cfg = load_config()
    assert set(cfg.category_to_group) == LIVE_CATEGORIES
    assert set(cfg.model.targets) <= set(cfg.groups)
    assert cfg.data.raw_dir.name == "raw"


def test_duplicate_category_rejected(tmp_path):
    with pytest.raises(ValueError, match="in both"):
        make_config(tmp_path, groups__other=["all-other-crimes", "larceny"])


def test_lag_shorter_than_horizon_rejected(tmp_path):
    with pytest.raises(ValueError, match="horizon"):
        make_config(tmp_path, features__horizon=2)  # default lags include 1


def test_unknown_target_rejected(tmp_path):
    with pytest.raises(ValueError, match="targets"):
        make_config(tmp_path, model__targets=["property", "cyber"])
