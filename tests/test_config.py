import pytest

from denvercrime.config import DEFAULT_CONFIG, load_config
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


def test_overrides_merge_over_shared_params(tmp_path):
    cfg = make_config(tmp_path, model__overrides={"person": {"num_leaves": 7, "objective": "tweedie"}})
    assert cfg.model.params("person")["num_leaves"] == 7
    assert cfg.model.params("person")["objective"] == "tweedie"
    assert cfg.model.params("property")["objective"] == "poisson"
    with pytest.raises(ValueError, match="overrides"):
        make_config(tmp_path, model__overrides={"society": {"num_leaves": 7}})


def test_outer_ring_must_be_wider(tmp_path):
    with pytest.raises(ValueError, match="outer_ring"):
        make_config(tmp_path, features__outer_ring=1)


def test_comparison_configs_extend_the_default():
    root = DEFAULT_CONFIG.parent
    v1 = load_config(root / "compare_v1_model.toml")
    assert v1.features.ewm_halflives == () and v1.features.dropped_for("property") == ()
    assert v1.model.overrides == {} and v1.cleaning.exclude_neighborhoods == ("dia",)
    assert v1.data.processed_dir.name == "compare_v1_model"
    raw = load_config(root / "compare_uncleaned.toml")
    assert raw.cleaning.exclude_neighborhoods == () and raw.cleaning.exclude_addresses == ()
    assert raw.features == load_config().features


def test_per_target_feature_drops(tmp_path):
    cfg = make_config(tmp_path, features__drop_groups=["trend"],
                      features__target_drop_groups={"property": ["cross_group", "trend"]})
    assert cfg.features.dropped_for("property") == ("trend", "cross_group")
    assert cfg.features.dropped_for("person") == ("trend",)


def test_excluded_addresses_are_normalised():
    cfg = load_config()
    assert "490 W COLFAX AVE" in cfg.cleaning.exclude_addresses
    assert "dia" in cfg.cleaning.exclude_neighborhoods
