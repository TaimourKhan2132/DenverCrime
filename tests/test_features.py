import h3
import numpy as np
import pandas as pd
import pytest

from denvercrime.config import FeatureConfig
from denvercrime.features.build import (
    _ewm,
    build_features,
    check_panel,
    feature_group,
    holiday_counts,
    select_features,
)
from denvercrime.features.spatial import cell_table, neighbor_matrix

CELL_A = h3.latlng_to_cell(39.7392, -104.9903, 8)
CELL_B = sorted(h3.grid_ring(CELL_A, 1))[0]
CELL_R2 = sorted(set(h3.grid_ring(CELL_A, 2)) - set(h3.grid_disk(CELL_B, 1)))[0]  # 2 from A, not next to B
CELL_C = sorted(h3.grid_ring(CELL_A, 5))[0]  # far from everything
CELLS = [CELL_A, CELL_B, CELL_R2, CELL_C]
WEEKS = pd.date_range("2023-01-02", periods=120, freq="7D")
GROUPS = ["property", "person"]


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([CELLS, WEEKS], names=["cell", "week"])
    return pd.DataFrame({"y_property": rng.poisson(3, len(idx)), "y_person": rng.poisson(1, len(idx))},
                        index=idx).reset_index()


def _build(panel, horizon=1, lags=(1, 2, 52), full=True):
    fcfg = FeatureConfig(horizon=horizon, lags=lags, rolling_windows=(4, 13), neighbor_ring=1,
                         ewm_halflives=(4.0, 13.0) if full else (), trend_ratios=full,
                         outer_ring=2 if full else 0, holidays=full)
    outer = neighbor_matrix(CELLS, 2, annulus=True) if full else None
    return build_features(panel, GROUPS, fcfg, neighbor_matrix(CELLS, 1), cell_table(CELLS), outer)


def _series(frame, cell, col):
    return frame.loc[frame["cell"] == cell, col].to_numpy()


def test_lags_and_rolling_means():
    frame, _ = _build(_panel())
    y = _series(frame, CELL_A, "y_property").astype(float)
    np.testing.assert_array_equal(_series(frame, CELL_A, "property_lag1")[1:], y[:-1])
    assert np.isnan(_series(frame, CELL_A, "property_lag1")[0])
    mean4 = _series(frame, CELL_A, "property_mean4")
    assert mean4[10] == pytest.approx(y[6:10].mean())  # weeks t-4..t-1
    assert np.isnan(mean4[3]) and not np.isnan(mean4[4])
    np.testing.assert_array_equal(_series(frame, CELL_A, "property_lag52")[52:], y[:-52])


def test_neighbor_outer_ring_and_city_features():
    frame, _ = _build(_panel())
    y_b = _series(frame, CELL_B, "y_property").astype(float)
    np.testing.assert_allclose(_series(frame, CELL_A, "property_nbr_lag1")[1:], y_b[:-1])  # B is A's only ring-1 neighbour
    assert np.allclose(np.nan_to_num(_series(frame, CELL_C, "property_nbr_lag1")), 0)
    y_r2 = _series(frame, CELL_R2, "y_property").astype(float)
    np.testing.assert_allclose(_series(frame, CELL_A, "property_nbr2_mean4")[10], y_r2[6:10].mean())
    city = frame.groupby("week", sort=False)["y_property"].sum().to_numpy(dtype=float)
    np.testing.assert_allclose(_series(frame, CELL_C, "property_city_lag1")[1:], city[:-1])


def test_ewm_matches_pandas():
    x = np.array([[np.nan, 1.0, 3.0, 0.0, 2.0, 5.0]])
    expected = pd.Series(x[0]).ewm(halflife=2, adjust=False, ignore_na=True).mean().to_numpy()
    np.testing.assert_allclose(_ewm(x, 2.0)[0], expected)


def test_trend_ratio_and_holidays():
    frame, _ = _build(_panel())
    y = _series(frame, CELL_A, "y_property").astype(float)
    trend = _series(frame, CELL_A, "property_trend4")
    t = 55
    assert trend[t] == pytest.approx((y[t - 4 : t].mean() + 0.1) / (y[t - 52 : t].mean() + 0.1))
    assert np.isnan(trend[:52]).all()  # needs a full 52-week history
    counts = holiday_counts(pd.DatetimeIndex(["2023-07-03", "2023-07-10", "2023-12-25"]))
    assert counts.tolist() == [1, 0, 1]  # Independence Day, nothing, Christmas


@pytest.mark.parametrize("j", [30, 80])  # 80: after 52 weeks, so long-history features are populated
@pytest.mark.parametrize("horizon,lags", [(1, (1, 2, 52)), (2, (2, 3, 52))])
def test_no_feature_sees_the_current_or_future_weeks(horizon, lags, j):
    """Changing counts from week j on must not change any feature before week j + horizon."""
    panel = _panel()
    base, features = _build(panel, horizon=horizon, lags=lags)
    changed = panel.copy()
    future = changed["week"] >= WEEKS[j]
    changed.loc[future, ["y_property", "y_person"]] += 50
    perturbed, _ = _build(changed, horizon=horizon, lags=lags)

    visible = base["week"] < WEEKS[j] + pd.Timedelta(weeks=horizon)
    if j >= 60:  # by then every feature, including 52-week ones, has values to compare
        assert base.loc[visible, features].notna().any().all()
    pd.testing.assert_frame_equal(base.loc[visible, features], perturbed.loc[visible, features])
    # ...and the perturbation does reach the features once it is allowed to.
    later = base["week"] == WEEKS[j] + pd.Timedelta(weeks=horizon)
    assert not base.loc[later, features].equals(perturbed.loc[later, features])


def test_feature_groups_and_selection():
    _, features = _build(_panel())
    assert feature_group("person_nbr2_mean13", GROUPS) == ("person", "outer_spatial")
    assert feature_group("property_ewm13", GROUPS) == ("property", "ewm")
    assert feature_group("holidays", GROUPS) == (None, "holidays")
    own = select_features(features, "property", GROUPS, ("cross_group",))
    assert not any(f.startswith("person_") for f in own) and "property_lag1" in own
    no_space = select_features(features, "property", GROUPS, ("spatial", "outer_spatial"))
    assert not any("nbr" in f for f in no_space)
    with pytest.raises(ValueError):
        select_features(features, "property", GROUPS, ("colour",))


def test_v1_features_are_unchanged_when_extras_are_off():
    _, features = _build(_panel(), full=False)
    assert not any(k in f for f in features for k in ("ewm", "trend", "nbr2", "holidays"))


def test_incomplete_or_unsorted_panel_rejected():
    panel = _panel()
    with pytest.raises(ValueError):
        check_panel(panel.drop(index=5))
    with pytest.raises(ValueError):
        check_panel(panel.sample(frac=1, random_state=0))
