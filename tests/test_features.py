import h3
import numpy as np
import pandas as pd
import pytest

from denvercrime.config import FeatureConfig
from denvercrime.features.build import build_features, check_panel
from denvercrime.features.spatial import cell_table, neighbor_matrix

CELL_A = h3.latlng_to_cell(39.7392, -104.9903, 8)
CELL_B = sorted(h3.grid_ring(CELL_A, 1))[0]
CELL_C = sorted(h3.grid_ring(CELL_A, 3))[0]  # far from both
CELLS = [CELL_A, CELL_B, CELL_C]
WEEKS = pd.date_range("2023-01-02", periods=60, freq="7D")


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([CELLS, WEEKS], names=["cell", "week"])
    return pd.DataFrame({"y_property": rng.poisson(3, len(idx)), "y_person": rng.poisson(1, len(idx))},
                        index=idx).reset_index()


def _build(panel, horizon=1, lags=(1, 2, 52), windows=(4, 13)):
    fcfg = FeatureConfig(horizon=horizon, lags=lags, rolling_windows=windows, neighbor_ring=1)
    return build_features(panel, ["property", "person"], fcfg, neighbor_matrix(CELLS, 1), cell_table(CELLS))


def _series(frame, cell, col):
    return frame.loc[frame["cell"] == cell, col].to_numpy()


def test_lags_and_rolling_means():
    panel = _panel()
    frame, _ = _build(panel)
    y = _series(frame, CELL_A, "y_property").astype(float)
    np.testing.assert_array_equal(_series(frame, CELL_A, "property_lag1")[1:], y[:-1])
    assert np.isnan(_series(frame, CELL_A, "property_lag1")[0])
    mean4 = _series(frame, CELL_A, "property_mean4")
    assert mean4[10] == pytest.approx(y[6:10].mean())  # weeks t-4..t-1
    assert np.isnan(mean4[3]) and not np.isnan(mean4[4])
    np.testing.assert_array_equal(_series(frame, CELL_A, "property_lag52")[52:], y[:-52])


def test_neighbor_and_city_features():
    frame, _ = _build(_panel())
    y_b = _series(frame, CELL_B, "y_property").astype(float)
    nbr_a = _series(frame, CELL_A, "property_nbr_lag1")
    np.testing.assert_allclose(nbr_a[1:], y_b[:-1])  # B is A's only neighbour in the set
    assert np.allclose(np.nan_to_num(_series(frame, CELL_C, "property_nbr_lag1")), 0)
    city = frame.groupby("week")["y_property"].sum().to_numpy(dtype=float)
    np.testing.assert_allclose(_series(frame, CELL_C, "property_city_lag1")[1:], city[:-1])


@pytest.mark.parametrize("horizon,lags", [(1, (1, 2, 52)), (2, (2, 3, 52))])
def test_no_feature_sees_the_current_or_future_weeks(horizon, lags):
    """Changing counts at week j must not change any feature before week j + horizon."""
    panel = _panel()
    base, features = _build(panel, horizon=horizon, lags=lags)
    j = 30
    changed = panel.copy()
    future = changed["week"] >= WEEKS[j]
    changed.loc[future, ["y_property", "y_person"]] += 50
    perturbed, _ = _build(changed, horizon=horizon, lags=lags)

    visible = base["week"] < WEEKS[j] + pd.Timedelta(weeks=horizon)
    pd.testing.assert_frame_equal(base.loc[visible, features], perturbed.loc[visible, features])
    # ...and the perturbation does reach the features once it is allowed to.
    later = base["week"] == WEEKS[j] + pd.Timedelta(weeks=horizon)
    assert not base.loc[later, features].equals(perturbed.loc[later, features])


def test_incomplete_or_unsorted_panel_rejected():
    panel = _panel()
    with pytest.raises(ValueError):
        check_panel(panel.drop(index=5))
    with pytest.raises(ValueError):
        check_panel(panel.sample(frac=1, random_state=0))
