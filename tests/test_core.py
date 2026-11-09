import numpy as np
import pandas as pd
import pytest

from forecast import backtest as bt
from forecast.features import FEATURES, H, make_features
from forecast.replenish import replenishment


def toy_panel(days: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-01-01", periods=days, freq="D")
    idx = pd.MultiIndex.from_product([[1, 2], ["A", "B"], dates], names=["store_nbr", "family", "date"])
    df = idx.to_frame(index=False)
    df["sales"] = rng.poisson(20, len(df)).astype("float32")
    df["onpromotion"] = rng.integers(0, 3, len(df)).astype("float32")
    df["oil"] = rng.normal(50, 5, len(df)).astype("float32")
    for c in ["city", "state", "type", "cluster"]:
        df[c] = df["store_nbr"].astype(str)
    for c in ["store_closed", "hol_national", "hol_regional", "hol_local", "event_national", "work_day"]:
        df[c] = 0
    return df


def test_features_only_see_sales_at_least_H_days_old():
    panel = toy_panel()
    cut = pd.Timestamp("2017-01-01")
    changed = panel.copy()
    changed.loc[changed["date"] >= cut, ["sales", "oil"]] *= 7  # rewrite the "future"
    a, b = make_features(panel)[FEATURES], make_features(changed)[FEATURES]
    safe = panel["date"] < cut + pd.Timedelta(days=H)
    pd.testing.assert_frame_equal(a[safe], b[safe])
    assert not a[~safe].equals(b[~safe]), "perturbation should reach rows >= cut + H"


def test_baselines_ignore_data_from_the_forecast_window():
    panel = toy_panel()
    start = pd.Timestamp("2017-01-01")
    changed = panel.copy()
    changed.loc[changed["date"] >= start, "sales"] = 1e6
    for f in (bt.moving_average, bt.seasonal_naive):
        np.testing.assert_array_equal(f(panel, start), f(changed, start))


def test_split_has_no_overlap():
    feat = make_features(toy_panel())
    start = pd.Timestamp("2017-01-01")
    train, test = bt.split(feat, start)
    assert train["date"].max() < start <= test["date"].min()
    assert test["date"].nunique() == H


def test_metrics():
    m = bt.metrics(np.array([10, 0, 30]), np.array([12, 1, 27]))
    assert m["MAE"] == pytest.approx(2.0)
    assert m["WAPE"] == pytest.approx(6 / 40)
    assert m["bias"] == pytest.approx(0.0)


def test_replenishment():
    flat = [10.0] * 14
    r = replenishment(flat, resid_std=0.0, on_hand=50, lead_time_days=2, review_days=7)
    assert r["order_qty"] == 40 and r["days_of_cover"] == 5 and not r["at_risk"]
    r = replenishment(flat, resid_std=3.0, on_hand=5)
    assert r["at_risk"] and r["stockout_risk_before_delivery"] > 0.99 and r["safety_stock"] > 0
    assert replenishment(flat, resid_std=3.0, on_hand=500)["order_qty"] == 0
    with pytest.raises(ValueError):
        replenishment(flat, 1.0, 10, lead_time_days=10, review_days=7)
