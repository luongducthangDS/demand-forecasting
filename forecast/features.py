"""Leakage-safe features for a direct H-day-ahead forecast.

Rule: a row dated t may only use sales observed at t - H or earlier. With H = 14 one model predicts
every day of a 14-day window from the same forecast origin, without feeding predictions back.
Known-in-advance inputs (calendar, holidays, planned promotions) are used at t.
"""
import numpy as np
import pandas as pd

H = 14
CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "cluster"]
SALES_LAGS = [14, 21, 28, 35, 42, 364]
ROLL_WINDOWS = [7, 14, 28, 56, 112]


def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Panel is a complete grid sorted by (store, family, date), so pivot <-> ravel round-trips exactly.
    return df.pivot_table(index="date", columns=["store_nbr", "family"], values=col, dropna=False, sort=True)


def _long(wide: pd.DataFrame) -> np.ndarray:
    return wide.to_numpy(dtype="float32").T.ravel()


def make_features(panel: pd.DataFrame, horizon: int = H) -> pd.DataFrame:
    df = panel.copy()
    df["y"] = np.log1p(df["sales"])
    w = _wide(df, "y")
    # ponytail: closed-store days (25 Dec) are 0 by construction; masking them out of rolling stats is a later refinement.

    for lag in SALES_LAGS:
        assert lag >= horizon, "lag shorter than horizon leaks the future"
        df[f"lag_{lag}"] = _long(w.shift(lag))

    base = w.shift(horizon)
    for win in ROLL_WINDOWS:
        df[f"roll_mean_{win}"] = _long(base.rolling(win, min_periods=1).mean())
    df["roll_std_28"] = _long(base.rolling(28, min_periods=2).std())
    df["roll_max_28"] = _long(base.rolling(28, min_periods=1).max())
    # Same weekday over the last 4 available weeks: lags 14, 21, 28, 35 all share t's weekday.
    df["dow_mean_4w"] = df[[f"lag_{k}" for k in (14, 21, 28, 35)]].mean(axis=1)
    df["zero_share_56"] = _long((base == 0).rolling(56, min_periods=1).mean())

    promo = _wide(df, "onpromotion")
    df["promo_log"] = np.log1p(df["onpromotion"])
    df["promo_vs_28d"] = df["onpromotion"] - _long(promo.shift(horizon).rolling(28, min_periods=1).mean())

    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype("int8")
    df["dom"] = d.dt.day.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["doy"] = d.dt.dayofyear.astype("int16")
    df["payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype("int8")
    # Oil price is published daily, but the future price is unknown at forecast time: use t - H.
    df["oil_lag"] = _long(_wide(df, "oil").shift(horizon))

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


FEATURES = (
    CATEGORICAL
    + [f"lag_{k}" for k in SALES_LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + ["roll_std_28", "roll_max_28", "dow_mean_4w", "zero_share_56", "onpromotion", "promo_log", "promo_vs_28d"]
    + ["dow", "dom", "month", "doy", "payday", "oil_lag", "store_closed"]
    + ["hol_national", "hol_regional", "hol_local", "event_national", "work_day"]
)
