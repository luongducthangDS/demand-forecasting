"""Metrics, baselines, models and rolling-origin backtest."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from forecast.features import FEATURES, H

TRAIN_START = "2015-01-01"  # first date with a valid lag_364 (panel starts 2014-01-01)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "WAPE": float(np.abs(e).sum() / np.abs(y).sum()),
        "RMSLE": float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))),
        "bias": float(e.sum() / y.sum()),  # >0 means over-forecast (excess stock), <0 under-forecast (stockouts)
    }


def window(df: pd.DataFrame, start: pd.Timestamp, horizon: int = H) -> pd.Series:
    return (df["date"] >= start) & (df["date"] < start + pd.Timedelta(days=horizon))


# ---------- baselines (use only sales before `start`) ----------

def moving_average(df: pd.DataFrame, start: pd.Timestamp, days: int = 28) -> np.ndarray:
    hist = df[(df["date"] < start) & (df["date"] >= start - pd.Timedelta(days=days))]
    ma = hist.groupby(["store_nbr", "family"], observed=True)["sales"].mean().rename("p")
    test = df.loc[window(df, start), ["store_nbr", "family"]]
    return test.join(ma, on=["store_nbr", "family"])["p"].to_numpy()


def seasonal_naive(df: pd.DataFrame, start: pd.Timestamp) -> np.ndarray:
    """Same weekday in the last observed week before the forecast origin."""
    last_week = df[(df["date"] < start) & (df["date"] >= start - pd.Timedelta(days=7))]
    ref = last_week.assign(dow=last_week["date"].dt.dayofweek).set_index(["store_nbr", "family", "dow"])["sales"]
    test = df.loc[window(df, start), ["store_nbr", "family", "date"]]
    keys = pd.MultiIndex.from_arrays([test["store_nbr"], test["family"], test["date"].dt.dayofweek])
    return ref.reindex(keys).to_numpy()


# ---------- gradient boosting ----------

LGB_DEFAULT = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "verbose": -1, "num_threads": 0, "seed": 42}


def fit_lgb(train: pd.DataFrame, params: dict, rounds: int, valid: pd.DataFrame | None = None):
    dtrain = lgb.Dataset(train[FEATURES], train["y"], categorical_feature="auto", free_raw_data=False)
    sets, cbs = [], []
    if valid is not None:
        sets = [lgb.Dataset(valid[FEATURES], valid["y"], reference=dtrain)]
        cbs = [lgb.early_stopping(50, verbose=False)]
    return lgb.train(params, dtrain, rounds, valid_sets=sets, callbacks=cbs)


def fit_xgb(train: pd.DataFrame, rounds: int = 600):
    model = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.05, max_depth=10, min_child_weight=50,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                             max_cat_to_onehot=1, n_jobs=-1, random_state=42)
    return model.fit(train[FEATURES], train["y"])


def postprocess(test: pd.DataFrame, log_pred: np.ndarray) -> np.ndarray:
    p = np.clip(np.expm1(log_pred), 0, None)
    # Series with zero sales for the 56 days before the origin (product not carried) -> forecast 0.
    p[test["zero_share_56"].to_numpy() == 1.0] = 0.0
    p[test["store_closed"].to_numpy() == 1] = 0.0
    return p


def split(feat: pd.DataFrame, start: pd.Timestamp):
    train = feat[(feat["date"] >= TRAIN_START) & (feat["date"] < start)]
    test = feat[window(feat, start)]
    return train, test


# ---------- persistence: serve whichever model wins the backtest ----------

def save_model(model, art) -> None:
    for f in ("model.txt", "model.json"):
        (art / f).unlink(missing_ok=True)
    if isinstance(model, lgb.Booster):
        model.save_model(str(art / "model.txt"))
    else:
        model.save_model(str(art / "model.json"))


def load_model(art):
    if (art / "model.txt").exists():
        return lgb.Booster(model_file=str(art / "model.txt"))
    model = xgb.XGBRegressor(enable_categorical=True)
    model.load_model(str(art / "model.json"))
    return model


def model_path(art):
    return art / "model.txt" if (art / "model.txt").exists() else art / "model.json"


def contributions(model, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP values from the libraries' native implementation (handles categorical splits)."""
    if isinstance(model, lgb.Booster):
        return model.predict(X, pred_contrib=True)[:, :-1]
    return model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
