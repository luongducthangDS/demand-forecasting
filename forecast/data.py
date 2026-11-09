"""Load the Favorita Store Sales files into one daily panel: (date, store, family) -> sales + exogenous columns."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
# Mirror of the Kaggle "Store Sales - Time Series Forecasting" files (Corporación Favorita, Ecuador).
MIRROR = "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/"
FILES = ["train.csv", "test.csv", "stores.csv", "oil.csv", "holidays_events.csv"]


def download(raw: Path = RAW) -> None:
    import urllib.request
    raw.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (raw / f).exists():
            print("downloading", f)
            urllib.request.urlretrieve(MIRROR + f, raw / f)


def _holiday_flags(panel: pd.DataFrame, hol: pd.DataFrame) -> pd.DataFrame:
    # A "transferred" holiday is a normal working day; the celebrated day is the row typed "Transfer".
    hol = hol[~hol["transferred"]].copy()
    hol["date"] = pd.to_datetime(hol["date"])
    off = hol[hol["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"])]

    nat = off[off["locale"] == "National"]["date"].unique()
    panel["hol_national"] = panel["date"].isin(nat).astype("int8")
    panel["event_national"] = panel["date"].isin(hol[hol["type"] == "Event"]["date"].unique()).astype("int8")
    panel["work_day"] = panel["date"].isin(hol[hol["type"] == "Work Day"]["date"].unique()).astype("int8")

    for locale, col in [("Regional", "state"), ("Local", "city")]:
        keys = off[off["locale"] == locale][["date", "locale_name"]].drop_duplicates()
        keys = keys.rename(columns={"locale_name": col}).assign(**{f"hol_{locale.lower()}": 1})
        panel = panel.merge(keys, on=["date", col], how="left")
        panel[f"hol_{locale.lower()}"] = panel[f"hol_{locale.lower()}"].fillna(0).astype("int8")
    return panel


def build_panel(raw: Path = RAW, start: str = "2014-01-01") -> pd.DataFrame:
    """Train + test rows on a complete date grid. Test rows (future) have sales = NaN and is_future = 1."""
    train = pd.read_csv(raw / "train.csv", parse_dates=["date"]).drop(columns="id")
    test = pd.read_csv(raw / "test.csv", parse_dates=["date"]).drop(columns="id")
    df = pd.concat([train, test.assign(sales=np.nan)], ignore_index=True)
    df = df[df["date"] >= start]

    # The source has no rows for 25 Dec (stores closed). Reindex so lags are true calendar lags.
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product(
        [dates, sorted(df["store_nbr"].unique()), sorted(df["family"].unique())],
        names=["date", "store_nbr", "family"],
    )
    df = df.set_index(["date", "store_nbr", "family"]).reindex(grid).reset_index()
    last_train = train["date"].max()
    df["is_future"] = (df["date"] > last_train).astype("int8")
    df.loc[df["is_future"] == 0, "sales"] = df.loc[df["is_future"] == 0, "sales"].fillna(0.0)
    df["onpromotion"] = df["onpromotion"].fillna(0).astype("float32")
    df["store_closed"] = ((df["date"].dt.month == 12) & (df["date"].dt.day == 25)).astype("int8")

    stores = pd.read_csv(raw / "stores.csv")
    df = df.merge(stores, on="store_nbr", how="left")
    df = _holiday_flags(df, pd.read_csv(raw / "holidays_events.csv"))

    oil = pd.read_csv(raw / "oil.csv", parse_dates=["date"]).set_index("date")["dcoilwtico"]
    oil = oil.reindex(pd.date_range(oil.index.min(), df["date"].max(), freq="D")).ffill().bfill()
    df["oil"] = df["date"].map(oil).astype("float32")

    df["sales"] = df["sales"].astype("float32")
    return df.sort_values(["store_nbr", "family", "date"], ignore_index=True)


if __name__ == "__main__":
    download()
