"""Inference API + dashboard.

    uvicorn forecast.api:app --port 8000
DATABASE_URL defaults to SQLite in artifacts/; docker-compose points it at PostgreSQL.
"""
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text

from forecast.backtest import load_model, metrics, model_path, postprocess
from forecast.features import CATEGORICAL, FEATURES, H
from forecast.replenish import replenishment

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
engine = create_engine(os.getenv("DATABASE_URL", f"sqlite:///{(ART / 'forecast.db').as_posix()}"))
S: dict = {}  # loaded model + frames
latency: deque = deque(maxlen=5000)  # ponytail: in-process metrics, export to Prometheus when there are >1 replicas
counts = {"requests": 0, "errors": 0}


def publish_artifacts() -> None:
    """Copy training artifacts into the DB the dashboard reads; feedback tables are never overwritten."""
    for name in ("history", "backtest_predictions", "series_stats"):
        pd.read_parquet(ART / f"{name}.parquet").astype({"family": str, "store_nbr": int}).to_sql(
            name, engine, if_exists="replace", index=False, chunksize=10_000)
    with engine.begin() as c:
        c.execute(text("CREATE TABLE IF NOT EXISTS prediction_log (request_id VARCHAR(36), created_at VARCHAR(32), "
                       "model_version VARCHAR(64), date DATE, store_nbr INTEGER, family VARCHAR(64), forecast FLOAT)"))
        c.execute(text("CREATE TABLE IF NOT EXISTS actuals (date DATE, store_nbr INTEGER, family VARCHAR(64), "
                       "sales FLOAT, received_at VARCHAR(32))"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    S["model"] = load_model(ART)
    S["version"] = time.strftime("%Y%m%d-%H%M%S", time.localtime(model_path(ART).stat().st_mtime))
    fut = pd.read_parquet(ART / "future_features.parquet")
    # Parquet drops category dtype on int columns (store_nbr, cluster). The future window holds every series,
    # so sorted uniques reproduce the training categories (pandas builds them the same way).
    for c in CATEGORICAL:
        if not isinstance(fut[c].dtype, pd.CategoricalDtype):
            fut[c] = pd.Categorical(fut[c], categories=sorted(fut[c].unique()))
    S["future"] = fut
    S["stats"] = pd.read_parquet(ART / "series_stats.parquet").set_index(["store_nbr", "family"])
    S["report"] = json.loads((ART / "report.json").read_text())
    publish_artifacts()
    yield


app = FastAPI(title="Demand Forecasting API", lifespan=lifespan)


@app.middleware("http")
async def track(request: Request, call_next):
    t0 = time.perf_counter()
    counts["requests"] += 1
    try:
        resp = await call_next(request)
    except Exception:
        counts["errors"] += 1
        raise
    if resp.status_code >= 500:
        counts["errors"] += 1
    latency.append((request.url.path, (time.perf_counter() - t0) * 1000))
    return resp


def _rows(store_nbr: int, family: str) -> pd.DataFrame:
    f = S["future"]
    rows = f[(f["store_nbr"] == store_nbr) & (f["family"] == family)].sort_values("date")
    if rows.empty:
        raise HTTPException(404, f"unknown series store={store_nbr} family={family}")
    return rows


def predict(store_nbr: int, family: str, log: bool = True) -> pd.DataFrame:
    rows = _rows(store_nbr, family)
    out = rows[["date"]].assign(forecast=postprocess(rows, S["model"].predict(rows[FEATURES])))
    if log:
        now, rid = datetime.now(timezone.utc).isoformat(), str(uuid.uuid4())
        out.assign(date=out["date"].dt.date, request_id=rid, created_at=now, model_version=S["version"], store_nbr=store_nbr,
                   family=family)[["request_id", "created_at", "model_version", "date", "store_nbr", "family",
                                   "forecast"]].to_sql("prediction_log", engine, if_exists="append", index=False)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_version": S["version"], "model": S["report"]["selected_model"],
            "forecast_window": [str(S["future"]["date"].min().date()), str(S["future"]["date"].max().date())]}


@app.get("/series")
def series():
    f = S["future"]
    return {"stores": sorted(int(s) for s in f["store_nbr"].unique()),
            "families": sorted(str(x) for x in f["family"].unique())}


@app.get("/forecast")
def forecast(store_nbr: int, family: str):
    out = predict(store_nbr, family)
    return {"store_nbr": store_nbr, "family": family, "model_version": S["version"], "horizon_days": H,
            "forecast": [{"date": str(d.date()), "sales": round(float(v), 3)} for d, v in zip(out["date"], out["forecast"])]}


@app.get("/series/{store_nbr}/{family}")
def series_detail(store_nbr: int, family: str):
    """History, backtest (actual vs forecast vs MA28) and the live forecast for one series, for the dashboard."""
    q = {"s": store_nbr, "f": family}
    hist = pd.read_sql(text("SELECT date, sales, onpromotion FROM history WHERE store_nbr=:s AND family=:f ORDER BY date"), engine, params=q)
    bt = pd.read_sql(text("SELECT date, fold, sales, forecast, \"MovingAverage28\" AS ma28 FROM backtest_predictions "
                          "WHERE store_nbr=:s AND family=:f ORDER BY date"), engine, params=q)
    fc = predict(store_nbr, family, log=False)
    sm = metrics(bt["sales"], bt["forecast"]) if bt["sales"].sum() > 0 else None
    to_list = lambda d: json.loads(d.assign(date=pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")).to_json(orient="records"))  # noqa: E731
    return {"history": to_list(hist), "backtest": to_list(bt),
            "forecast": to_list(fc.rename(columns={"forecast": "sales"})), "backtest_metrics": sm}


class Item(BaseModel):
    family: str
    on_hand: float = Field(ge=0)
    on_order: float = Field(0, ge=0)
    lead_time_days: int = Field(2, ge=1, le=H)
    review_days: int = Field(7, ge=0, le=H)


class ReplenishRequest(BaseModel):
    store_nbr: int
    service_level: float = Field(0.95, ge=0.5, lt=1)
    items: list[Item] = Field(min_length=1, max_length=100)


@app.post("/replenishment")
def replenish(req: ReplenishRequest):
    out = []
    for it in req.items:
        if it.lead_time_days + it.review_days > H:
            raise HTTPException(422, f"lead_time_days + review_days must be <= {H}")
        fc = predict(req.store_nbr, it.family)["forecast"].tolist()
        sd = float(S["stats"]["resid_std"].get((req.store_nbr, it.family), 0.0))
        r = replenishment(fc, sd, it.on_hand, it.on_order, it.lead_time_days, it.review_days, req.service_level)
        out.append({"family": it.family, **r})
    return {"store_nbr": req.store_nbr, "service_level": req.service_level,
            "items": sorted(out, key=lambda r: -r["stockout_risk_before_delivery"])}


class Actual(BaseModel):
    date: date
    store_nbr: int
    family: str
    sales: float = Field(ge=0)


@app.post("/actuals")
def post_actuals(rows: list[Actual]):
    now = datetime.now(timezone.utc).isoformat()
    pd.DataFrame([r.model_dump() for r in rows]).assign(received_at=now).to_sql(
        "actuals", engine, if_exists="append", index=False)
    return {"inserted": len(rows)}


@app.get("/monitoring")
def monitoring():
    lat = np.array([ms for _, ms in latency]) if latency else np.array([0.0])
    live = None
    # Latest logged forecast per (date, series) joined with posted actuals -> live accuracy vs backtest.
    df = pd.read_sql(text(
        "SELECT a.sales, p.forecast FROM actuals a JOIN ("
        "  SELECT date, store_nbr, family, forecast, ROW_NUMBER() OVER (PARTITION BY date, store_nbr, family "
        "  ORDER BY created_at DESC) AS rn FROM prediction_log) p "
        "ON p.date = a.date AND p.store_nbr = a.store_nbr AND p.family = a.family AND p.rn = 1"), engine)
    base = S["report"]["backtest_wape"]
    if len(df) and df["sales"].sum() > 0:
        live = metrics(df["sales"], df["forecast"]) | {"n": len(df)}
        live["drift_alert"] = live["WAPE"] > 1.25 * base  # ponytail: fixed threshold; tune per business cost
    return {"requests": counts["requests"], "errors": counts["errors"],
            "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)},
            "backtest_wape": base, "live": live, "tables": inspect(engine).get_table_names()}


@app.get("/")
def dashboard():
    return FileResponse(Path(__file__).parent / "dashboard.html")
