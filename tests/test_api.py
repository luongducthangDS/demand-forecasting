"""Runs against real artifacts (python -m forecast.train first); uses a throwaway SQLite DB."""
import os
from pathlib import Path

import pytest

ART = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
pytestmark = pytest.mark.skipif(not (ART / "report.json").exists(), reason="run forecast.train first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["DATABASE_URL"] = f"sqlite:///{(tmp_path_factory.mktemp('db') / 't.db').as_posix()}"
    from fastapi.testclient import TestClient

    from forecast.api import app
    with TestClient(app) as c:
        yield c


def test_forecast_replenish_monitoring_flow(client):
    assert client.get("/health").json()["status"] == "ok"
    fc = client.get("/forecast", params={"store_nbr": 44, "family": "GROCERY I"}).json()["forecast"]
    assert len(fc) == 14 and all(d["sales"] >= 0 for d in fc)

    r = client.post("/replenishment", json={"store_nbr": 44, "items": [{"family": "GROCERY I", "on_hand": 0}]})
    item = r.json()["items"][0]
    assert item["at_risk"] and item["order_qty"] > 0

    assert client.get("/forecast", params={"store_nbr": 44, "family": "NOPE"}).status_code == 404
    assert client.post("/replenishment", json={"store_nbr": 44, "items": [
        {"family": "GROCERY I", "on_hand": 1, "lead_time_days": 10, "review_days": 10}]}).status_code == 422

    actual = [{"date": d["date"], "store_nbr": 44, "family": "GROCERY I", "sales": d["sales"] * 1.1} for d in fc]
    assert client.post("/actuals", json=actual).json()["inserted"] == 14
    live = client.get("/monitoring").json()["live"]
    assert live["n"] == 14 and live["WAPE"] == pytest.approx(1 / 11, rel=1e-3)

    detail = client.get("/series/44/GROCERY I").json()
    assert detail["history"] and detail["backtest"] and len(detail["forecast"]) == 14
