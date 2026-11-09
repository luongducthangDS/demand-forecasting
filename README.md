# Demand Forecasting: store × product family, 14 days ahead

Daily sales forecast for 54 stores × 33 product families (1,782 series) of Corporación Favorita (Ecuador),
used to flag stockout risk and propose order quantities. Covers the full ML lifecycle: data pipeline,
leakage-safe features, baselines, tuned gradient boosting, rolling-origin backtest, SHAP, MLflow tracking,
FastAPI inference, PostgreSQL logging, a dashboard and online accuracy monitoring.

## Results

Rolling-origin backtest, 4 origins × 14 days (2017-06-21 → 2017-08-15), 1,782 series, ~100k forecasts per model:

| Model | MAE | RMSE | WAPE | WAPE std (folds) | Bias |
|---|---|---|---|---|---|
| **XGBoost (served)** | **53.4** | **206.9** | **0.113** | 0.012 | −2.7% |
| LightGBM, Optuna-tuned | 54.6 | 210.4 | 0.115 | 0.013 | −1.8% |
| LightGBM, default | 54.7 | 209.9 | 0.116 | 0.011 | −2.4% |
| Seasonal naive (same weekday, last week) | 89.7 | 339.4 | 0.190 | 0.029 | +3.2% |
| Moving average 28 days | 109.4 | 379.7 | 0.231 | 0.020 | +1.8% |

- WAPE halves vs the moving average (−51%) and drops 40% vs seasonal naive.
- Optuna (15 trials on a separate window) improved LightGBM by only 0.2%: the features carry the model,
  not the hyperparameters. XGBoost won on the backtest, so the pipeline serves it; the choice is data-driven.
- Error is 63% higher on national holidays (WAPE 0.182 vs 0.112); BEVERAGES, GROCERY I and CLEANING
  account for 61% of absolute error. Bias is slightly negative (under-forecast), the costly side for stockouts.
- Top SHAP drivers: 7-day rolling mean (as of t−14), lag 14, same-weekday mean, product family, promotions.
- API: ~36 ms p50 per live 14-day inference including the PostgreSQL log write (local, CPU).

Full table and error analysis: [docs/BENCHMARK.md](docs/BENCHMARK.md). Feature importance: `artifacts/shap_importance.png`.

## Pipeline

```
data/raw/*.csv ─► forecast/data.py      complete daily grid, holidays (national/regional/local, transfers),
                                         oil price, store metadata, future rows from test.csv
              ─► forecast/features.py   lags ≥ 14 days, rolling stats on the t−14 series, promo, calendar
              ─► forecast/train.py      Optuna (validation window) ─► 4-fold rolling backtest
                                         ─► best model refit ─► artifacts/ + MLflow + docs/BENCHMARK.md
              ─► forecast/api.py        live inference, replenishment, actuals feedback, monitoring
                                         PostgreSQL: history, backtest, prediction_log, actuals
```

## Design decisions

| Decision | Why |
|---|---|
| Direct 14-day model, every sales lag ≥ 14 | One model predicts the whole window from one origin; no recursive error build-up and no way to read sales the business would not have yet. |
| Oil price lagged 14 days | `oil.csv` contains future prices; using them is leakage. |
| Planned promotions used at t | Promotions are scheduled in advance (they are given in `test.csv`). |
| Target `log1p(sales)` | Sales span 0 to >100k per day across families; log keeps big series from dominating the loss. |
| Tuning window 2017-06-07…06-20, before all test windows | Hyperparameters never see a backtest window. |
| Rolling-origin backtest, 4 × 14 days | Reports variance across origins, not one lucky split. |
| Series with zero sales for 56 days → forecast 0 | Families a store does not carry. |
| Safety stock from per-series backtest residual std | Uncertainty measured on held-out data, not on training fit. |

Leakage is tested, not assumed: `tests/test_core.py` rewrites all sales and oil prices after a cutoff and
asserts every feature row before cutoff + 14 days is unchanged, and that baselines ignore the forecast window.

## Run

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m forecast.data                              # download data (~125 MB) into data/raw/
python -m forecast.train --trials 15                 # tuning + backtest + final model
pytest -q
uvicorn forecast.api:app --port 8000                 # dashboard at http://localhost:8000
mlflow ui --backend-store-uri sqlite:///mlflow.db    # experiment tracking
```

With PostgreSQL (after training):

```bash
docker compose up --build                            # API + Postgres, dashboard at http://localhost:8000
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /forecast?store_nbr=44&family=GROCERY I` | Live 14-day inference; every call is written to `prediction_log` |
| `POST /replenishment` | Order-up-to quantity, safety stock, stockout probability before delivery, days of cover |
| `POST /actuals` | Feed real sales back |
| `GET /monitoring` | Request count, errors, p50/p95 latency, live WAPE vs backtest WAPE, drift alert at 1.25× |
| `GET /series/{store}/{family}` | History, backtest and forecast for the dashboard |

```bash
curl -X POST localhost:8000/replenishment -H "Content-Type: application/json" \
  -d '{"store_nbr":44,"service_level":0.95,"items":[{"family":"GROCERY I","on_hand":3000,"lead_time_days":2,"review_days":7}]}'
```

## Limitations

- The dataset has no inventory, lead times or costs. Stock on hand is an API input; replenishment is a
  standard order-up-to policy, not an optimised one.
- Safety stock assumes independent daily errors (σ·√days). Real errors are autocorrelated, so risk is
  understated; the upgrade is an empirical quantile of backtest errors summed over the protection window.
- The live forecast window is 2017-08-16…08-29 (the competition test period). Its actual sales are not
  public, so live accuracy is shown only for actuals posted to `/actuals`.
- Product family is a category, not a SKU. SKU-level demand is sparser and would need intermittent-demand
  methods (Croston/TSB, Tweedie loss).
- Retraining is a manual `python -m forecast.train`; no scheduler, no model registry promotion step.
