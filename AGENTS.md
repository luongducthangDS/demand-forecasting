# demand-forecasting: agent notes

Project 01 in the AI Engineer portfolio roadmap. Repo: https://github.com/luongducthangDS/demand-forecasting.

## 2026-11-09: built end to end

- Data: Kaggle "Store Sales" (Favorita, Ecuador), from the Hugging Face mirror `t4tiana/store-sales-time-series-forecasting`; `python -m forecast.data` downloads it into `data/raw/` (gitignored).
- Model files (`artifacts/model.*`, 45 MB) are gitignored: run `python -m forecast.train` before the API or Docker.
- Pipeline: `forecast/data.py` → `features.py` (direct 14-day model, every sales lag ≥ 14, oil lagged 14) → `train.py` (Optuna on a window before the backtest, 4-fold rolling backtest, final refit of the winning model, SHAP via native `pred_contrib`, MLflow in `mlflow.db`) → `api.py` + `dashboard.html`.
- Result (last full run, 3,808 s): XGBoost WAPE 0.113, LightGBM tuned 0.115, seasonal naive 0.190, MA28 0.231. Details in `docs/BENCHMARK.md`. XGBoost is served (`artifacts/model.json`).
- Verified: `pytest` 6 passed (leakage perturbation test included); uvicorn local; `docker compose up --build` with PostgreSQL, including `/actuals` → `/monitoring`.

## Gotchas

- Use `.venv` in this folder. The shell's default `python` is DocuMindAI's venv. 4 packages were installed there by mistake on 09/11 and then uninstalled; their dependencies (numba, llvmlite, etc.) may remain.
- Do not run anything CPU-heavy while training. One Optuna trial went from 12 s to 33 min while pytest ran in parallel (OpenMP oversubscription).
- Parquet drops the category dtype on int columns (`store_nbr`, `cluster`); `api.py` restores it at load. Do not remove that.
- MLflow 3 refuses the file store; use the sqlite URI (already set in train.py). Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` from this folder.

## Remaining

- Safety stock uses σ·√days (independent errors); upgrade to the empirical quantile of summed backtest errors.
- Holiday-day error is 63% higher than on normal days; next feature work: days-to/from holiday, holiday × family.
