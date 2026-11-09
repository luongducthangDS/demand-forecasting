"""End-to-end pipeline: features -> Optuna tuning -> rolling-origin backtest -> final model -> artifacts.

    python -m forecast.train                # full run (15 Optuna trials)
    python -m forecast.train --trials 0     # skip tuning, use defaults
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap

from forecast import backtest as bt
from forecast.data import build_panel
from forecast.features import FEATURES, H, make_features

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
VAL_START = pd.Timestamp("2017-06-07")  # tuning window, strictly before every backtest window
FOLDS = [pd.Timestamp(d) for d in ("2017-06-21", "2017-07-05", "2017-07-19", "2017-08-02")]
FUTURE_START = pd.Timestamp("2017-08-16")
KEYS = ["date", "store_nbr", "family"]
COLS = KEYS + [f for f in FEATURES if f not in KEYS]


def tune(feat: pd.DataFrame, trials: int) -> tuple[dict, int]:
    train, val = bt.split(feat, VAL_START)
    train = train[train["date"] >= "2016-06-01"]  # ponytail: shorter window keeps a trial ~1 min; final fits use all data
    if trials == 0:
        m = bt.fit_lgb(train, bt.LGB_DEFAULT, 2000, valid=val)
        return dict(bt.LGB_DEFAULT), m.best_iteration

    def objective(trial: optuna.Trial) -> float:
        params = dict(bt.LGB_DEFAULT,
                      num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
                      min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
                      feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
                      bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                      lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
                      objective=trial.suggest_categorical("objective", ["regression", "huber"]))
        m = bt.fit_lgb(train, params, 2000, valid=val)
        trial.set_user_attr("rounds", m.best_iteration)
        return bt.metrics(val["sales"], bt.postprocess(val, m.predict(val[FEATURES])))["WAPE"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    best = dict(bt.LGB_DEFAULT, **study.best_params)
    print(f"optuna best val WAPE={study.best_value:.4f} params={study.best_params}")
    return best, study.best_trial.user_attrs["rounds"]


def run_backtest(panel: pd.DataFrame, feat: pd.DataFrame, params: dict, rounds: int) -> pd.DataFrame:
    rows = []
    for k, start in enumerate(FOLDS):
        t0 = time.time()
        train, test = bt.split(feat, start)
        out = test[KEYS + ["sales", "hol_national"]].copy()
        out["fold"] = k
        out["MovingAverage28"] = bt.moving_average(panel, start)
        out["SeasonalNaive"] = bt.seasonal_naive(panel, start)
        out["XGBoost"] = bt.postprocess(test, bt.fit_xgb(train).predict(test[FEATURES]))
        out["LightGBM_default"] = bt.postprocess(
            test, bt.fit_lgb(train, bt.LGB_DEFAULT, 600).predict(test[FEATURES]))
        out["LightGBM_tuned"] = bt.postprocess(test, bt.fit_lgb(train, params, rounds).predict(test[FEATURES]))
        rows.append(out)
        print(f"fold {k} {start.date()} done in {time.time() - t0:.0f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


MODELS = ["MovingAverage28", "SeasonalNaive", "XGBoost", "LightGBM_default", "LightGBM_tuned"]


def summarize(pred: pd.DataFrame) -> dict:
    s = {m: bt.metrics(pred["sales"], pred[m]) for m in MODELS}
    for m in MODELS:
        per_fold = [bt.metrics(g["sales"], g[m])["WAPE"] for _, g in pred.groupby("fold")]
        s[m]["WAPE_fold_std"] = float(np.std(per_fold))
    return s


def error_analysis(pred: pd.DataFrame, best: str) -> dict:
    p = pred.assign(err=(pred[best] - pred["sales"]).abs(),
                    horizon=lambda d: (d["date"] - d.groupby("fold")["date"].transform("min")).dt.days + 1)
    wape = lambda g: g["err"].sum() / g["sales"].sum()  # noqa: E731
    fam = p.groupby("family", observed=True).apply(
        lambda g: pd.Series({"WAPE": wape(g), "share_of_error": g["err"].sum() / p["err"].sum()}),
        include_groups=False)
    return {
        "by_family": fam.sort_values("share_of_error", ascending=False).head(10).round(4).to_dict("index"),
        "by_horizon_day": p.groupby("horizon").apply(wape, include_groups=False).round(4).to_dict(),
        "holiday_vs_normal": p.groupby("hol_national").apply(wape, include_groups=False).round(4).to_dict(),
        "by_fold": p.groupby("fold").apply(wape, include_groups=False).round(4).to_dict(),
    }


def write_benchmark(summary: dict, errors: dict, best: str, params: dict, rounds: int, n_series: int) -> str:
    lines = ["# Benchmark: rolling-origin backtest", "",
             f"{len(FOLDS)} folds × {H}-day horizon, origins {', '.join(str(f.date()) for f in FOLDS)}; "
             f"{n_series} series (store × family). Metrics pooled over all folds (unit: sales units/day).", "",
             "| Model | MAE | RMSE | WAPE | WAPE std across folds | RMSLE | Bias |",
             "|---|---|---|---|---|---|---|"]
    for m, v in sorted(summary.items(), key=lambda kv: kv[1]["WAPE"]):
        lines.append(f"| {m}{' **(selected)**' if m == best else ''} | {v['MAE']:.1f} | {v['RMSE']:.1f} | "
                     f"{v['WAPE']:.4f} | {v['WAPE_fold_std']:.4f} | {v['RMSLE']:.4f} | {v['bias']:+.4f} |")
    ma = summary["MovingAverage28"]["WAPE"]
    lines += ["", f"{best} lowers WAPE by {100 * (1 - summary[best]['WAPE'] / ma):.1f}% vs MovingAverage28.", "",
              "## Error analysis (selected model)", "", "WAPE by horizon day (1 = day after origin):", "",
              "| " + " | ".join(str(k) for k in errors["by_horizon_day"]) + " |",
              "|" + "---|" * len(errors["by_horizon_day"]),
              "| " + " | ".join(f"{v:.3f}" for v in errors["by_horizon_day"].values()) + " |", "",
              f"WAPE per fold: {', '.join(f'{k}: {v:.3f}' for k, v in errors['by_fold'].items())}", "",
              f"WAPE on normal days: {errors['holiday_vs_normal'].get(0, float('nan')):.3f}; on national holidays: {errors['holiday_vs_normal'].get(1, float('nan')):.3f}", "",
              "Top 10 families by share of total absolute error:", "",
              "| Family | WAPE | Share of error |", "|---|---|---|"]
    for fam, v in errors["by_family"].items():
        lines.append(f"| {fam} | {v['WAPE']:.3f} | {v['share_of_error']:.1%} |")
    lines += ["", f"Tuned params ({rounds} rounds): `{json.dumps({k: params[k] for k in sorted(params) if k not in ('verbose', 'num_threads', 'seed')})}`"]
    return "\n".join(lines) + "\n"


def explain(model, sample: pd.DataFrame) -> dict:
    sv = bt.contributions(model, sample[FEATURES])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    shap.summary_plot(sv, sample[FEATURES], plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(ART / "shap_importance.png", dpi=120)
    plt.close()
    return imp.head(15).round(4).to_dict()


def main(trials: int) -> None:
    ART.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("demand-forecasting")

    t0 = time.time()
    panel = build_panel()
    feat = make_features(panel)
    print(f"features ready: {feat.shape} in {time.time() - t0:.0f}s")

    with mlflow.start_run(run_name="backtest"):
        params, rounds = tune(feat, trials)
        mlflow.log_params({**{k: v for k, v in params.items() if k not in ("verbose", "num_threads")},
                           "rounds": rounds, "horizon": H, "folds": len(FOLDS), "optuna_trials": trials})

        pred = run_backtest(panel, feat, params, rounds)
        summary = summarize(pred)
        best = min(("XGBoost", "LightGBM_default", "LightGBM_tuned"), key=lambda m: summary[m]["WAPE"])
        for m, v in summary.items():
            mlflow.log_metrics({f"{m}.{k}": x for k, x in v.items()})
        errors = error_analysis(pred, best)

        # Final model: everything before the real future window, same params.
        train_all = feat[(feat["date"] >= bt.TRAIN_START) & (feat["date"] < FUTURE_START)]
        if best == "XGBoost":
            final = bt.fit_xgb(train_all)
        elif best == "LightGBM_tuned":
            final = bt.fit_lgb(train_all, params, rounds)
        else:
            final = bt.fit_lgb(train_all, bt.LGB_DEFAULT, 600)
        bt.save_model(final, ART)

        future = feat[bt.window(feat, FUTURE_START)]
        future[COLS].to_parquet(ART / "future_features.parquet")
        pred[KEYS + ["fold", "sales", "MovingAverage28", best]].rename(columns={best: "forecast"}).to_parquet(
            ART / "backtest_predictions.parquet")
        resid = (pred["sales"] - pred[best]).groupby([pred["store_nbr"], pred["family"]], observed=True)
        stats = pd.DataFrame({"resid_std": resid.std(), "bias": resid.mean()}).reset_index()
        stats.to_parquet(ART / "series_stats.parquet")
        hist = panel[(panel["date"] >= FUTURE_START - pd.Timedelta(days=120)) & (panel["is_future"] == 0)]
        hist[KEYS + ["sales", "onpromotion"]].to_parquet(ART / "history.parquet")

        top = explain(final, train_all.sample(5000, random_state=0))
        report = {"selected_model": best, "summary": summary, "errors": errors, "shap_top15": top,
                  "params": params, "rounds": rounds, "future_start": str(FUTURE_START.date()), "horizon": H,
                  "backtest_wape": summary[best]["WAPE"]}
        (ART / "report.json").write_text(json.dumps(report, indent=2, default=str))
        md = write_benchmark(summary, errors, best, params, rounds, feat.groupby(["store_nbr", "family"], observed=True).ngroups)
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "BENCHMARK.md").write_text(md, encoding="utf-8")
        for f in (bt.model_path(ART), ART / "report.json", ART / "shap_importance.png"):
            mlflow.log_artifact(str(f))
        mlflow.log_artifact(str(ROOT / "docs" / "BENCHMARK.md"))
        print(md)
        print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    main(ap.parse_args().trials)
