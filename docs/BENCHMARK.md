# Benchmark: rolling-origin backtest

4 folds × 14-day horizon, origins 2017-06-21, 2017-07-05, 2017-07-19, 2017-08-02; 1782 series (store × family). Metrics pooled over all folds (unit: sales units/day).

| Model | MAE | RMSE | WAPE | WAPE std across folds | RMSLE | Bias |
|---|---|---|---|---|---|---|
| XGBoost **(selected)** | 53.4 | 206.9 | 0.1129 | 0.0116 | 0.3897 | -0.0272 |
| LightGBM_tuned | 54.6 | 210.4 | 0.1153 | 0.0131 | 0.3881 | -0.0182 |
| LightGBM_default | 54.7 | 209.9 | 0.1155 | 0.0114 | 0.3904 | -0.0236 |
| SeasonalNaive | 89.7 | 339.4 | 0.1895 | 0.0292 | 0.5454 | +0.0320 |
| MovingAverage28 | 109.4 | 379.7 | 0.2310 | 0.0195 | 0.4767 | +0.0176 |

XGBoost lowers WAPE by 51.1% vs MovingAverage28.

## Error analysis (selected model)

WAPE by horizon day (1 = day after origin):

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.095 | 0.119 | 0.111 | 0.096 | 0.106 | 0.113 | 0.111 | 0.104 | 0.133 | 0.140 | 0.125 | 0.115 | 0.106 | 0.114 |

WAPE per fold: 0: 0.102, 1: 0.101, 2: 0.126, 3: 0.122

WAPE on normal days: 0.112; on national holidays: 0.182

Top 10 families by share of total absolute error:

| Family | WAPE | Share of error |
|---|---|---|
| BEVERAGES | 0.127 | 25.5% |
| GROCERY I | 0.086 | 22.2% |
| CLEANING | 0.184 | 13.6% |
| PRODUCE | 0.072 | 9.6% |
| DAIRY | 0.083 | 4.1% |
| BREAD/BAKERY | 0.098 | 3.0% |
| HOME CARE | 0.152 | 2.7% |
| MEATS | 0.124 | 2.7% |
| PERSONAL CARE | 0.137 | 2.4% |
| POULTRY | 0.104 | 2.3% |

Tuned params (298 rounds): `{"bagging_fraction": 0.8905396926012249, "bagging_freq": 1, "feature_fraction": 0.6546165660580835, "lambda_l2": 0.013577294146441802, "learning_rate": 0.05, "min_data_in_leaf": 106, "num_leaves": 237, "objective": "huber"}`
