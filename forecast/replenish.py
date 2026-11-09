"""Order-up-to replenishment from a daily forecast and the backtest error spread.

The dataset has no inventory, so stock on hand is an input (from the ERP/WMS in a real deployment).
"""
from math import sqrt
from statistics import NormalDist


def replenishment(forecast: list[float], resid_std: float, on_hand: float, on_order: float = 0.0,
                  lead_time_days: int = 2, review_days: int = 7, service_level: float = 0.95) -> dict:
    protect = lead_time_days + review_days
    if lead_time_days < 1 or review_days < 0 or protect > len(forecast):
        raise ValueError(f"lead_time_days >= 1 and lead_time + review <= {len(forecast)} required")
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be in [0.5, 1)")
    z = NormalDist().inv_cdf(service_level)
    position = on_hand + on_order

    # ponytail: sqrt(days) assumes independent daily errors; real errors are autocorrelated, so this
    # understates risk. Upgrade: empirical quantile of backtest errors summed over the protection window.
    mu_p, sd_p = sum(forecast[:protect]), resid_std * sqrt(protect)
    order_up_to = mu_p + z * sd_p

    mu_lt, sd_lt = sum(forecast[:lead_time_days]), resid_std * sqrt(lead_time_days)
    if sd_lt > 0:
        risk = 1 - NormalDist(mu_lt, sd_lt).cdf(position)
    else:
        risk = float(mu_lt > position)

    cum, cover = 0.0, len(forecast)
    for day, f in enumerate(forecast):
        cum += f
        if cum > on_hand:
            cover = day
            break

    return {
        "demand_protection_window": round(mu_p, 2),
        "safety_stock": round(z * sd_p, 2),
        "order_up_to": round(order_up_to, 2),
        "order_qty": round(max(0.0, order_up_to - position), 2),
        "stockout_risk_before_delivery": round(risk, 4),
        "at_risk": risk > 1 - service_level,
        "days_of_cover": cover,
    }
