import numpy as np


RISK_LEVEL_LOW = "low"
RISK_LEVEL_MEDIUM = "medium"
RISK_LEVEL_HIGH = "high"


def normalize_risk_level(value):
    raw = str(value if value is not None else "medium").strip().lower()
    if raw in {"1", "low"}:
        return RISK_LEVEL_LOW
    if raw in {"3", "high"}:
        return RISK_LEVEL_HIGH
    return RISK_LEVEL_MEDIUM


def resolve_risk_percent(
    risk_level=None,
    risk_pct=None,
    risk_percent_low=0.5,
    risk_percent_medium=1.0,
    risk_percent_high=2.0,
):
    if risk_pct is not None:
        return float(risk_pct)

    level = normalize_risk_level(risk_level)
    if level == RISK_LEVEL_LOW:
        return float(risk_percent_low)
    if level == RISK_LEVEL_HIGH:
        return float(risk_percent_high)
    return float(risk_percent_medium)


def calc_auto_lot(
    balance,
    risk_pct=None,
    risk_level=None,
    lot_risk_pips=50.0,
    pip_value_per_lot=10.0,
    min_lot=0.01,
    lot_step=0.01,
    risk_percent_low=0.5,
    risk_percent_medium=1.0,
    risk_percent_high=2.0,
):
    resolved_risk_pct = resolve_risk_percent(
        risk_level=risk_level,
        risk_pct=risk_pct,
        risk_percent_low=risk_percent_low,
        risk_percent_medium=risk_percent_medium,
        risk_percent_high=risk_percent_high,
    )

    risk_amount = float(balance) * float(resolved_risk_pct) / 100.0
    denom = max(float(lot_risk_pips) * float(pip_value_per_lot), 1e-9)
    raw_lot = risk_amount / denom

    step = max(float(lot_step), 1e-9)
    stepped = step * int(np.floor(raw_lot / step))
    lot = max(float(min_lot), stepped)
    step_str = f"{step:.10f}".rstrip("0")
    decimals = len(step_str.split(".")[1]) if "." in step_str else 0
    return round(lot, decimals)
