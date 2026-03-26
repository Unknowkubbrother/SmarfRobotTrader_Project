from __future__ import annotations

import json
from collections.abc import Mapping

from .trading_schedule import normalize_trading_schedule


DEFAULT_RISK_LEVEL = "medium"
DEFAULT_RISK_MODE = "level"
CUSTOM_LOT_RISK_MODE = "custom_lot"
SUPPORTED_RISK_MODES = {DEFAULT_RISK_MODE, CUSTOM_LOT_RISK_MODE}
DEFAULT_RISK_PROFILE_MAP: dict[str, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
}
DEFAULT_RISK_PIPS = 50.0
DEFAULT_PIP_VALUE_PER_LOT = 10.0
MIN_LOT = 0.01
LOT_STEP = 0.01


def _parse_mapping(value) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def normalize_risk_level(value) -> str:
    raw = value.value if hasattr(value, "value") else value
    text = str(raw or "").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    return DEFAULT_RISK_LEVEL


def normalize_risk_mode(value) -> str:
    raw = value.value if hasattr(value, "value") else value
    text = str(raw or "").strip().lower()
    if text in SUPPORTED_RISK_MODES:
        return text
    return DEFAULT_RISK_MODE


def normalize_custom_lot(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        lot = float(value)
    except (TypeError, ValueError):
        return None
    if lot < MIN_LOT:
        return None
    steps = int(lot / LOT_STEP)
    normalized = max(MIN_LOT, LOT_STEP * steps)
    return round(normalized, 2)


def parse_bot_runtime_settings(value, *, risk_level=None) -> dict[str, object]:
    payload = _parse_mapping(value)
    risk_payload = payload.get("risk") if isinstance(payload.get("risk"), Mapping) else {}

    custom_lot = normalize_custom_lot(risk_payload.get("custom_lot"))
    risk_mode = normalize_risk_mode(risk_payload.get("mode"))
    if risk_mode == CUSTOM_LOT_RISK_MODE and custom_lot is None:
        risk_mode = DEFAULT_RISK_MODE

    return {
        "schedule": normalize_trading_schedule(payload),
        "risk_level": normalize_risk_level(risk_level),
        "risk_mode": risk_mode,
        "custom_lot": custom_lot,
    }


def serialize_bot_runtime_settings(
    *,
    schedule=None,
    risk_mode=None,
    custom_lot=None,
) -> dict[str, object]:
    normalized_schedule = normalize_trading_schedule(schedule)
    normalized_custom_lot = normalize_custom_lot(custom_lot)
    normalized_risk_mode = normalize_risk_mode(risk_mode)
    if normalized_risk_mode == CUSTOM_LOT_RISK_MODE and normalized_custom_lot is None:
        normalized_risk_mode = DEFAULT_RISK_MODE

    payload: dict[str, object] = {
        "schedule": normalized_schedule,
        "risk": {
            "mode": normalized_risk_mode,
        },
    }
    if normalized_risk_mode == CUSTOM_LOT_RISK_MODE and normalized_custom_lot is not None:
        payload["risk"]["custom_lot"] = normalized_custom_lot
    return payload


def merge_bot_runtime_settings(
    current,
    *,
    schedule=None,
    risk_mode=None,
    custom_lot=None,
) -> dict[str, object]:
    existing = parse_bot_runtime_settings(current)
    next_schedule = existing["schedule"] if schedule is None else schedule
    next_risk_mode = existing["risk_mode"] if risk_mode is None else risk_mode
    next_custom_lot = existing["custom_lot"] if custom_lot is None else custom_lot
    if normalize_risk_mode(next_risk_mode) != CUSTOM_LOT_RISK_MODE:
        next_custom_lot = None
    return serialize_bot_runtime_settings(
        schedule=next_schedule,
        risk_mode=next_risk_mode,
        custom_lot=next_custom_lot,
    )


def estimate_lot_size(
    *,
    balance,
    risk_level=None,
    risk_mode=None,
    custom_lot=None,
    risk_profile_map: Mapping[str, float] | None = None,
    risk_pips: float = DEFAULT_RISK_PIPS,
    pip_value_per_lot: float = DEFAULT_PIP_VALUE_PER_LOT,
) -> float:
    normalized_risk_mode = normalize_risk_mode(risk_mode)
    normalized_custom_lot = normalize_custom_lot(custom_lot)
    if normalized_risk_mode == CUSTOM_LOT_RISK_MODE and normalized_custom_lot is not None:
        return normalized_custom_lot

    profile = dict(DEFAULT_RISK_PROFILE_MAP)
    if isinstance(risk_profile_map, Mapping):
        for key, value in risk_profile_map.items():
            level = normalize_risk_level(key)
            try:
                pct = float(value)
            except (TypeError, ValueError):
                continue
            if pct > 0:
                profile[level] = pct

    level = normalize_risk_level(risk_level)
    pct = float(profile.get(level, profile[DEFAULT_RISK_LEVEL]))
    try:
        account_balance = float(balance)
    except (TypeError, ValueError):
        account_balance = 0.0

    safe_risk_pips = max(1.0, float(risk_pips or DEFAULT_RISK_PIPS))
    safe_pip_value = max(0.0001, float(pip_value_per_lot or DEFAULT_PIP_VALUE_PER_LOT))
    risk_amount = account_balance * pct / 100.0
    lot = risk_amount / (safe_risk_pips * safe_pip_value)
    steps = int(lot / LOT_STEP)
    normalized_lot = max(MIN_LOT, LOT_STEP * steps)
    return round(normalized_lot, 2)
