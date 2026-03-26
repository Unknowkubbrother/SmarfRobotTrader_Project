from __future__ import annotations

import json
from collections.abc import Mapping


DEFAULT_TRADING_SCHEDULE: dict[str, bool] = {
    "mon": True,
    "tue": True,
    "wed": True,
    "thu": True,
    "fri": True,
    "sat": False,
    "sun": False,
}


_DAY_ALIASES: dict[str, tuple[str, ...]] = {
    "mon": ("mon", "monday"),
    "tue": ("tue", "tues", "tuesday"),
    "wed": ("wed", "weds", "wednesday"),
    "thu": ("thu", "thur", "thurs", "thursday"),
    "fri": ("fri", "friday"),
    "sat": ("sat", "saturday"),
    "sun": ("sun", "sunday"),
}

_ALIAS_TO_KEY = {
    alias: day_key
    for day_key, aliases in _DAY_ALIASES.items()
    for alias in aliases
}


def normalize_trading_schedule(value) -> dict[str, bool]:
    payload: Mapping[str, object]
    if isinstance(value, Mapping):
        payload = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = {}
        payload = parsed if isinstance(parsed, Mapping) else {}
    else:
        payload = {}

    nested_schedule = payload.get("schedule") if isinstance(payload, Mapping) else None
    if isinstance(nested_schedule, Mapping):
        payload = nested_schedule

    normalized = dict(DEFAULT_TRADING_SCHEDULE)
    for raw_key, raw_value in payload.items():
        key = _ALIAS_TO_KEY.get(str(raw_key).strip().lower())
        if key is None:
            continue
        normalized[key] = bool(raw_value)
    return normalized
