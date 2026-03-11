from __future__ import annotations

import asyncio
import html
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from io import BytesIO
from typing import Any

import httpx

try:
    from prisma import Json
except Exception:
    Json = None

from ..database.client import db
from .notification_delivery import build_absolute_related_link, claim_notification_dedupe, dispatch_notification_to_user
from .subscription_access import sync_subscription_status_from_invoices
try:
    from prisma.errors import UniqueViolationError
except Exception:
    UniqueViolationError = None

try:
    import stripe
except Exception:
    stripe = None

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if stripe is not None and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

logger = logging.getLogger(__name__)

_BILLING_POLL_SECONDS = max(60, int(os.getenv("SUBSCRIPTION_BILLING_POLL_SECONDS", "900")))
_PROCESSABLE_SUB_STATUSES = {"active"}
_BILLING_NOTIFICATION_RELATED_LINK = "/subscription"
_BILLING_NOTIFICATION_DEDUPE_TTL_SECONDS = 60 * 60 * 24 * 45
_BILLING_BASE_CURRENCY = "usd"
_BILLING_CURRENCY = str(
    os.getenv("STRIPE_BILLING_CURRENCY")
    or os.getenv("STRIPE_CHECKOUT_CURRENCY")
    or _BILLING_BASE_CURRENCY
    or _BILLING_BASE_CURRENCY
).strip().lower() or _BILLING_BASE_CURRENCY

try:
    _BILLING_EXCHANGE_RATE = float(str(os.getenv("STRIPE_BILLING_EXCHANGE_RATE", "") or "").strip())
except Exception:
    _BILLING_EXCHANGE_RATE = 0.0

_PROMPTPAY_CURRENCY = str(os.getenv("STRIPE_PROMPTPAY_CURRENCY", "thb") or "thb").strip().lower() or "thb"
try:
    _PROMPTPAY_EXCHANGE_RATE = float(str(os.getenv("STRIPE_PROMPTPAY_USD_TO_THB_RATE", "") or "").strip())
except Exception:
    _PROMPTPAY_EXCHANGE_RATE = 0.0
_PROMPTPAY_FX_PROVIDER_URL = str(
    os.getenv("STRIPE_PROMPTPAY_FX_PROVIDER_URL", "https://api.frankfurter.dev/v1/latest") or ""
).strip()
try:
    _PROMPTPAY_FX_TIMEOUT_SECONDS = float(str(os.getenv("STRIPE_PROMPTPAY_FX_TIMEOUT_SEC", "5") or "5").strip())
except Exception:
    _PROMPTPAY_FX_TIMEOUT_SECONDS = 5.0
try:
    _PROMPTPAY_FX_CACHE_TTL_SECONDS = int(str(os.getenv("STRIPE_PROMPTPAY_FX_CACHE_TTL_SEC", "3600") or "3600").strip())
except Exception:
    _PROMPTPAY_FX_CACHE_TTL_SECONDS = 3600
_promptpay_fx_cache: dict[str, Any] = {}


@dataclass
class BillingCycleResult:
    subscription_id: str
    invoice_id: str | None
    invoice_created: bool
    status: str
    amount: float
    period_start: date
    period_end: date
    note: str = ""


@dataclass
class BillingRunSummary:
    processed_subscriptions: int = 0
    created_invoices: int = 0
    paid_invoices: int = 0
    pending_invoices: int = 0
    skipped_invoices: int = 0
    failed_invoices: int = 0


@dataclass
class ChargeAttemptResult:
    status: str
    payment_intent_id: str | None
    paid_at: datetime | None
    note: str
    local_payment_method_id: str | None = None
    provider_payment_method_id: str | None = None
    request_id: str | None = None
    charge_id: str | None = None
    balance_transaction_id: str | None = None
    payment_breakdown: dict[str, Any] | None = None
    payment_method_details: dict[str, Any] | None = None
    payment_error_details: dict[str, Any] | None = None


_ZERO_DECIMAL_CURRENCIES = {
    "BIF",
    "CLP",
    "DJF",
    "GNF",
    "JPY",
    "KMF",
    "KRW",
    "MGA",
    "PYG",
    "RWF",
    "UGX",
    "VND",
    "VUV",
    "XAF",
    "XOF",
    "XPF",
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def get_billing_base_currency() -> str:
    return _BILLING_BASE_CURRENCY.upper()


def get_billing_currency() -> str:
    return _BILLING_CURRENCY.upper()


def get_billing_exchange_rate(*, require_config: bool = False) -> float:
    if _BILLING_CURRENCY == _BILLING_BASE_CURRENCY:
        return 1.0
    if _BILLING_EXCHANGE_RATE > 0:
        return _BILLING_EXCHANGE_RATE
    if require_config:
        raise RuntimeError(
            "STRIPE_BILLING_EXCHANGE_RATE must be configured when billing currency is not USD"
        )
    return 1.0


def get_promptpay_currency() -> str:
    return _PROMPTPAY_CURRENCY.upper()


def promptpay_fx_provider_enabled() -> bool:
    return bool(_PROMPTPAY_FX_PROVIDER_URL)


def promptpay_fallback_rate_configured() -> bool:
    return _PROMPTPAY_EXCHANGE_RATE > 0


def promptpay_checkout_configured() -> bool:
    return promptpay_fx_provider_enabled() or promptpay_fallback_rate_configured()


def get_promptpay_exchange_rate(*, require_config: bool = False) -> float:
    if _PROMPTPAY_EXCHANGE_RATE > 0:
        return _PROMPTPAY_EXCHANGE_RATE
    if require_config:
        raise RuntimeError("STRIPE_PROMPTPAY_USD_TO_THB_RATE must be configured for PromptPay checkout")
    return 0.0


async def resolve_promptpay_exchange_rate(*, require_config: bool = False) -> float:
    now = datetime.now(timezone.utc)
    cached_rate = _to_float(_promptpay_fx_cache.get("rate"), 0.0)
    cached_at = _promptpay_fx_cache.get("fetched_at")
    if (
        cached_rate > 0
        and isinstance(cached_at, datetime)
        and (now - cached_at).total_seconds() < max(_PROMPTPAY_FX_CACHE_TTL_SECONDS, 0)
    ):
        return cached_rate

    if promptpay_fx_provider_enabled():
        try:
            timeout = httpx.Timeout(max(_PROMPTPAY_FX_TIMEOUT_SECONDS, 1.0))
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    _PROMPTPAY_FX_PROVIDER_URL,
                    params={"base": "USD", "symbols": get_promptpay_currency()},
                )
                response.raise_for_status()
                payload = response.json()
            rates = payload.get("rates") or {}
            rate = _to_float(rates.get(get_promptpay_currency()), 0.0)
            if rate > 0:
                _promptpay_fx_cache.clear()
                _promptpay_fx_cache.update(
                    {
                        "rate": rate,
                        "fetched_at": now,
                        "provider_date": payload.get("date"),
                    }
                )
                return rate
            logger.warning("PromptPay FX provider returned invalid rate payload: %s", payload)
        except Exception as error:
            logger.warning("PromptPay FX provider lookup failed: %s", error)

    fallback_rate = get_promptpay_exchange_rate(require_config=False)
    if fallback_rate > 0:
        return fallback_rate
    if require_config:
        raise RuntimeError("PromptPay FX rate provider failed and no fallback rate is configured")
    return 0.0


def convert_invoice_amount_to_promptpay_currency(
    amount_usd: float,
    *,
    require_config: bool = False,
    exchange_rate: float | None = None,
) -> float:
    rate = _to_float(exchange_rate, 0.0) if exchange_rate is not None else get_promptpay_exchange_rate(require_config=require_config)
    if rate <= 0:
        if require_config:
            raise RuntimeError("PromptPay exchange rate is not configured")
        return 0.0
    return round(max(amount_usd, 0.0) * rate, 2)


def promptpay_amount_to_minor_units(
    amount_usd: float,
    *,
    require_config: bool = False,
    exchange_rate: float | None = None,
) -> int:
    payment_amount = convert_invoice_amount_to_promptpay_currency(
        amount_usd,
        require_config=require_config,
        exchange_rate=exchange_rate,
    )
    currency = get_promptpay_currency()
    factor = 1 if currency in _ZERO_DECIMAL_CURRENCIES else 100
    return int(round(payment_amount * factor))


def convert_invoice_amount_to_billing_currency(amount_usd: float, *, require_config: bool = False) -> float:
    rate = get_billing_exchange_rate(require_config=require_config)
    return round(max(amount_usd, 0.0) * rate, 2)


def billing_amount_to_minor_units(amount_usd: float, *, require_config: bool = False) -> int:
    payment_amount = convert_invoice_amount_to_billing_currency(amount_usd, require_config=require_config)
    currency = get_billing_currency()
    factor = 1 if currency in _ZERO_DECIMAL_CURRENCIES else 100
    return int(round(payment_amount * factor))


def format_currency_amount(amount: float, currency: str | None = None) -> str:
    currency_code = str(currency or get_billing_base_currency()).upper()
    if currency_code == "USD":
        return f"${amount:,.2f}"
    if currency_code == "THB":
        return f"THB {amount:,.2f}"
    return f"{currency_code} {amount:,.2f}"


def format_billing_currency(amount_usd: float, *, require_config: bool = False) -> str:
    return format_currency_amount(
        convert_invoice_amount_to_billing_currency(amount_usd, require_config=require_config),
        get_billing_currency(),
    )


def build_expected_payment_breakdown(
    invoice_amount_usd: float,
    *,
    expected_payment_amount: float | None = None,
    expected_payment_currency: str | None = None,
    configured_exchange_rate: float | None = None,
) -> dict[str, Any]:
    payment_currency = str(expected_payment_currency or get_billing_currency()).upper()
    require_config = invoice_amount_usd > 0
    payment_amount = (
        round(_to_float(expected_payment_amount, 0.0), 2)
        if expected_payment_amount is not None
        else convert_invoice_amount_to_billing_currency(invoice_amount_usd, require_config=require_config)
    )
    exchange_rate = (
        _to_float(configured_exchange_rate, 0.0)
        if configured_exchange_rate is not None
        else get_billing_exchange_rate(require_config=require_config)
    )
    return _strip_none_values(
        {
            "invoice_amount": round(invoice_amount_usd, 2),
            "invoice_currency": get_billing_base_currency(),
            "expected_payment_amount": payment_amount,
            "expected_payment_currency": payment_currency,
            "configured_exchange_rate": exchange_rate if payment_currency != get_billing_base_currency() else None,
        }
    ) or {}


def merge_payment_breakdown_with_expected_amount(
    payment_breakdown: dict[str, Any] | None,
    *,
    invoice_amount_usd: float,
    expected_payment_amount: float | None = None,
    expected_payment_currency: str | None = None,
    configured_exchange_rate: float | None = None,
) -> dict[str, Any]:
    merged = dict(payment_breakdown or {})
    merged.update(
        build_expected_payment_breakdown(
            invoice_amount_usd,
            expected_payment_amount=expected_payment_amount,
            expected_payment_currency=expected_payment_currency,
            configured_exchange_rate=configured_exchange_rate,
        )
    )
    return merged


def get_invoice_payment_amount_details(invoice: Any) -> dict[str, Any]:
    calculated_fee = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
    payment_breakdown = getattr(invoice, "paymentBreakdown", None)
    if isinstance(payment_breakdown, dict):
        breakdown = payment_breakdown
    else:
        breakdown = {}

    invoice_amount = round(_to_float(breakdown.get("invoice_amount"), calculated_fee), 2)
    invoice_currency = str(breakdown.get("invoice_currency") or get_billing_base_currency()).upper()
    expected_payment_amount = round(
        _to_float(
            breakdown.get("expected_payment_amount"),
            convert_invoice_amount_to_billing_currency(calculated_fee),
        ),
        2,
    )
    expected_payment_currency = str(
        breakdown.get("expected_payment_currency") or get_billing_currency()
    ).upper()
    actual_payment_amount = breakdown.get("payment_amount")
    actual_payment_currency = breakdown.get("payment_currency")
    configured_exchange_rate = _to_float(
        breakdown.get("configured_exchange_rate"),
        get_billing_exchange_rate(),
    )

    return {
        "invoice_amount": invoice_amount,
        "invoice_currency": invoice_currency,
        "expected_payment_amount": expected_payment_amount,
        "expected_payment_currency": expected_payment_currency,
        "actual_payment_amount": _to_float(actual_payment_amount, 0.0)
        if actual_payment_amount is not None
        else None,
        "actual_payment_currency": str(actual_payment_currency or "").upper() or None,
        "configured_exchange_rate": configured_exchange_rate,
    }


def _strip_none_values(payload: dict[str, Any]) -> dict[str, Any] | None:
    cleaned = {
        key: value
        for key, value in payload.items()
        if value is not None and value != "" and value != [] and value != {}
    }
    return cleaned or None


def _sanitize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "to_dict_recursive"):
        try:
            return _sanitize_json(value.to_dict_recursive())
        except Exception:
            pass
    if isinstance(value, dict):
        return {
            str(key): _sanitize_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(item) for item in value]
    if hasattr(value, "items"):
        try:
            return {
                str(key): _sanitize_json(item)
                for key, item in value.items()
            }
        except Exception:
            pass
    return str(value)


def _to_prisma_json(value: Any) -> Any:
    sanitized = _sanitize_json(value)
    if Json is None:
        return sanitized
    return Json(sanitized)


def _minor_to_major(amount_minor: Any, currency: str | None) -> float | None:
    minor_value = _to_int(amount_minor, None)
    if minor_value is None:
        return None
    currency_code = str(currency or "").upper()
    factor = 1 if currency_code in _ZERO_DECIMAL_CURRENCIES else 100
    decimals = 0 if factor == 1 else 2
    return round(minor_value / factor, decimals)


def minor_to_major_amount(amount_minor: Any, currency: str | None) -> float | None:
    return _minor_to_major(amount_minor, currency)


def _extract_presentment_breakdown(stripe_object: Any) -> dict[str, Any] | None:
    if not stripe_object or not hasattr(stripe_object, "get"):
        return None

    presentment = _sanitize_json(stripe_object.get("presentment_details") or {})
    if not isinstance(presentment, dict):
        return None

    presentment_amount_minor = _to_int(presentment.get("presentment_amount"), None)
    presentment_currency = str(presentment.get("presentment_currency") or "").upper() or None
    if presentment_amount_minor is None or not presentment_currency:
        return None

    return _strip_none_values(
        {
            "presentment_amount_minor": presentment_amount_minor,
            "presentment_amount": _minor_to_major(presentment_amount_minor, presentment_currency),
            "presentment_currency": presentment_currency,
        }
    )


def _extract_request_id(value: Any) -> str | None:
    if value is None:
        return None

    request_id = getattr(value, "request_id", None)
    if request_id:
        return str(request_id)

    last_response = getattr(value, "last_response", None)
    response_request_id = getattr(last_response, "request_id", None)
    if response_request_id:
        return str(response_request_id)

    return None


def _enum_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _extract_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return None


def _datetime_at_start(value: date) -> datetime:
    return datetime.combine(value, time.min)


def _datetime_at_end(value: date) -> datetime:
    return datetime.combine(value, time.max)


def _stripe_enabled() -> bool:
    return stripe is not None and bool(STRIPE_SECRET_KEY)


def calculate_fee(
    *,
    net_profit: float,
    fee_type: str,
    fee_value: float,
    min_profit_threshold: float,
) -> float:
    if net_profit <= 0 or net_profit <= min_profit_threshold:
        return 0.0

    if fee_type == "fixed":
        return max(fee_value, 0.0)

    return max((net_profit * fee_value) / 100.0, 0.0)


def billing_period_for_due_date(due_date: date) -> tuple[date, date]:
    period_end = due_date - timedelta(days=1)
    period_start = period_end - timedelta(days=6)
    return (period_start, period_end)


def next_due_date_for_period_end(period_end: date) -> date:
    return period_end + timedelta(days=8)


async def _stripe_call(callable_obj, *args, **kwargs):
    return await asyncio.to_thread(callable_obj, *args, **kwargs)


async def _find_existing_invoice(subscription_id: str, period_start: date, period_end: date):
    return await db.invoice.find_first(
        where={
            "subId": subscription_id,
            "billingStartDate": _datetime_at_start(period_start),
            "billingEndDate": _datetime_at_start(period_end),
        }
    )


def _invoice_to_daily_aggregate_billing_status(status: Any) -> str:
    normalized = str(_enum_value(status) or "unbilled").strip().lower()
    if normalized in {"pending", "paid", "skipped", "failed"}:
        return normalized
    return "unbilled"


async def _get_unbilled_period_daily_aggregates(user_id: str, period_start: date, period_end: date) -> list[Any]:
    await release_orphaned_period_daily_aggregate_reservations(
        user_id=user_id,
        period_start=period_start,
        period_end=period_end,
    )
    return await db.dailyaggregate.find_many(
        where={
            "account": {
                "userId": user_id,
                "recordStatus": "active",
            },
            "date": {
                "gte": _datetime_at_start(period_start),
                "lte": _datetime_at_end(period_end),
            },
            "billingInvoiceId": None,
        }
    )


async def release_orphaned_period_daily_aggregate_reservations(
    *,
    user_id: str,
    period_start: date,
    period_end: date,
) -> int:
    reserved_rows = await db.dailyaggregate.find_many(
        where={
            "account": {
                "userId": user_id,
                "recordStatus": "active",
            },
            "date": {
                "gte": _datetime_at_start(period_start),
                "lte": _datetime_at_end(period_end),
            },
            "billingInvoiceId": {"not": None},
        }
    )
    reserved_invoice_ids = {
        str(getattr(row, "billingInvoiceId", "") or "").strip()
        for row in reserved_rows
        if str(getattr(row, "billingInvoiceId", "") or "").strip()
    }
    if not reserved_invoice_ids:
        return 0

    existing_invoices = await db.invoice.find_many(
        where={"id": {"in": list(reserved_invoice_ids)}}
    )
    existing_invoice_ids = {str(getattr(invoice, "id", "") or "").strip() for invoice in existing_invoices}
    orphan_invoice_ids = [invoice_id for invoice_id in reserved_invoice_ids if invoice_id not in existing_invoice_ids]
    if not orphan_invoice_ids:
        return 0

    result = await db.dailyaggregate.update_many(
        where={
            "account": {
                "userId": user_id,
                "recordStatus": "active",
            },
            "date": {
                "gte": _datetime_at_start(period_start),
                "lte": _datetime_at_end(period_end),
            },
            "billingInvoiceId": {"in": orphan_invoice_ids},
        },
        data={
            "billingInvoiceId": None,
            "billingStatus": "unbilled",
        },
    )
    return _to_int(result, 0) or 0


async def get_assignable_period_daily_aggregates(
    *,
    user_id: str,
    period_start: date,
    period_end: date,
    invoice_id: str | None = None,
) -> list[Any]:
    await release_orphaned_period_daily_aggregate_reservations(
        user_id=user_id,
        period_start=period_start,
        period_end=period_end,
    )

    where_clause: dict[str, Any] = {
        "account": {
            "userId": user_id,
            "recordStatus": "active",
        },
        "date": {
            "gte": _datetime_at_start(period_start),
            "lte": _datetime_at_end(period_end),
        },
    }
    if invoice_id:
        where_clause["OR"] = [
            {"billingInvoiceId": None},
            {"billingInvoiceId": invoice_id},
        ]
    else:
        where_clause["billingInvoiceId"] = None

    return await db.dailyaggregate.find_many(where=where_clause)


def _sum_daily_aggregate_profit(rows: list[Any]) -> float:
    return round(sum(_to_float(getattr(row, "dailyNetProfit", None)) for row in rows), 2)


async def reserve_daily_aggregates_for_invoice(
    invoice_id: str,
    rows: list[Any],
    *,
    status: str,
) -> None:
    row_ids = [
        str(getattr(row, "id", "") or "").strip()
        for row in rows
        if str(getattr(row, "id", "") or "").strip()
    ]
    if not invoice_id or not row_ids:
        return

    await db.dailyaggregate.update_many(
        where={
            "id": {"in": row_ids},
            "billingInvoiceId": None,
        },
        data={
            "billingInvoiceId": invoice_id,
            "billingStatus": _invoice_to_daily_aggregate_billing_status(status),
        },
    )


async def sync_daily_aggregate_status_for_invoice(invoice_id: str, status: Any) -> None:
    normalized_status = _invoice_to_daily_aggregate_billing_status(status)
    if not invoice_id or normalized_status == "unbilled":
        return

    await db.dailyaggregate.update_many(
        where={"billingInvoiceId": invoice_id},
        data={"billingStatus": normalized_status},
    )


async def get_minimum_next_billing_date_for_subscription(subscription_id: str) -> date | None:
    latest_invoice = await db.invoice.find_first(
        where={
            "subId": subscription_id,
            "billingEndDate": {"not": None},
        },
        order={"billingEndDate": "desc"},
    )
    latest_period_end = _extract_date(getattr(latest_invoice, "billingEndDate", None)) if latest_invoice else None
    if not latest_period_end:
        return None
    return next_due_date_for_period_end(latest_period_end)


async def normalize_subscription_next_billing_date(
    subscription: Any,
    *,
    fallback_due_date: date | None = None,
) -> tuple[Any, date | None]:
    current_due_date = _extract_date(getattr(subscription, "nextBillingDate", None))
    normalized_due_date = current_due_date or fallback_due_date
    minimum_due_date = await get_minimum_next_billing_date_for_subscription(str(getattr(subscription, "id", "") or ""))
    if minimum_due_date and (normalized_due_date is None or normalized_due_date < minimum_due_date):
        normalized_due_date = minimum_due_date

    if normalized_due_date and current_due_date != normalized_due_date:
        subscription = await db.subscription.update(
            where={"id": str(subscription.id)},
            data={"nextBillingDate": _datetime_at_start(normalized_due_date)},
        )

    return subscription, normalized_due_date


async def reconcile_open_invoice_amount(
    invoice: Any,
    *,
    subscription: Any,
    user_id: str,
) -> Any:
    invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "pending").strip().lower()
    if invoice_status not in {"pending", "failed"}:
        return invoice

    period_start = _extract_date(getattr(invoice, "billingStartDate", None))
    period_end = _extract_date(getattr(invoice, "billingEndDate", None))
    if not period_start or not period_end:
        return invoice

    assignable_rows = await get_assignable_period_daily_aggregates(
        user_id=user_id,
        period_start=period_start,
        period_end=period_end,
        invoice_id=str(invoice.id),
    )
    unassigned_rows = [
        row for row in assignable_rows
        if not str(getattr(row, "billingInvoiceId", "") or "").strip()
    ]
    if unassigned_rows:
        await reserve_daily_aggregates_for_invoice(
            str(invoice.id),
            unassigned_rows,
            status=invoice_status,
        )

    recalculated_profit = _sum_daily_aggregate_profit(assignable_rows)
    recalculated_fee = round(
        calculate_fee(
            net_profit=recalculated_profit,
            fee_type=str(_enum_value(getattr(subscription, "feeType", None)) or "percentage"),
            fee_value=_to_float(getattr(subscription, "feeValue", None), 0.0),
            min_profit_threshold=_to_float(getattr(subscription, "minProfitThreshold", None), 0.0),
        ),
        2,
    )
    current_profit = round(_to_float(getattr(invoice, "totalPeriodProfit", None), 0.0), 2)
    current_fee = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
    needs_update = bool(unassigned_rows) or recalculated_profit != current_profit or recalculated_fee != current_fee
    if not needs_update:
        return invoice

    next_status = invoice_status if recalculated_fee > 0 else "skipped"
    update_payload: dict[str, Any] = {
        "totalPeriodProfit": Decimal(str(recalculated_profit)),
        "calculatedFee": Decimal(str(recalculated_fee)),
        "status": next_status,
        "paymentBreakdown": _to_prisma_json(build_expected_payment_breakdown(recalculated_fee)),
        "processorRequestId": None,
        "paidAt": None,
    }
    updated_invoice = await db.invoice.update(
        where={"id": str(invoice.id)},
        data=update_payload,
    )
    await sync_daily_aggregate_status_for_invoice(str(updated_invoice.id), next_status)
    await sync_subscription_status_from_invoices(str(subscription.id))
    return updated_invoice


async def _get_default_payment_method(user_id: str, default_method_id: str | None):
    if not default_method_id:
        return None
    return await db.userpaymentmethod.find_first(
        where={
            "id": default_method_id,
            "userId": user_id,
            "isActive": True,
        }
    )


async def _get_specific_payment_method(user_id: str, payment_method_id: str | None):
    if not payment_method_id:
        return None
    return await db.userpaymentmethod.find_first(
        where={
            "id": payment_method_id,
            "userId": user_id,
            "isActive": True,
        }
    )


async def _get_charge_candidate_methods(user_id: str, default_method_id: str | None) -> list[Any]:
    methods = await db.userpaymentmethod.find_many(
        where={
            "userId": user_id,
            "isActive": True,
        }
    )
    if not methods:
        return []

    ordered_methods = sorted(
        methods,
        key=lambda method: (
            0 if str(getattr(method, "id", "") or "") == str(default_method_id or "") else 1,
            0 if bool(getattr(method, "isDefault", False)) else 1,
            str(getattr(method, "id", "") or ""),
        ),
    )

    unique_methods: list[Any] = []
    seen_ids: set[str] = set()
    for method in ordered_methods:
        method_id = str(getattr(method, "id", "") or "")
        if not method_id or method_id in seen_ids:
            continue
        seen_ids.add(method_id)
        unique_methods.append(method)
    return unique_methods


async def _create_invoice_record(
    *,
    subscription_id: str,
    period_start: date,
    period_end: date,
    total_period_profit: float,
    calculated_fee: float,
    status: str,
    payment_method_used: str | None = None,
    stripe_payment_intent_id: str | None = None,
    paid_at: datetime | None = None,
):
    expected_payment_breakdown = build_expected_payment_breakdown(calculated_fee)
    return await db.invoice.create(
        data={
            "subId": subscription_id,
            "billingStartDate": _datetime_at_start(period_start),
            "billingEndDate": _datetime_at_start(period_end),
            "totalPeriodProfit": Decimal(str(round(total_period_profit, 2))),
            "calculatedFee": Decimal(str(round(calculated_fee, 2))),
            "status": status,
            "paymentMethodUsed": payment_method_used,
            "stripePaymentIntentId": stripe_payment_intent_id,
            "paymentBreakdown": _to_prisma_json(expected_payment_breakdown),
            "paidAt": paid_at,
        }
    )


def _invoice_charge_update_payload(
    charge_result: ChargeAttemptResult,
    *,
    payment_method_used: str | None,
    status: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status or charge_result.status,
        "paymentMethodUsed": payment_method_used,
        "stripePaymentIntentId": charge_result.payment_intent_id,
        "stripeChargeId": charge_result.charge_id,
        "stripeBalanceTxnId": charge_result.balance_transaction_id,
        "processorRequestId": charge_result.request_id,
        "paymentBreakdown": _to_prisma_json(charge_result.payment_breakdown),
        "paymentMethodDetails": _to_prisma_json(charge_result.payment_method_details),
        "paymentErrorDetails": _to_prisma_json(charge_result.payment_error_details),
        "paidAt": charge_result.paid_at,
    }
    return payload


def _escape_text(value: Any) -> str:
    return html.escape(str(value or "").strip())


def _trim_notification_text(value: Any, *, max_length: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 1].rstrip()}…"


def _format_display_date(value: Any, *, include_time: bool = False) -> str:
    if not value:
        return "-"
    if isinstance(value, datetime):
        current = value
        if current.tzinfo is not None:
            current = current.astimezone(timezone.utc)
        return current.strftime("%b %d, %Y %H:%M UTC" if include_time else "%b %d, %Y")
    if isinstance(value, date):
        return value.strftime("%b %d, %Y")
    return str(value)


def _format_period_display(start_value: date | None, end_value: date | None) -> str:
    if start_value and end_value:
        return f"{_format_display_date(start_value)} - {_format_display_date(end_value)}"
    if start_value:
        return _format_display_date(start_value)
    if end_value:
        return _format_display_date(end_value)
    return "-"


def _format_currency(amount: float, currency: str = "USD") -> str:
    return format_currency_amount(amount, currency)


def _invoice_label(invoice: Any) -> str:
    invoice_id = str(getattr(invoice, "id", "") or "").strip()
    return f"INV-{invoice_id[:8].upper()}" if invoice_id else "INV-UNKNOWN"


def _invoice_notification_reason(invoice: Any, note: str | None = None) -> str | None:
    explicit_note = str(note or "").strip()
    if explicit_note:
        return _trim_notification_text(explicit_note)

    payment_error_details = getattr(invoice, "paymentErrorDetails", None)
    if isinstance(payment_error_details, dict):
        message = str(payment_error_details.get("message") or "").strip()
        if message:
            return _trim_notification_text(message)

    return None


async def _load_notification_user(user: Any) -> Any | None:
    if not user:
        return None
    user_id = str(getattr(user, "id", "") or "").strip()
    if not user_id:
        return None

    notification_config = getattr(user, "notificationConfig", None)
    if notification_config is not None:
        return user

    return await db.user.find_unique(
        where={"id": user_id},
        include={"notificationConfig": True},
    )


async def _resolve_invoice_payment_method_for_notification(
    *,
    invoice: Any,
    subscription: Any | None,
    user_id: str,
) -> Any | None:
    payment_method_id = str(getattr(invoice, "paymentMethodUsed", "") or "").strip()
    if payment_method_id:
        payment_method = await db.userpaymentmethod.find_first(
            where={
                "id": payment_method_id,
                "userId": user_id,
            }
        )
        if payment_method:
            return payment_method

    default_method_id = str(getattr(subscription, "defaultPaymentMethodId", "") or "").strip() if subscription else ""
    if default_method_id:
        return await db.userpaymentmethod.find_first(
            where={
                "id": default_method_id,
                "userId": user_id,
            }
        )

    return await db.userpaymentmethod.find_first(
        where={
            "userId": user_id,
            "isDefault": True,
        }
    )


def _payment_method_label(payment_method: Any | None) -> str:
    if not payment_method:
        return "No saved card selected"
    brand = str(getattr(payment_method, "cardBrand", "") or "").strip().upper() or "CARD"
    last4 = str(getattr(payment_method, "cardLast4", "") or "").strip()
    if last4:
        return f"{brand} ending in {last4}"
    return brand


def _build_billing_notification_copy(
    *,
    event_type: str,
    invoice_label: str,
    invoice_amount: float,
    payment_amount: float,
    payment_currency: str,
    period_label: str,
    reason: str | None,
    collection_mode: str,
    source: str,
) -> tuple[str, str, str, str]:
    amount_label = _format_currency(payment_amount, payment_currency)
    manual_hint = "Manual collection is enabled. Please open billing and pay this invoice." if collection_mode == "manual" else "Review the billing page if you need to update your payment method."

    if event_type == "payment_received":
        title = "Billing payment received"
        subject = f"Payment received for {invoice_label} - SmarfRobotTrade"
        message = f"{invoice_label} was paid successfully. Collected {amount_label} for {period_label}."
        intro = "Your weekly billing payment was collected successfully."
        return title, subject, message, intro

    if event_type == "payment_failed":
        title = "Automatic billing failed" if source == "automatic" else "Billing payment failed"
        subject = f"Payment failed for {invoice_label} - SmarfRobotTrade"
        suffix = f" Reason: {reason}" if reason else ""
        message = (
            f"We could not collect {amount_label} for {invoice_label} ({period_label})."
            f"{suffix} {manual_hint}"
        ).strip()
        intro = "We could not complete your billing payment."
        return title, subject, message, intro

    if event_type == "invoice_skipped":
        title = "Billing cycle skipped"
        subject = f"Billing cycle skipped for {invoice_label} - SmarfRobotTrade"
        suffix = f" Reason: {reason}" if reason else ""
        message = f"{invoice_label} for {period_label} was skipped.{suffix}".strip()
        intro = "No charge will be collected for this billing period."
        return title, subject, message, intro

    title = "New billing invoice ready"
    subject = f"Billing invoice ready ({invoice_label}) - SmarfRobotTrade"
    suffix = f" {reason}" if reason else ""
    message = (
        f"{invoice_label} is ready for {period_label}. Amount due {amount_label}."
        f"{suffix} {manual_hint}"
    ).strip()
    intro = "A new weekly billing invoice is ready."
    return title, subject, message, intro


def _build_billing_notification_email_html(
    *,
    user: Any,
    subscription: Any | None,
    invoice: Any,
    payment_method: Any | None,
    title: str,
    intro: str,
    reason: str | None,
) -> str:
    invoice_label = _invoice_label(invoice)
    period_start = _extract_date(getattr(invoice, "billingStartDate", None))
    period_end = _extract_date(getattr(invoice, "billingEndDate", None))
    created_at = getattr(invoice, "createdAt", None)
    paid_at = getattr(invoice, "paidAt", None)
    status_key = str(_enum_value(getattr(invoice, "status", None)) or "pending").strip().lower()
    status_label = status_key.replace("_", " ").title()
    amount_details = get_invoice_payment_amount_details(invoice)
    invoice_amount = _to_float(getattr(invoice, "calculatedFee", None), 0.0)
    expected_payment_amount = _to_float(amount_details.get("expected_payment_amount"), 0.0)
    expected_payment_currency = str(amount_details.get("expected_payment_currency") or get_billing_currency()).upper()
    actual_payment_amount_raw = amount_details.get("actual_payment_amount")
    actual_payment_currency = str(amount_details.get("actual_payment_currency") or "").upper()
    payment_amount = _to_float(actual_payment_amount_raw, 0.0) if actual_payment_amount_raw is not None else expected_payment_amount
    payment_currency = actual_payment_currency or expected_payment_currency
    total_period_profit = _to_float(getattr(invoice, "totalPeriodProfit", None), 0.0)
    fee_type = str(_enum_value(getattr(subscription, "feeType", None)) or "percentage") if subscription else "percentage"
    fee_value = _to_float(getattr(subscription, "feeValue", None), 0.0) if subscription else 0.0
    collection_mode = str(_enum_value(getattr(subscription, "collectionMode", None)) or "automatic").title() if subscription else "Automatic"
    user_name = str(getattr(user, "username", "") or "").strip() or str(getattr(user, "email", "") or "").strip() or "Trader"
    payment_method_label = _format_invoice_payment_method_display(invoice, payment_method)
    if payment_method_label == "Not available" and status_key in {"pending", "failed"}:
        payment_method_label = "To be selected at checkout"

    status_palette = {
        "paid": ("#DCFCE7", "#166534"),
        "pending": ("#FEF3C7", "#B45309"),
        "failed": ("#FEE2E2", "#B91C1C"),
        "skipped": ("#E2E8F0", "#475569"),
    }
    status_bg, status_fg = status_palette.get(status_key, ("#F1F5F9", "#334155"))
    reason_block = ""
    if reason:
        reason_block = (
            '<div style="margin-top:20px; padding:14px 16px; border-radius:8px; background:#fff7ed; '
            'border:1px solid #fdba74;">'
            '<div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#9a3412; font-weight:700;">'
            'Details</div>'
            f'<div style="margin-top:8px; color:#7c2d12; line-height:1.6;">{_escape_text(reason)}</div>'
            "</div>"
        )

    fee_model_label = f"{fee_type.title()} at {fee_value:.2f}{'%' if fee_type == 'percentage' else ' USD'}"
    amount_title = "Amount Paid" if status_key == "paid" else "Amount Due"

    action_url = build_absolute_related_link(_BILLING_NOTIFICATION_RELATED_LINK) or _BILLING_NOTIFICATION_RELATED_LINK

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape_text(title)}</title>
</head>
<body style="margin:0; padding:24px; background:#f8fafc; font-family:Arial, sans-serif; color:#0f172a;">
  <div style="max-width:720px; margin:0 auto; background:#ffffff; border:1px solid #d7dee7; border-radius:10px; overflow:hidden;">
    <div style="padding:24px 28px; border-bottom:1px solid #d7dee7;">
      <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
        <tr>
          <td valign="top">
            <div style="font-size:11px; letter-spacing:0.1em; text-transform:uppercase; color:#64748b; font-weight:700;">SmarfRobotTrade</div>
            <div style="margin-top:6px; font-size:28px; font-weight:800; color:#0f172a;">{_escape_text(title)}</div>
            <div style="margin-top:6px; color:#475569; line-height:1.6;">{_escape_text(intro)}</div>
          </td>
          <td valign="top" align="right">
            <div style="font-size:18px; font-weight:800; color:#0f172a;">{_escape_text(invoice_label)}</div>
            <div style="margin-top:8px; display:inline-block; padding:4px 10px; border-radius:999px; background:{status_bg}; color:{status_fg}; font-size:12px; font-weight:700;">
              {_escape_text(status_label)}
            </div>
            <div style="margin-top:10px; font-size:12px; color:#64748b;">Issued {_escape_text(_format_display_date(created_at, include_time=True))}</div>
            <div style="margin-top:4px; font-size:12px; color:#64748b;">Paid {_escape_text(_format_display_date(paid_at, include_time=True))}</div>
          </td>
        </tr>
      </table>
    </div>

    <div style="padding:28px;">
      <div style="font-size:16px; color:#334155; line-height:1.7;">Hi {_escape_text(user_name)},</div>
      <div style="margin-top:8px; color:#475569; line-height:1.7;">{_escape_text(intro)}</div>

      <div style="margin-top:20px; font-size:13px; font-weight:700; color:#0f172a;">Invoice Details</div>
      <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin-top:8px; border:1px solid #d7dee7; border-radius:8px; overflow:hidden; border-collapse:separate; border-spacing:0;">
        <tr>
          <td style="width:180px; padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b;">Customer</td>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:14px; font-weight:700; color:#0f172a;">{_escape_text(user_name)}</td>
        </tr>
        <tr>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b;">Email</td>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:14px; color:#334155;">{_escape_text(getattr(user, "email", "") or "-")}</td>
        </tr>
        <tr>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b;">Billing period</td>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:14px; font-weight:700; color:#0f172a;">{_escape_text(_format_period_display(period_start, period_end))}</td>
        </tr>
        <tr>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b;">Payment method</td>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:14px; font-weight:700; color:#0f172a;">{_escape_text(payment_method_label)}</td>
        </tr>
        <tr>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b;">Charge currency</td>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:14px; color:#0f172a;">{_escape_text(payment_currency)}</td>
        </tr>
        <tr>
          <td style="padding:10px 14px; font-size:12px; font-weight:700; color:#64748b;">Fee model</td>
          <td style="padding:10px 14px; font-size:14px; color:#0f172a;">{_escape_text(fee_model_label)}</td>
        </tr>
      </table>

      <div style="margin-top:20px; font-size:13px; font-weight:700; color:#0f172a;">Charge Breakdown</div>
      <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin-top:8px; border:1px solid #d7dee7; border-radius:8px; overflow:hidden; border-collapse:separate; border-spacing:0;">
        <tr style="background:#f8fafc;">
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b;">Description</td>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b;">Reference</td>
          <td style="padding:10px 14px; border-bottom:1px solid #e2e8f0; font-size:12px; font-weight:700; color:#64748b; text-align:right;">Amount</td>
        </tr>
        <tr>
          <td style="padding:12px 14px; font-size:14px; color:#0f172a;">
            <div style="font-weight:700;">Weekly performance fee</div>
            <div style="margin-top:4px; color:#64748b;">{_escape_text(_format_period_display(period_start, period_end))}</div>
          </td>
          <td style="padding:12px 14px; font-size:14px; color:#334155;">Net profit {_escape_text(_format_currency(total_period_profit, get_billing_base_currency()))} at {_escape_text(f"{fee_value:.2f}{'%' if fee_type == 'percentage' else ' USD'}")}</td>
          <td style="padding:12px 14px; font-size:20px; font-weight:800; color:#0f172a; text-align:right; white-space:nowrap;">{_escape_text(_format_currency(payment_amount, payment_currency))}</td>
        </tr>
      </table>

      <div style="margin-top:20px; font-size:13px; font-weight:700; color:#0f172a;">Settlement</div>
      <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin-top:8px; border:1px solid #d7dee7; border-radius:8px; overflow:hidden; border-collapse:separate; border-spacing:0;">
        <tr>
          <td valign="top" width="52%" style="padding:16px; border-right:1px solid #e2e8f0;">
            <div style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase;">{_escape_text(amount_title)}</div>
            <div style="margin-top:8px; font-size:34px; line-height:1.1; font-weight:800; color:#0f172a;">{_escape_text(_format_currency(payment_amount, payment_currency))}</div>
            <div style="margin-top:8px; color:#334155;">Via {_escape_text(payment_method_label)}</div>
            <div style="margin-top:4px; color:#64748b;">Billing basis {_escape_text(_format_currency(invoice_amount, get_billing_base_currency()))}</div>
          </td>
          <td valign="top" width="48%" style="padding:16px;">
            <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="color:#0f172a;">
              <tr>
                <td style="padding:4px 0; font-size:12px; font-weight:700; color:#64748b;">Net profit</td>
                <td style="padding:4px 0; font-size:13px; font-weight:700; text-align:right;">{_escape_text(_format_currency(total_period_profit, get_billing_base_currency()))}</td>
              </tr>
              <tr>
                <td style="padding:4px 0; font-size:12px; font-weight:700; color:#64748b;">USD fee basis</td>
                <td style="padding:4px 0; font-size:13px; font-weight:700; text-align:right;">{_escape_text(_format_currency(invoice_amount, get_billing_base_currency()))}</td>
              </tr>
              <tr>
                <td style="padding:4px 0; font-size:12px; font-weight:700; color:#64748b;">Status</td>
                <td style="padding:4px 0; font-size:13px; font-weight:700; text-align:right;">{_escape_text(status_label)}</td>
              </tr>
              <tr>
                <td style="padding:4px 0; font-size:12px; font-weight:700; color:#64748b;">Paid at</td>
                <td style="padding:4px 0; font-size:13px; font-weight:700; text-align:right;">{_escape_text(_format_display_date(paid_at, include_time=True))}</td>
              </tr>
            </table>
          </td>
        </tr>
      </table>

      <div style="margin-top:20px; padding:16px; border:1px solid #e2e8f0; border-radius:8px; background:#ffffff;">
        <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:700;">Summary</div>
        <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin-top:10px; color:#0f172a;">
          <tr>
            <td style="padding:6px 0; color:#64748b;">Collection mode</td>
            <td style="padding:6px 0; text-align:right; font-weight:700;">{_escape_text(collection_mode)}</td>
          </tr>
          <tr>
            <td style="padding:6px 0; color:#64748b;">Open billing</td>
            <td style="padding:6px 0; text-align:right;"><a href="{_escape_text(action_url)}" style="color:#1d4ed8; text-decoration:none; font-weight:700;">Subscription &amp; Billing</a></td>
          </tr>
        </table>
      </div>

      {reason_block}

      <div style="margin-top:24px;">
        <a href="{_escape_text(action_url)}" style="display:inline-block; background:#0f172a; color:#ffffff; text-decoration:none; padding:11px 18px; border-radius:8px; font-weight:700;">
          Open Billing
        </a>
      </div>
    </div>
  </div>
</body>
</html>"""


async def notify_invoice_event(
    *,
    invoice: Any,
    user: Any,
    subscription: Any | None = None,
    event_type: str,
    note: str | None = None,
    source: str = "system",
    event_token: str | None = None,
) -> dict[str, bool]:
    try:
        notification_user = await _load_notification_user(user)
        if not notification_user:
            return {"in_app": False, "email": False, "discord": False}

        user_id = str(getattr(notification_user, "id", "") or "").strip()
        if not user_id:
            return {"in_app": False, "email": False, "discord": False}

        resolved_subscription = subscription
        if resolved_subscription is None:
            sub_id = str(getattr(invoice, "subId", "") or "").strip()
            if sub_id:
                resolved_subscription = await db.subscription.find_unique(where={"id": sub_id})

        payment_method = await _resolve_invoice_payment_method_for_notification(
            invoice=invoice,
            subscription=resolved_subscription,
            user_id=user_id,
        )
        invoice_label = _invoice_label(invoice)
        amount_details = get_invoice_payment_amount_details(invoice)
        amount = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
        actual_payment_amount = amount_details.get("actual_payment_amount")
        payment_amount = round(
            _to_float(
                actual_payment_amount if actual_payment_amount is not None else amount_details.get("expected_payment_amount"),
                0.0,
            ),
            2,
        )
        payment_currency = str(
            amount_details.get("actual_payment_currency")
            or amount_details.get("expected_payment_currency")
            or get_billing_currency()
        ).upper()
        period_start = _extract_date(getattr(invoice, "billingStartDate", None))
        period_end = _extract_date(getattr(invoice, "billingEndDate", None))
        period_label = _format_period_display(period_start, period_end)
        collection_mode = str(_enum_value(getattr(resolved_subscription, "collectionMode", None)) or "automatic").strip().lower()
        reason = _invoice_notification_reason(invoice, note)

        title, subject, message, intro = _build_billing_notification_copy(
            event_type=event_type,
            invoice_label=invoice_label,
            invoice_amount=amount,
            payment_amount=payment_amount,
            payment_currency=payment_currency,
            period_label=period_label,
            reason=reason,
            collection_mode=collection_mode,
            source=source,
        )

        dedupe_token = _trim_notification_text(
            event_token
            or getattr(invoice, "stripePaymentIntentId", None)
            or getattr(invoice, "paidAt", None)
            or getattr(invoice, "createdAt", None)
            or event_type,
            max_length=120,
        )
        if not claim_notification_dedupe(
            key=f"billing_notification:{str(getattr(invoice, 'id', '') or '')}:{event_type}:{dedupe_token}",
            ttl_seconds=_BILLING_NOTIFICATION_DEDUPE_TTL_SECONDS,
        ):
            return {"in_app": False, "email": False, "discord": False}

        email_html = _build_billing_notification_email_html(
            user=notification_user,
            subscription=resolved_subscription,
            invoice=invoice,
            payment_method=payment_method,
            title=title,
            intro=intro,
            reason=reason,
        )

        return await dispatch_notification_to_user(
            notification_user,
            title=title,
            message=message,
            related_link=_BILLING_NOTIFICATION_RELATED_LINK,
            email_subject=subject,
            email_html=email_html,
            action_label="Open billing",
            send_discord_channel=False,
        )
    except Exception:
        logger.exception(
            "failed to dispatch billing notification | invoice=%s event=%s",
            getattr(invoice, "id", None),
            event_type,
        )
        return {"in_app": False, "email": False, "discord": False}


async def _create_or_get_invoice_record(
    *,
    subscription_id: str,
    period_start: date,
    period_end: date,
    total_period_profit: float,
    calculated_fee: float,
    status: str,
    payment_method_used: str | None = None,
    stripe_payment_intent_id: str | None = None,
    paid_at: datetime | None = None,
) -> tuple[Any, bool]:
    try:
        invoice = await _create_invoice_record(
            subscription_id=subscription_id,
            period_start=period_start,
            period_end=period_end,
            total_period_profit=total_period_profit,
            calculated_fee=calculated_fee,
            status=status,
            payment_method_used=payment_method_used,
            stripe_payment_intent_id=stripe_payment_intent_id,
            paid_at=paid_at,
        )
        return (invoice, True)
    except Exception as error:
        if UniqueViolationError is None or not isinstance(error, UniqueViolationError):
            raise

        existing_invoice = await _find_existing_invoice(subscription_id, period_start, period_end)
        if existing_invoice:
            return (existing_invoice, False)
        raise


def _build_payment_breakdown(payment_intent: Any) -> dict[str, Any] | None:
    if not payment_intent:
        return None

    latest_charge = _sanitize_json(payment_intent.get("latest_charge"))
    balance_transaction = None
    if isinstance(latest_charge, dict):
        balance_transaction = _sanitize_json(latest_charge.get("balance_transaction"))
    if not isinstance(balance_transaction, dict):
        balance_transaction = None

    payment_currency = str(payment_intent.get("currency") or "").upper() or None
    settlement_currency = str((balance_transaction or {}).get("currency") or "").upper() or None

    breakdown = _strip_none_values(
        {
            "payment_amount_minor": _to_int(payment_intent.get("amount"), None),
            "payment_amount": _minor_to_major(payment_intent.get("amount"), payment_currency),
            "payment_currency": payment_currency,
            "amount_received_minor": _to_int(payment_intent.get("amount_received"), None),
            "amount_received": _minor_to_major(payment_intent.get("amount_received"), payment_currency),
            "settlement_amount_minor": _to_int((balance_transaction or {}).get("amount"), None),
            "settlement_amount": _minor_to_major((balance_transaction or {}).get("amount"), settlement_currency),
            "settlement_currency": settlement_currency,
            "exchange_rate": _to_float((balance_transaction or {}).get("exchange_rate"), 0.0)
            if (balance_transaction or {}).get("exchange_rate") is not None
            else None,
            "fee_minor": _to_int((balance_transaction or {}).get("fee"), None),
            "fee_amount": _minor_to_major((balance_transaction or {}).get("fee"), settlement_currency),
            "net_minor": _to_int((balance_transaction or {}).get("net"), None),
            "net_amount": _minor_to_major((balance_transaction or {}).get("net"), settlement_currency),
            "available_on": _sanitize_json((balance_transaction or {}).get("available_on")),
            "reporting_category": (balance_transaction or {}).get("reporting_category"),
        }
    ) or {}

    presentment_breakdown = _extract_presentment_breakdown(payment_intent)
    if presentment_breakdown:
        breakdown.update(presentment_breakdown)

    return breakdown or None


def _build_payment_method_details(
    *,
    local_payment_method: Any | None,
    stripe_payment_method: Any | None = None,
) -> dict[str, Any] | None:
    payment_method = _sanitize_json(stripe_payment_method) if stripe_payment_method else {}
    if not isinstance(payment_method, dict):
        payment_method = {}

    card = payment_method.get("card") or {}
    if not isinstance(card, dict):
        card = {}

    billing_details = payment_method.get("billing_details") or {}
    if not isinstance(billing_details, dict):
        billing_details = {}

    address = billing_details.get("address") or {}
    if not isinstance(address, dict):
        address = {}

    local_brand = str(getattr(local_payment_method, "cardBrand", "") or "").strip() or None
    local_last4 = str(getattr(local_payment_method, "cardLast4", "") or "").strip() or None
    local_type = str(getattr(local_payment_method, "type", "") or "").strip() or None

    return _strip_none_values(
        {
            "local_method_id": str(getattr(local_payment_method, "id", "") or "").strip() or None,
            "provider_method_id": payment_method.get("id")
            or str(getattr(local_payment_method, "providerMethodId", "") or "").strip()
            or None,
            "type": payment_method.get("type") or local_type,
            "brand": card.get("brand") or local_brand,
            "funding": card.get("funding"),
            "last4": card.get("last4") or local_last4,
            "exp_month": card.get("exp_month") or getattr(local_payment_method, "expiryMonth", None),
            "exp_year": card.get("exp_year") or getattr(local_payment_method, "expiryYear", None),
            "fingerprint": card.get("fingerprint"),
            "country": card.get("country"),
            "issuer": card.get("issuer"),
            "wallet": _sanitize_json(card.get("wallet")),
            "billing_name": billing_details.get("name"),
            "billing_email": billing_details.get("email"),
            "billing_phone": billing_details.get("phone"),
            "billing_address": _strip_none_values(
                {
                    "line1": address.get("line1"),
                    "line2": address.get("line2"),
                    "city": address.get("city"),
                    "state": address.get("state"),
                    "postal_code": address.get("postal_code"),
                    "country": address.get("country"),
                }
            ),
        }
    )


def _build_payment_error_details(error: Any | None, *, payment_intent: Any | None = None) -> dict[str, Any] | None:
    last_payment_error = _sanitize_json(payment_intent.get("last_payment_error")) if payment_intent else None
    if last_payment_error is not None and not isinstance(last_payment_error, dict):
        last_payment_error = {"raw": last_payment_error}

    raw_error = _sanitize_json(getattr(error, "json_body", None))
    error_payload = _sanitize_json(getattr(error, "error", None))
    if error_payload is not None and not isinstance(error_payload, dict):
        error_payload = {"raw": error_payload}

    detail = _strip_none_values(
        {
            "message": getattr(error, "user_message", None)
            or getattr(error, "message", None)
            or (last_payment_error or {}).get("message"),
            "code": getattr(error, "code", None) or (last_payment_error or {}).get("code"),
            "type": getattr(error, "type", None) or (last_payment_error or {}).get("type"),
            "param": getattr(error, "param", None) or (last_payment_error or {}).get("param"),
            "decline_code": (last_payment_error or {}).get("decline_code"),
            "request_id": _extract_request_id(error) or _extract_request_id(payment_intent),
            "request_log_url": getattr(error, "request_log_url", None),
            "charge": (last_payment_error or {}).get("charge"),
            "payment_method_type": (last_payment_error or {}).get("payment_method_type"),
            "raw_error": raw_error,
            "error_payload": error_payload,
            "last_payment_error": last_payment_error,
        }
    )
    return detail


def _build_charge_attempt_summary(
    *,
    attempt_number: int,
    local_payment_method: Any,
    charge_result: ChargeAttemptResult,
) -> dict[str, Any] | None:
    method_details = charge_result.payment_method_details if isinstance(charge_result.payment_method_details, dict) else {}
    error_details = charge_result.payment_error_details if isinstance(charge_result.payment_error_details, dict) else {}

    return _strip_none_values(
        {
            "attempt_number": attempt_number,
            "local_method_id": charge_result.local_payment_method_id
            or str(getattr(local_payment_method, "id", "") or "").strip()
            or None,
            "provider_method_id": charge_result.provider_payment_method_id
            or str(getattr(local_payment_method, "providerMethodId", "") or "").strip()
            or None,
            "status": charge_result.status,
            "note": charge_result.note,
            "request_id": charge_result.request_id,
            "payment_intent_id": charge_result.payment_intent_id,
            "charge_id": charge_result.charge_id,
            "brand": method_details.get("brand") or str(getattr(local_payment_method, "cardBrand", "") or "").strip() or None,
            "last4": method_details.get("last4") or str(getattr(local_payment_method, "cardLast4", "") or "").strip() or None,
            "funding": method_details.get("funding"),
            "error_message": error_details.get("message"),
            "error_code": error_details.get("code"),
            "decline_code": error_details.get("decline_code"),
        }
    )


def _attach_attempt_history(
    charge_result: ChargeAttemptResult,
    attempts: list[dict[str, Any] | None],
) -> ChargeAttemptResult:
    attempts_payload = [attempt for attempt in attempts if attempt]
    if not attempts_payload:
        return charge_result

    method_details = (
        dict(charge_result.payment_method_details)
        if isinstance(charge_result.payment_method_details, dict)
        else {}
    )
    method_details["attempts"] = attempts_payload
    charge_result.payment_method_details = method_details

    if charge_result.status == "failed" or charge_result.payment_error_details:
        error_details = (
            dict(charge_result.payment_error_details)
            if isinstance(charge_result.payment_error_details, dict)
            else {}
        )
        error_details["attempts"] = attempts_payload
        charge_result.payment_error_details = error_details

    return charge_result


async def _charge_invoice(
    *,
    user: Any,
    subscription: Any,
    local_payment_method: Any,
    amount: float,
    period_start: date,
    period_end: date,
) -> ChargeAttemptResult:
    local_method_id = str(getattr(local_payment_method, "id", "") or "").strip() or None
    provider_method_id = str(getattr(local_payment_method, "providerMethodId", "") or "").strip() or None

    if amount <= 0:
        return ChargeAttemptResult(
            status="skipped",
            payment_intent_id=None,
            paid_at=None,
            note="No billable fee for this period",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
        )

    if not _stripe_enabled():
        return ChargeAttemptResult(
            status="pending",
            payment_intent_id=None,
            paid_at=None,
            note="Stripe is not configured on the server",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
        )

    if not provider_method_id:
        return ChargeAttemptResult(
            status="failed",
            payment_intent_id=None,
            paid_at=None,
            note="Default payment method is missing Stripe provider id",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
            payment_method_details=_build_payment_method_details(local_payment_method=local_payment_method),
        )

    stripe_customer_id = str(getattr(user, "stripeCustomerId", "") or "").strip()
    if not stripe_customer_id:
        return ChargeAttemptResult(
            status="pending",
            payment_intent_id=None,
            paid_at=None,
            note="User is missing Stripe customer id",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
            payment_method_details=_build_payment_method_details(local_payment_method=local_payment_method),
        )

    billing_currency = get_billing_currency().lower()
    amount_minor = max(0, billing_amount_to_minor_units(amount, require_config=True))
    if amount_minor <= 0:
        return ChargeAttemptResult(
            status="skipped",
            payment_intent_id=None,
            paid_at=None,
            note="Calculated fee rounded down to zero",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
        )

    try:
        payment_intent = await _stripe_call(
            stripe.PaymentIntent.create,
            amount=amount_minor,
            currency=billing_currency,
            customer=stripe_customer_id,
            payment_method=provider_method_id,
            confirm=True,
            off_session=True,
            description=f"SmarfRobotTrade weekly billing {period_start.isoformat()} to {period_end.isoformat()}",
            metadata={
                "subscription_id": str(getattr(subscription, "id", "") or ""),
                "user_id": str(getattr(user, "id", "") or ""),
                "billing_start_date": period_start.isoformat(),
                "billing_end_date": period_end.isoformat(),
                "invoice_currency": get_billing_base_currency(),
                "invoice_amount": f"{round(amount, 2):.2f}",
                "charge_currency": get_billing_currency(),
                "configured_exchange_rate": f"{get_billing_exchange_rate(require_config=True):.6f}",
            },
            expand=["latest_charge.balance_transaction", "payment_method"],
        )
    except Exception as error:
        return ChargeAttemptResult(
            status="failed",
            payment_intent_id=None,
            paid_at=None,
            note=str(error),
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
            request_id=_extract_request_id(error),
            payment_method_details=_build_payment_method_details(local_payment_method=local_payment_method),
            payment_error_details=_build_payment_error_details(error),
        )

    payment_status = str(payment_intent.get("status") or "").strip().lower()
    payment_intent_id = payment_intent.get("id")
    latest_charge = payment_intent.get("latest_charge")
    latest_charge_data = _sanitize_json(latest_charge) if latest_charge else {}
    if not isinstance(latest_charge_data, dict):
        latest_charge_data = {}
    balance_transaction_data = latest_charge_data.get("balance_transaction") or {}
    if not isinstance(balance_transaction_data, dict):
        balance_transaction_data = {}
    charge_id = latest_charge_data.get("id")
    balance_transaction_id = balance_transaction_data.get("id")
    request_id = _extract_request_id(payment_intent)
    payment_breakdown = _build_payment_breakdown(payment_intent)
    payment_breakdown = merge_payment_breakdown_with_expected_amount(
        payment_breakdown,
        invoice_amount_usd=amount,
    )
    payment_method_details = _build_payment_method_details(
        local_payment_method=local_payment_method,
        stripe_payment_method=payment_intent.get("payment_method"),
    )
    error_details = _build_payment_error_details(None, payment_intent=payment_intent)

    if payment_status == "succeeded":
        return ChargeAttemptResult(
            status="paid",
            payment_intent_id=payment_intent_id,
            paid_at=datetime.now(timezone.utc),
            note="Charge succeeded",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
            request_id=request_id,
            charge_id=charge_id,
            balance_transaction_id=balance_transaction_id,
            payment_breakdown=payment_breakdown,
            payment_method_details=payment_method_details,
            payment_error_details=error_details,
        )

    if payment_status in {"processing", "requires_capture"}:
        return ChargeAttemptResult(
            status="pending",
            payment_intent_id=payment_intent_id,
            paid_at=None,
            note=f"Stripe payment intent status: {payment_status}",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
            request_id=request_id,
            charge_id=charge_id,
            balance_transaction_id=balance_transaction_id,
            payment_breakdown=payment_breakdown,
            payment_method_details=payment_method_details,
            payment_error_details=error_details,
        )

    if payment_status in {"requires_payment_method", "requires_action", "canceled"}:
        return ChargeAttemptResult(
            status="failed",
            payment_intent_id=payment_intent_id,
            paid_at=None,
            note=f"Stripe payment intent status: {payment_status}",
            local_payment_method_id=local_method_id,
            provider_payment_method_id=provider_method_id,
            request_id=request_id,
            charge_id=charge_id,
            balance_transaction_id=balance_transaction_id,
            payment_breakdown=payment_breakdown,
            payment_method_details=payment_method_details,
            payment_error_details=error_details,
        )

    return ChargeAttemptResult(
        status="pending",
        payment_intent_id=payment_intent_id,
        paid_at=None,
        note=f"Unhandled Stripe payment intent status: {payment_status or 'unknown'}",
        local_payment_method_id=local_method_id,
        provider_payment_method_id=provider_method_id,
        request_id=request_id,
        charge_id=charge_id,
        balance_transaction_id=balance_transaction_id,
        payment_breakdown=payment_breakdown,
        payment_method_details=payment_method_details,
        payment_error_details=error_details,
    )


async def _charge_invoice_with_fallback(
    *,
    user: Any,
    subscription: Any,
    payment_methods: list[Any],
    amount: float,
    period_start: date,
    period_end: date,
) -> ChargeAttemptResult:
    if not payment_methods:
        return ChargeAttemptResult(
            status="pending",
            payment_intent_id=None,
            paid_at=None,
            note="No active payment method available",
        )

    attempts: list[dict[str, Any] | None] = []
    last_result: ChargeAttemptResult | None = None

    for index, local_payment_method in enumerate(payment_methods, start=1):
        charge_result = await _charge_invoice(
            user=user,
            subscription=subscription,
            local_payment_method=local_payment_method,
            amount=amount,
            period_start=period_start,
            period_end=period_end,
        )
        attempts.append(
            _build_charge_attempt_summary(
                attempt_number=index,
                local_payment_method=local_payment_method,
                charge_result=charge_result,
            )
        )
        last_result = charge_result

        if charge_result.status in {"paid", "pending", "skipped"}:
            return _attach_attempt_history(charge_result, attempts)

    if last_result is None:
        return ChargeAttemptResult(
            status="pending",
            payment_intent_id=None,
            paid_at=None,
            note="No payment attempt was executed",
        )

    if len(attempts) > 1:
        last_result.note = f"All payment methods failed. Last error: {last_result.note}"

    return _attach_attempt_history(last_result, attempts)


async def process_due_subscription(subscription: Any, *, today: date | None = None) -> list[BillingCycleResult]:
    today = today or date.today()
    subscription_status = str(_enum_value(getattr(subscription, "status", None)) or "active").lower()
    if subscription_status not in _PROCESSABLE_SUB_STATUSES:
        return []

    subscription, due_date = await normalize_subscription_next_billing_date(subscription)
    if not due_date or due_date > today:
        return []

    user = getattr(subscription, "user", None)
    if not user:
        user = await db.user.find_unique(where={"id": str(subscription.userId)})
    if not user:
        return []

    user_id = str(getattr(user, "id", "") or "")
    if not user_id:
        return []

    results: list[BillingCycleResult] = []
    next_due_date = due_date

    for _ in range(104):
        if next_due_date > today:
            break

        current_cycle_due_date = next_due_date
        period_start, period_end = billing_period_for_due_date(current_cycle_due_date)
        existing_invoice = await _find_existing_invoice(str(subscription.id), period_start, period_end)
        if existing_invoice:
            existing_status = str(_enum_value(getattr(existing_invoice, "status", None)) or "pending").lower()
            if existing_status in {"pending", "failed"}:
                existing_invoice = await reconcile_open_invoice_amount(
                    existing_invoice,
                    subscription=subscription,
                    user_id=user_id,
                )
                existing_status = str(_enum_value(getattr(existing_invoice, "status", None)) or existing_status).lower()
            if existing_status in {"pending", "failed"}:
                await sync_subscription_status_from_invoices(str(subscription.id))
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(existing_invoice.id),
                    invoice_created=False,
                    status=existing_status,
                    amount=round(_to_float(existing_invoice.calculatedFee), 2),
                    period_start=period_start,
                    period_end=period_end,
                    note="Existing invoice reused",
                )
            )
            next_due_date = current_cycle_due_date + timedelta(days=7)
            if existing_status in {"paid", "skipped"}:
                continue
            break

        period_daily_aggregates = await _get_unbilled_period_daily_aggregates(user_id, period_start, period_end)
        net_profit = _sum_daily_aggregate_profit(period_daily_aggregates)
        fee_type = str(_enum_value(getattr(subscription, "feeType", None)) or "percentage")
        collection_mode = str(_enum_value(getattr(subscription, "collectionMode", None)) or "automatic").strip().lower()
        fee_value = _to_float(getattr(subscription, "feeValue", None), 0.0)
        min_profit_threshold = _to_float(getattr(subscription, "minProfitThreshold", None), 0.0)
        calculated_fee = round(
            calculate_fee(
                net_profit=net_profit,
                fee_type=fee_type,
                fee_value=fee_value,
                min_profit_threshold=min_profit_threshold,
            ),
            2,
        )

        charge_candidate_methods = await _get_charge_candidate_methods(
            user_id=user_id,
            default_method_id=str(getattr(subscription, "defaultPaymentMethodId", "") or "") or None,
        )
        local_payment_method = charge_candidate_methods[0] if charge_candidate_methods else None

        if calculated_fee <= 0:
            invoice, invoice_created = await _create_or_get_invoice_record(
                subscription_id=str(subscription.id),
                period_start=period_start,
                period_end=period_end,
                total_period_profit=net_profit,
                calculated_fee=0.0,
                status="skipped",
                payment_method_used=str(getattr(local_payment_method, "id", "") or "") or None,
            )
            if invoice_created:
                await reserve_daily_aggregates_for_invoice(
                    str(invoice.id),
                    period_daily_aggregates,
                    status="skipped",
                )
            invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "skipped")
            invoice_amount = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(invoice.id),
                    invoice_created=invoice_created,
                    status=invoice_status,
                    amount=invoice_amount,
                    period_start=period_start,
                    period_end=period_end,
                    note="Net profit did not exceed billing threshold" if invoice_created else "Existing invoice reused",
                )
            )
            next_due_date = current_cycle_due_date + timedelta(days=7)
            continue

        if collection_mode == "manual":
            invoice, invoice_created = await _create_or_get_invoice_record(
                subscription_id=str(subscription.id),
                period_start=period_start,
                period_end=period_end,
                total_period_profit=net_profit,
                calculated_fee=calculated_fee,
                status="pending",
            )
            if invoice_created:
                await reserve_daily_aggregates_for_invoice(
                    str(invoice.id),
                    period_daily_aggregates,
                    status="pending",
                )
            invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "pending")
            invoice_amount = round(_to_float(getattr(invoice, "calculatedFee", None), calculated_fee), 2)
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(invoice.id),
                    invoice_created=invoice_created,
                    status=invoice_status,
                    amount=invoice_amount,
                    period_start=period_start,
                    period_end=period_end,
                    note="Manual collection mode requires customer payment" if invoice_created else "Existing invoice reused",
                )
            )
            await sync_subscription_status_from_invoices(str(subscription.id))
            if invoice_created:
                await notify_invoice_event(
                    invoice=invoice,
                    user=user,
                    subscription=subscription,
                    event_type="invoice_ready",
                    note="Manual collection mode requires customer payment before bot usage resumes.",
                    source="automatic",
                    event_token="created-manual",
                )
            next_due_date = current_cycle_due_date + timedelta(days=7)
            break

        if not charge_candidate_methods:
            invoice, invoice_created = await _create_or_get_invoice_record(
                subscription_id=str(subscription.id),
                period_start=period_start,
                period_end=period_end,
                total_period_profit=net_profit,
                calculated_fee=calculated_fee,
                status="pending",
            )
            if invoice_created:
                await reserve_daily_aggregates_for_invoice(
                    str(invoice.id),
                    period_daily_aggregates,
                    status="pending",
                )
            invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "pending")
            invoice_amount = round(_to_float(getattr(invoice, "calculatedFee", None), calculated_fee), 2)
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(invoice.id),
                    invoice_created=invoice_created,
                    status=invoice_status,
                    amount=invoice_amount,
                    period_start=period_start,
                    period_end=period_end,
                    note="No active payment method available" if invoice_created else "Existing invoice reused",
                )
            )
            await sync_subscription_status_from_invoices(str(subscription.id))
            if invoice_created:
                await notify_invoice_event(
                    invoice=invoice,
                    user=user,
                    subscription=subscription,
                    event_type="invoice_ready",
                    note="Automatic billing could not start because no active payment method is available.",
                    source="automatic",
                    event_token="created-no-payment-method",
                )
            next_due_date = current_cycle_due_date + timedelta(days=7)
            break

        invoice, invoice_created = await _create_or_get_invoice_record(
            subscription_id=str(subscription.id),
            period_start=period_start,
            period_end=period_end,
            total_period_profit=net_profit,
            calculated_fee=calculated_fee,
            status="pending",
            payment_method_used=str(local_payment_method.id),
        )
        if invoice_created:
            await reserve_daily_aggregates_for_invoice(
                str(invoice.id),
                period_daily_aggregates,
                status="pending",
            )
        if not invoice_created:
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(invoice.id),
                    invoice_created=False,
                    status=str(_enum_value(getattr(invoice, "status", None)) or "pending"),
                    amount=round(_to_float(getattr(invoice, "calculatedFee", None), calculated_fee), 2),
                    period_start=period_start,
                    period_end=period_end,
                    note="Existing invoice reused",
                )
            )
            next_due_date = current_cycle_due_date + timedelta(days=7)
            if str(_enum_value(getattr(invoice, "status", None)) or "pending").lower() in {"paid", "skipped"}:
                continue
            break

        charge_result = await _charge_invoice_with_fallback(
            user=user,
            subscription=subscription,
            payment_methods=charge_candidate_methods,
            amount=calculated_fee,
            period_start=period_start,
            period_end=period_end,
        )
        invoice = await db.invoice.update(
            where={"id": str(invoice.id)},
            data=_invoice_charge_update_payload(
                charge_result,
                payment_method_used=charge_result.local_payment_method_id or str(local_payment_method.id),
            ),
        )
        await sync_daily_aggregate_status_for_invoice(str(invoice.id), getattr(invoice, "status", None) or charge_result.status)
        await sync_subscription_status_from_invoices(str(subscription.id))
        invoice_status = str(_enum_value(getattr(invoice, "status", None)) or charge_result.status).lower()
        if invoice_status == "paid":
            await notify_invoice_event(
                invoice=invoice,
                user=user,
                subscription=subscription,
                event_type="payment_received",
                note=charge_result.note,
                source="automatic",
                event_token=charge_result.payment_intent_id or charge_result.request_id or "auto-paid",
            )
        elif invoice_status == "failed":
            await notify_invoice_event(
                invoice=invoice,
                user=user,
                subscription=subscription,
                event_type="payment_failed",
                note=charge_result.note,
                source="automatic",
                event_token=charge_result.payment_intent_id or charge_result.request_id or charge_result.note or "auto-failed",
            )
        elif invoice_status == "pending":
            await notify_invoice_event(
                invoice=invoice,
                user=user,
                subscription=subscription,
                event_type="invoice_ready",
                note=charge_result.note,
                source="automatic",
                event_token=charge_result.payment_intent_id or charge_result.request_id or "auto-pending",
            )
        results.append(
            BillingCycleResult(
                subscription_id=str(subscription.id),
                invoice_id=str(invoice.id),
                invoice_created=True,
                status=invoice_status,
                amount=round(_to_float(getattr(invoice, "calculatedFee", None), calculated_fee), 2),
                period_start=period_start,
                period_end=period_end,
                note=charge_result.note,
            )
        )
        next_due_date = current_cycle_due_date + timedelta(days=7)
        if invoice_status in {"paid", "skipped"}:
            continue
        break

    if next_due_date != due_date:
        await db.subscription.update(
            where={"id": str(subscription.id)},
            data={"nextBillingDate": _datetime_at_start(next_due_date)},
        )

    return results


def summarize_billing_results(results: list[BillingCycleResult]) -> BillingRunSummary:
    summary = BillingRunSummary()
    touched_subscriptions: set[str] = set()

    for result in results:
        touched_subscriptions.add(result.subscription_id)
        if result.invoice_created:
            summary.created_invoices += 1

        if result.status == "paid":
            summary.paid_invoices += 1
        elif result.status == "pending":
            summary.pending_invoices += 1
        elif result.status == "skipped":
            summary.skipped_invoices += 1
        elif result.status == "failed":
            summary.failed_invoices += 1

    summary.processed_subscriptions = len(touched_subscriptions)
    return summary


async def process_all_due_billing(*, today: date | None = None, user_id: str | None = None) -> BillingRunSummary:
    today = today or date.today()

    where_clause: dict[str, Any] = {
        "status": {"in": list(_PROCESSABLE_SUB_STATUSES)},
        "nextBillingDate": {"lte": _datetime_at_end(today)},
    }
    if user_id:
        where_clause["userId"] = user_id

    subscriptions = await db.subscription.find_many(
        where=where_clause,
        include={"user": True},
        order={"nextBillingDate": "asc"},
    )

    all_results: list[BillingCycleResult] = []
    for subscription in subscriptions:
        try:
            all_results.extend(await process_due_subscription(subscription, today=today))
        except Exception:
            logger.exception("subscription billing failed for subscription %s", getattr(subscription, "id", None))

    return summarize_billing_results(all_results)


async def pay_invoice_now(
    invoice: Any,
    *,
    user: Any,
    selected_payment_method_id: str | None = None,
) -> Any:
    invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "").lower()
    if invoice_status == "paid":
        return invoice

    subscription = getattr(invoice, "subscription", None)
    if not subscription:
        subscription = await db.subscription.find_unique(where={"id": str(invoice.subId)})
    if not subscription:
        raise ValueError("Subscription not found for invoice")

    user_id = str(getattr(user, "id", "") or "")
    selected_method_id = str(selected_payment_method_id or "").strip() or None
    if selected_method_id:
        selected_method = await _get_specific_payment_method(user_id, selected_method_id)
        if not selected_method:
            raise ValueError("Selected payment method is not available")
        charge_candidate_methods = [selected_method]
    else:
        charge_candidate_methods = await _get_charge_candidate_methods(
            user_id=user_id,
            default_method_id=str(getattr(subscription, "defaultPaymentMethodId", "") or "") or None,
        )
    local_payment_method = charge_candidate_methods[0] if charge_candidate_methods else None
    if not charge_candidate_methods:
        raise ValueError("No active payment method available")

    amount = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
    period_start = _extract_date(getattr(invoice, "billingStartDate", None))
    period_end = _extract_date(getattr(invoice, "billingEndDate", None))
    if not period_start or not period_end:
        raise ValueError("Invoice billing period is incomplete")

    charge_result = await _charge_invoice_with_fallback(
        user=user,
        subscription=subscription,
        payment_methods=charge_candidate_methods,
        amount=amount,
        period_start=period_start,
        period_end=period_end,
    )
    if charge_result.status != "paid":
        updated_invoice = await db.invoice.update(
            where={"id": str(invoice.id)},
            data=_invoice_charge_update_payload(
                charge_result,
                payment_method_used=charge_result.local_payment_method_id or str(local_payment_method.id),
                status="failed" if charge_result.status == "failed" else "pending",
            ),
        )
        await sync_daily_aggregate_status_for_invoice(
            str(invoice.id),
            "failed" if charge_result.status == "failed" else "pending",
        )
        await sync_subscription_status_from_invoices(str(subscription.id))
        await notify_invoice_event(
            invoice=updated_invoice,
            user=user,
            subscription=subscription,
            event_type="payment_failed" if charge_result.status == "failed" else "invoice_ready",
            note=charge_result.note,
            source="manual",
            event_token=charge_result.payment_intent_id or charge_result.request_id or charge_result.note or "manual-attempt",
        )
        raise ValueError(charge_result.note or "Unable to collect payment")

    updated_invoice = await db.invoice.update(
        where={"id": str(invoice.id)},
        data=_invoice_charge_update_payload(
            charge_result,
            payment_method_used=charge_result.local_payment_method_id or str(local_payment_method.id),
            status="paid",
        ),
    )
    await sync_daily_aggregate_status_for_invoice(str(updated_invoice.id), getattr(updated_invoice, "status", None) or "paid")
    await sync_subscription_status_from_invoices(str(subscription.id))
    await notify_invoice_event(
        invoice=updated_invoice,
        user=user,
        subscription=subscription,
        event_type="payment_received",
        note=charge_result.note,
        source="manual",
        event_token=charge_result.payment_intent_id or charge_result.request_id or getattr(updated_invoice, "paidAt", None) or "manual-paid",
    )
    return updated_invoice


def _safe_text(value: Any) -> str:
    return html.escape(str(value or "").strip())


def _format_invoice_payment_method_display(invoice: Any, payment_method: Any | None = None) -> str:
    details = getattr(invoice, "paymentMethodDetails", None)
    payment_details = details if isinstance(details, dict) else {}

    payment_type = str(payment_details.get("type") or "").strip().lower()
    brand = str(payment_details.get("brand") or getattr(payment_method, "cardBrand", "") or "").strip().upper()
    last4 = str(payment_details.get("last4") or getattr(payment_method, "cardLast4", "") or "").strip()
    wallet = payment_details.get("wallet") if isinstance(payment_details.get("wallet"), dict) else {}
    wallet_type = str(wallet.get("type") or "").strip().lower()

    if payment_type == "promptpay":
        return "PromptPay"

    if payment_type == "card" or brand or last4:
        card_label = brand or "Card"
        if last4:
            card_label = f"{card_label} ending in {last4}"
        if wallet_type == "apple_pay":
            return f"Apple Pay ({card_label})"
        if wallet_type == "google_pay":
            return f"Google Pay ({card_label})"
        return card_label

    if payment_type:
        return payment_type.replace("_", " ").title()

    return "Not available"


def build_invoice_pdf(invoice: Any, *, user: Any, subscription: Any, payment_method: Any | None = None) -> bytes:
    try:
        from reportlab.lib.colors import HexColor
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfgen import canvas
    except Exception as error:
        raise RuntimeError("reportlab is not installed on the server") from error

    def format_datetime_display(value: Any, *, include_time: bool = False) -> str:
        if not value:
            return "-"
        if isinstance(value, datetime):
            current = value
            if current.tzinfo is not None:
                current = current.astimezone(timezone.utc)
            return current.strftime("%b %d, %Y %H:%M UTC" if include_time else "%b %d, %Y")
        if isinstance(value, date):
            return value.strftime("%b %d, %Y")
        return str(value)

    def format_period_display(start_value: date | None, end_value: date | None) -> str:
        if start_value and end_value:
            return f"{format_datetime_display(start_value)} - {format_datetime_display(end_value)}"
        if start_value:
            return format_datetime_display(start_value)
        if end_value:
            return format_datetime_display(end_value)
        return "-"

    def format_currency(amount: float, currency: str = "USD") -> str:
        return format_currency_amount(amount, currency)

    def wrap_text(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return ["-"]

        words = raw_text.split()
        if not words:
            return ["-"]

        lines: list[str] = []
        current_line = ""

        def width_of(value: str) -> float:
            return pdfmetrics.stringWidth(value, font_name, font_size)

        def split_long_token(token: str) -> list[str]:
            chunks: list[str] = []
            current_chunk = ""
            for char in token:
                candidate = f"{current_chunk}{char}"
                if current_chunk and width_of(candidate) > max_width:
                    chunks.append(current_chunk)
                    current_chunk = char
                else:
                    current_chunk = candidate
            if current_chunk:
                chunks.append(current_chunk)
            return chunks or [token]

        expanded_words: list[str] = []
        for word in words:
            if width_of(word) <= max_width:
                expanded_words.append(word)
                continue
            expanded_words.extend(split_long_token(word))

        for word in expanded_words:
            candidate = word if not current_line else f"{current_line} {word}"
            if current_line and width_of(candidate) > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = candidate

        if current_line:
            lines.append(current_line)

        return lines or ["-"]

    invoice_id = str(getattr(invoice, "id", "") or "")
    invoice_label = f"INV-{invoice_id[:8].upper()}" if invoice_id else "INV-UNKNOWN"
    status_key = str(_enum_value(getattr(invoice, "status", None)) or "pending").strip().lower()
    status = status_key.replace("_", " ").title()
    period_start = _extract_date(getattr(invoice, "billingStartDate", None))
    period_end = _extract_date(getattr(invoice, "billingEndDate", None))
    paid_at = getattr(invoice, "paidAt", None)
    created_at = getattr(invoice, "createdAt", None)
    total_period_profit = _to_float(getattr(invoice, "totalPeriodProfit", None), 0.0)
    calculated_fee = _to_float(getattr(invoice, "calculatedFee", None), 0.0)
    amount_details = get_invoice_payment_amount_details(invoice)
    expected_payment_amount = _to_float(amount_details.get("expected_payment_amount"), 0.0)
    expected_payment_currency = str(amount_details.get("expected_payment_currency") or get_billing_currency()).upper()
    actual_payment_amount_raw = amount_details.get("actual_payment_amount")
    actual_payment_currency = str(amount_details.get("actual_payment_currency") or "").upper()
    display_payment_amount = (
        _to_float(actual_payment_amount_raw, 0.0) if actual_payment_amount_raw is not None else expected_payment_amount
    )
    display_payment_currency = actual_payment_currency or expected_payment_currency
    fee_type = str(_enum_value(getattr(subscription, "feeType", None)) or "percentage")
    fee_value = _to_float(getattr(subscription, "feeValue", None), 0.0)

    payment_method_text = _format_invoice_payment_method_display(invoice, payment_method)

    status_palette = {
        "paid": ("#DCFCE7", "#166534"),
        "pending": ("#FEF3C7", "#B45309"),
        "failed": ("#FEE2E2", "#B91C1C"),
        "skipped": ("#E2E8F0", "#475569"),
    }
    status_bg, status_fg = status_palette.get(status_key, ("#E2E8F0", "#334155"))

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    pdf.setTitle(invoice_label)
    pdf.setAuthor("SmarfRobotTrade")
    pdf.setSubject("Weekly billing invoice")

    page_bg = HexColor("#FFFFFF")
    surface = HexColor("#FFFFFF")
    surface_muted = HexColor("#F8FAFC")
    border = HexColor("#D7DEE7")
    text_primary = HexColor("#0F172A")
    text_secondary = HexColor("#334155")
    text_muted = HexColor("#64748B")

    margin_x = 18 * mm
    margin_top = 16 * mm
    content_width = width - (margin_x * 2)
    page_top = height - margin_top

    pdf.setFillColor(page_bg)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)

    def draw_round_card(
        x: float,
        y_top: float,
        card_width: float,
        card_height: float,
        *,
        fill_color=surface,
        stroke_color=border,
        radius: float = 4,
    ):
        pdf.setFillColor(fill_color)
        pdf.setStrokeColor(stroke_color)
        pdf.roundRect(x, y_top - card_height, card_width, card_height, radius, fill=1, stroke=1)

    def draw_badge(x_right: float, y_top: float, label: str, *, bg_color, text_color):
        font_name = "Helvetica-Bold"
        font_size = 8
        padding_x = 8
        badge_height = 14
        label_width = pdfmetrics.stringWidth(label, font_name, font_size)
        badge_width = label_width + (padding_x * 2)
        pdf.setFillColor(bg_color)
        pdf.setStrokeColor(bg_color)
        x = x_right - badge_width
        pdf.roundRect(x, y_top - badge_height, badge_width, badge_height, 7, fill=1, stroke=0)
        pdf.setFillColor(text_color)
        pdf.setFont(font_name, font_size)
        pdf.drawCentredString(x + (badge_width / 2), y_top - 10.4, label)
        return badge_width

    def draw_label(x: float, y: float, label: str):
        pdf.setFillColor(text_muted)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(x, y, label.upper())

    def draw_value_block(
        x: float,
        y: float,
        text: str,
        *,
        font_name: str = "Helvetica",
        font_size: float = 10,
        color=text_primary,
        max_width: float,
        line_gap: float = 3,
    ):
        pdf.setFillColor(color)
        pdf.setFont(font_name, font_size)
        lines = wrap_text(text, font_name, font_size, max_width)
        current_y = y
        for line in lines:
            pdf.drawString(x, current_y, line)
            current_y -= font_size + line_gap
        return current_y

    def draw_rule(y: float):
        pdf.setStrokeColor(border)
        pdf.setLineWidth(1)
        pdf.line(margin_x, y, margin_x + content_width, y)

    header_left = margin_x
    header_right = margin_x + content_width
    header_top = page_top
    pdf.setFillColor(text_muted)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(header_left, header_top - 8, "SMARFROBOTTRADE")
    pdf.setFillColor(text_primary)
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(header_left, header_top - 24, "Invoice")
    pdf.setFillColor(text_secondary)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(header_left, header_top - 36, "Weekly billing statement")

    pdf.setFillColor(text_primary)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawRightString(header_right, header_top - 12, invoice_label)
    draw_badge(
        header_right,
        header_top - 22,
        status,
        bg_color=HexColor(status_bg if status_key == "paid" else "#F8FAFC"),
        text_color=HexColor(status_fg),
    )

    pdf.setFillColor(text_muted)
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(header_right, header_top - 40, f"Issued {format_datetime_display(created_at, include_time=True)}")
    pdf.drawRightString(header_right, header_top - 52, f"Paid {format_datetime_display(paid_at, include_time=True)}")
    header_bottom = header_top - 58
    draw_rule(header_bottom)

    customer_name = str(getattr(user, "username", "") or "").strip()
    customer_email = str(getattr(user, "email", "") or "").strip() or "Unknown"
    details_top = header_bottom - (8 * mm)
    pdf.setFillColor(text_primary)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin_x, details_top, "Invoice Details")

    detail_rows = [
        ("Customer", customer_name or customer_email),
        ("Email", customer_email),
        ("Billing period", format_period_display(period_start, period_end)),
        ("Payment method", payment_method_text),
        ("Charge currency", display_payment_currency),
        ("Fee model", f"{fee_type.title()} at {fee_value:.2f}{'%' if fee_type == 'percentage' else ' USD'}"),
        ("Status", status),
    ]

    detail_label_width = 34 * mm
    detail_value_width = content_width - detail_label_width - 24
    detail_row_height_cache: list[tuple[str, list[str], float]] = []
    detail_box_height = 12
    for label, value in detail_rows:
        value_lines = wrap_text(value, "Helvetica-Bold", 10, detail_value_width)
        row_height = max(18, 8 + (len(value_lines) * 13))
        detail_row_height_cache.append((label, value_lines, row_height))
        detail_box_height += row_height

    detail_box_top = details_top - 6
    draw_round_card(margin_x, detail_box_top, content_width, detail_box_height, fill_color=surface)

    detail_current_top = detail_box_top - 6
    for index, (label, value_lines, row_height) in enumerate(detail_row_height_cache):
        if index > 0:
            pdf.setStrokeColor(border)
            pdf.setLineWidth(1)
            pdf.line(margin_x, detail_current_top, margin_x + content_width, detail_current_top)
        row_text_y = detail_current_top - 12
        pdf.setFillColor(text_muted)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(margin_x + 10, row_text_y, label.upper())
        pdf.setFillColor(text_primary)
        pdf.setFont("Helvetica-Bold", 10)
        line_y = row_text_y
        for line in value_lines:
            pdf.drawString(margin_x + detail_label_width + 10, line_y, line)
            line_y -= 13
        detail_current_top -= row_height

    charges_title_y = detail_box_top - detail_box_height - (8 * mm)
    pdf.setFillColor(text_primary)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin_x, charges_title_y, "Charge Breakdown")

    charge_table_top = charges_title_y - 6
    description_width = content_width * 0.44
    reference_width = content_width * 0.34
    amount_width = content_width - description_width - reference_width
    charge_header_height = 10 * mm

    draw_round_card(margin_x, charge_table_top, content_width, charge_header_height, fill_color=surface_muted)
    pdf.setFillColor(text_muted)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(margin_x + 10, charge_table_top - 18, "DESCRIPTION")
    pdf.drawString(margin_x + description_width + 10, charge_table_top - 18, "REFERENCE")
    pdf.drawRightString(margin_x + content_width - 10, charge_table_top - 18, "AMOUNT")

    description_lines = wrap_text("Weekly performance fee", "Helvetica-Bold", 10, description_width - 20)
    period_lines = wrap_text(format_period_display(period_start, period_end), "Helvetica", 9, description_width - 20)
    reference_lines = wrap_text(
        f"Net profit {format_currency(total_period_profit, get_billing_base_currency())} at {fee_value:.2f}{'%' if fee_type == 'percentage' else ' USD'}",
        "Helvetica",
        9,
        reference_width - 20,
    )
    charge_row_height = max(
        22 * mm,
        12 + (len(description_lines) * 13) + (len(period_lines) * 12) + 4,
        12 + (len(reference_lines) * 12),
    )
    charge_row_top = charge_table_top - charge_header_height - 2
    draw_round_card(margin_x, charge_row_top, content_width, charge_row_height, fill_color=surface)

    description_y = charge_row_top - 14
    for line in description_lines:
        pdf.setFillColor(text_primary)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(margin_x + 10, description_y, line)
        description_y -= 13
    for line in period_lines:
        pdf.setFillColor(text_muted)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(margin_x + 10, description_y, line)
        description_y -= 12

    reference_y = charge_row_top - 16
    for line in reference_lines:
        pdf.setFillColor(text_secondary)
        pdf.setFont("Helvetica", 9)
        pdf.drawString(margin_x + description_width + 10, reference_y, line)
        reference_y -= 12

    pdf.setFillColor(text_primary)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawRightString(
        margin_x + content_width - 10,
        charge_row_top - 24,
        format_currency(display_payment_amount, display_payment_currency),
    )

    settlement_title_y = charge_row_top - charge_row_height - (8 * mm)
    pdf.setFillColor(text_primary)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin_x, settlement_title_y, "Settlement")

    settlement_box_top = settlement_title_y - 6
    settlement_box_height = 40 * mm
    draw_round_card(margin_x, settlement_box_top, content_width, settlement_box_height, fill_color=surface)

    split_x = margin_x + (content_width * 0.56)
    pdf.setStrokeColor(border)
    pdf.setLineWidth(1)
    pdf.line(split_x, settlement_box_top - 10, split_x, settlement_box_top - settlement_box_height + 10)

    amount_title = "Amount Paid" if status_key == "paid" else "Amount Due"
    pdf.setFillColor(text_muted)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(margin_x + 10, settlement_box_top - 14, amount_title.upper())
    pdf.setFillColor(text_primary)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(margin_x + 10, settlement_box_top - 36, format_currency(display_payment_amount, display_payment_currency))
    pdf.setFillColor(text_secondary)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin_x + 10, settlement_box_top - 50, f"Via {payment_method_text}")
    pdf.drawString(
        margin_x + 10,
        settlement_box_top - 62,
        f"Billing basis {format_currency(calculated_fee, get_billing_base_currency())}",
    )

    summary_rows = [
        ("Net profit", format_currency(total_period_profit, get_billing_base_currency())),
        ("USD fee basis", format_currency(calculated_fee, get_billing_base_currency())),
        ("Status", status),
        ("Paid at", format_datetime_display(paid_at, include_time=True)),
    ]
    summary_y = settlement_box_top - 16
    right_x = split_x + 10
    right_width = (margin_x + content_width) - right_x - 10
    summary_label_width = 22 * mm
    for label, value in summary_rows:
        pdf.setFillColor(text_muted)
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(right_x, summary_y, label.upper())
        pdf.setFillColor(text_primary)
        pdf.setFont("Helvetica-Bold", 9)
        value_lines = wrap_text(value, "Helvetica-Bold", 9, right_width - summary_label_width - 6)
        line_y = summary_y
        for line in value_lines:
            pdf.drawString(right_x + summary_label_width, line_y, line)
            line_y -= 11
        summary_y -= max(16, 6 + (len(value_lines) * 11))

    footer_y = settlement_box_top - settlement_box_height - (8 * mm)
    pdf.setFillColor(text_muted)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(margin_x, footer_y, "This document is generated automatically by SmarfRobotTrade.")

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def build_invoice_html(invoice: Any, *, user: Any, subscription: Any, payment_method: Any | None = None) -> str:
    invoice_id = str(getattr(invoice, "id", "") or "")
    invoice_label = f"INV-{invoice_id[:8].upper()}" if invoice_id else "INV-UNKNOWN"
    status = str(_enum_value(getattr(invoice, "status", None)) or "pending").replace("_", " ").title()
    period_start = _extract_date(getattr(invoice, "billingStartDate", None))
    period_end = _extract_date(getattr(invoice, "billingEndDate", None))
    paid_at = getattr(invoice, "paidAt", None)
    created_at = getattr(invoice, "createdAt", None)
    total_period_profit = _to_float(getattr(invoice, "totalPeriodProfit", None), 0.0)
    calculated_fee = _to_float(getattr(invoice, "calculatedFee", None), 0.0)
    amount_details = get_invoice_payment_amount_details(invoice)
    expected_payment_amount = _to_float(amount_details.get("expected_payment_amount"), 0.0)
    expected_payment_currency = str(amount_details.get("expected_payment_currency") or get_billing_currency()).upper()
    fee_type = str(_enum_value(getattr(subscription, "feeType", None)) or "percentage")
    fee_value = _to_float(getattr(subscription, "feeValue", None), 0.0)

    payment_method_text = _format_invoice_payment_method_display(invoice, payment_method)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_safe_text(invoice_label)}</title>
</head>
<body style="margin:0; padding:24px; background:#ffffff; font-family:Arial, sans-serif; color:#0f172a;">
  <div style="max-width:760px; margin:0 auto; background:#ffffff; border:1px solid #e2e8f0;">
    <div style="padding:24px 28px; border-bottom:1px solid #e2e8f0; color:#0f172a;">
      <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start;">
        <div>
          <div style="font-size:11px; letter-spacing:0.12em; text-transform:uppercase; color:#64748b;">SmarfRobotTrade</div>
          <h1 style="margin:8px 0 0 0; font-size:28px;">Invoice</h1>
          <div style="margin-top:6px; font-size:13px; color:#475569;">Weekly billing statement</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:18px; font-weight:700;">{_safe_text(invoice_label)}</div>
          <div style="margin-top:8px; font-size:13px; color:#475569;">Status: {_safe_text(status)}</div>
        </div>
      </div>
    </div>

    <div style="padding:28px;">
      <div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:20px; margin-bottom:28px;">
        <div>
          <div style="font-size:12px; text-transform:uppercase; color:#64748b; margin-bottom:8px;">Billed To</div>
          <div style="font-weight:700;">{_safe_text(getattr(user, 'username', '') or getattr(user, 'email', ''))}</div>
          <div style="color:#475569; margin-top:4px;">{_safe_text(getattr(user, 'email', ''))}</div>
        </div>
        <div>
          <div style="font-size:12px; text-transform:uppercase; color:#64748b; margin-bottom:8px;">Details</div>
          <div>Issued: {_safe_text(created_at.isoformat() if created_at else "-")}</div>
          <div>Paid: {_safe_text(paid_at.isoformat() if paid_at else "-")}</div>
          <div>Period: {_safe_text(period_start.isoformat() if period_start else "-")} to {_safe_text(period_end.isoformat() if period_end else "-")}</div>
        </div>
      </div>

      <table style="width:100%; border-collapse:collapse; margin-bottom:24px;">
        <thead>
          <tr style="background:#f8fafc;">
            <th style="padding:12px; border:1px solid #e2e8f0; text-align:left;">Item</th>
            <th style="padding:12px; border:1px solid #e2e8f0; text-align:left;">Period</th>
            <th style="padding:12px; border:1px solid #e2e8f0; text-align:right;">Amount</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding:12px; border:1px solid #e2e8f0;">Weekly performance fee</td>
            <td style="padding:12px; border:1px solid #e2e8f0;">{_safe_text(period_start.isoformat() if period_start else "-")} to {_safe_text(period_end.isoformat() if period_end else "-")}</td>
            <td style="padding:12px; border:1px solid #e2e8f0; text-align:right;">{_safe_text(format_currency_amount(expected_payment_amount, expected_payment_currency))}</td>
          </tr>
        </tbody>
      </table>

      <div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:20px;">
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:12px; background:#f8fafc;">
          <div style="font-size:12px; text-transform:uppercase; color:#64748b;">Payment Summary</div>
          <div style="margin-top:10px;">Net profit for period: <strong>{_safe_text(format_currency_amount(total_period_profit, get_billing_base_currency()))}</strong></div>
          <div style="margin-top:6px;">USD fee basis: <strong>{_safe_text(format_currency_amount(calculated_fee, get_billing_base_currency()))}</strong></div>
          <div style="margin-top:6px;">Payment method: <strong>{_safe_text(payment_method_text)}</strong></div>
          <div style="margin-top:6px;">Charge currency: <strong>{_safe_text(expected_payment_currency)}</strong></div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:12px; background:#ffffff;">
          <div style="font-size:12px; text-transform:uppercase; color:#64748b;">Amount Paid</div>
          <div style="margin-top:10px; font-size:28px; font-weight:800; color:#0f172a;">{_safe_text(format_currency_amount(expected_payment_amount, expected_payment_currency))}</div>
          <div style="margin-top:8px; color:#475569;">Via {_safe_text(payment_method_text)}</div>
          <div style="margin-top:4px; color:#475569;">USD basis {_safe_text(format_currency_amount(calculated_fee, get_billing_base_currency()))}</div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""


async def run_subscription_billing_worker() -> None:
    logger.info("subscription_billing_worker started | interval=%ds", _BILLING_POLL_SECONDS)
    while True:
        try:
            await process_all_due_billing()
        except Exception:
            logger.exception("subscription_billing_worker cycle failed")
        await asyncio.sleep(_BILLING_POLL_SECONDS)
