import asyncio
import html
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response

from ..database.client import db
from ..models.subscription_model import (
    AdminBillingConfigResponse,
    AdminInvoiceDetailResponse,
    AdminSubscriptionItemResponse,
    AdminSubscriptionInvoiceListResponse,
    AdminSubscriptionManagementResponse,
    AdminSubscriptionStatsResponse,
    AttachPaymentMethodRequest,
    CreateCheckoutSessionRequest,
    CreateCheckoutSessionResponse,
    CreateSetupIntentResponse,
    InvoiceResponse,
    PayInvoiceRequest,
    PaymentMethodResponse,
    ProcessDueBillingResponse,
    SubscriptionResponse,
    SubscriptionSummaryResponse,
    UpdateBillingConfigRequest,
    UpdateCollectionModeRequest,
    UpdateSubscriptionStatusRequest,
    WeeklyPreviewResponse,
)
from ..utils.subscription_billing import (
    ChargeAttemptResult,
    _build_payment_breakdown,
    _build_payment_error_details,
    _build_payment_method_details,
    _extract_presentment_breakdown as _build_presentment_breakdown,
    _format_invoice_payment_method_display,
    _extract_request_id,
    _invoice_charge_update_payload,
    _to_prisma_json,
    billing_period_for_due_date,
    build_invoice_pdf,
    build_expected_payment_breakdown,
    convert_invoice_amount_to_promptpay_currency,
    get_billing_base_currency,
    get_invoice_payment_amount_details,
    get_promptpay_currency,
    get_promptpay_exchange_rate,
    get_assignable_period_daily_aggregates,
    promptpay_checkout_configured,
    merge_payment_breakdown_with_expected_amount,
    normalize_subscription_next_billing_date,
    notify_invoice_event,
    pay_invoice_now,
    promptpay_amount_to_minor_units,
    process_all_due_billing,
    reconcile_open_invoice_amount,
    resolve_promptpay_exchange_rate,
    sync_daily_aggregate_status_for_invoice,
)
from ..utils.subscription_access import (
    get_user_subscription_access_state,
    suspend_user_bot_runtime,
    sync_subscription_status_from_invoices,
)
from ..utils.notification_delivery import (
    build_absolute_related_link,
    build_generic_notification_email_html,
    claim_notification_dedupe,
    dispatch_notification_to_user,
)
from .authentication import get_current_active_user

try:
    import stripe
except Exception:
    stripe = None

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if stripe is not None and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

subscription_router = APIRouter(tags=["Subscription"])
ADMIN_FEE_TYPES = {"percentage", "fixed"}
ADMIN_SUB_STATUSES = {"active", "past_due", "canceled"}
COLLECTION_MODES = {"automatic", "manual"}
_PAYMENT_METHOD_NOTIFICATION_RELATED_LINK = "/subscription"
STRIPE_WEBHOOK_SECRET = str(os.getenv("STRIPE_WEBHOOK_SECRET", "") or "").strip()
CHECKOUT_PAYMENT_FLOWS = {"card", "promptpay"}
PROMPTPAY_MIN_AMOUNT_THB = 10.0


def _to_float(value: Optional[Decimal], default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _enum_value(value):
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _to_date_string(value: Optional[datetime | date]) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def _to_datetime_string(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    return value.isoformat()


def _start_of_week(today: date) -> date:
    return today - timedelta(days=today.weekday())


def _next_monday(today: date) -> date:
    days_until = (7 - today.weekday()) % 7
    if days_until == 0:
        days_until = 7
    return today + timedelta(days=days_until)


def _parse_iso_date_or_none(raw_value: Optional[str], field_name: str) -> Optional[date]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{field_name} must be in YYYY-MM-DD format")


def _extract_date(value) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _roll_weekly_forward(base_date: date, today: date) -> date:
    # Keep weekly cadence and move to the nearest non-past cycle.
    candidate = base_date
    while candidate < today:
        candidate += timedelta(days=7)
    return candidate


def _resolve_default_next_billing_date(config, today: date) -> date:
    configured = _extract_date(getattr(config, "defaultNextBillingDate", None)) if config else None
    if configured:
        return _roll_weekly_forward(configured, today)
    return _next_monday(today)


def _resolve_subscription_next_billing_date(subscription_date_value, config, today: date) -> date:
    sub_date = _extract_date(subscription_date_value)
    if sub_date:
        return _roll_weekly_forward(sub_date, today)
    return _resolve_default_next_billing_date(config=config, today=today)


def _calculate_estimated_fee(
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


def _summarize_daily_aggregate_rows(rows: list[Any]) -> tuple[float, float, float]:
    gross_profit = 0.0
    gross_loss = 0.0

    for aggregate in rows:
        pnl = _to_float(getattr(aggregate, "dailyNetProfit", None), 0.0)
        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

    net_profit = gross_profit - gross_loss
    return (round(gross_profit, 2), round(gross_loss, 2), round(net_profit, 2))


async def _load_billing_preview_rows(
    *,
    user_id: str,
    period_start: date,
    period_end: date,
    invoice_id: str | None = None,
) -> list[Any]:
    if invoice_id:
        return await db.dailyaggregate.find_many(
            where={"billingInvoiceId": invoice_id}
        )

    return await get_assignable_period_daily_aggregates(
        user_id=user_id,
        period_start=period_start,
        period_end=period_end,
    )


def _require_admin(current_user):
    role = _enum_value(getattr(current_user, "role", None))
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


def _stripe_enabled() -> bool:
    return stripe is not None and bool(STRIPE_SECRET_KEY)


def _ensure_stripe_configured():
    if stripe is None:
        raise HTTPException(status_code=500, detail="Stripe SDK is not installed on server")
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe secret key is not configured")


async def _stripe_call(callable_obj, *args, **kwargs):
    return await asyncio.to_thread(callable_obj, *args, **kwargs)


def _stripe_object_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        result = value.get(key, default)
        return default if result is None else result
    try:
        result = getattr(value, key)
    except AttributeError:
        try:
            result = value[key]
        except Exception:
            return default
    except Exception:
        return default
    return default if result is None else result


def _stripe_object_list(value: Any, key: str) -> list[Any]:
    items = _stripe_object_get(value, key, [])
    if isinstance(items, list):
        return items
    if isinstance(items, tuple):
        return list(items)
    return []


def _resolve_frontend_base_url() -> str:
    frontend_url = build_absolute_related_link("/") or "http://localhost:3000/"
    return str(frontend_url).rstrip("/")


def _resolve_checkout_currency() -> str:
    return get_billing_base_currency().strip().lower() or "usd"


def _resolve_checkout_payment_method_types(payment_flow: str) -> list[str]:
    if payment_flow == "promptpay":
        return ["promptpay"]
    return ["card"]


def _build_checkout_success_url(invoice_id: str) -> str:
    return (
        f"{_resolve_frontend_base_url()}/subscription"
        f"?checkout=success&invoice_id={invoice_id}&session_id={{CHECKOUT_SESSION_ID}}"
    )


def _build_checkout_cancel_url(invoice_id: str) -> str:
    return f"{_resolve_frontend_base_url()}/subscription?checkout=cancelled&invoice_id={invoice_id}"


def _checkout_session_is_reusable(session) -> bool:
    return (
        str(_stripe_object_get(session, "status") or "").strip().lower() == "open"
        and bool(_stripe_object_get(session, "url"))
    )


def _extract_stripe_object_id(value) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        object_id = str(value.get("id") or "").strip()
        return object_id or None
    object_id = str(getattr(value, "id", "") or "").strip()
    return object_id or None


def _checkout_note_for_status(status: str) -> str:
    if status == "paid":
        return "Stripe Checkout payment completed."
    if status == "failed":
        return "Stripe Checkout payment failed."
    return "Awaiting Stripe Checkout payment completion."


def _escape_text(value: object) -> str:
    return html.escape(str(value or "").strip())


def _promptpay_enabled() -> bool:
    return promptpay_checkout_configured()


def _normalize_checkout_payment_flow(raw_value: str | None) -> str:
    payment_flow = str(raw_value or "card").strip().lower() or "card"
    if payment_flow not in CHECKOUT_PAYMENT_FLOWS:
        raise HTTPException(status_code=400, detail="payment_flow must be card or promptpay")
    if payment_flow == "promptpay" and not _promptpay_enabled():
        raise HTTPException(status_code=400, detail="PromptPay checkout is not configured")
    return payment_flow


def _payment_method_label(method) -> str:
    brand = str(getattr(method, "cardBrand", "") or "").strip().upper() or "CARD"
    last4 = str(getattr(method, "cardLast4", "") or "").strip()
    if last4:
        return f"{brand} ending in {last4}"
    return brand


async def _notify_payment_method_added(*, user, method, set_as_default: bool) -> None:
    try:
        method_id = str(getattr(method, "id", "") or "").strip()
        if not method_id:
            return

        provider_method_id = str(getattr(method, "providerMethodId", "") or "").strip() or method_id
        user_id = str(getattr(user, "id", "") or "").strip()
        if not user_id:
            return

        if not claim_notification_dedupe(
            key=f"payment_method_added:{user_id}:{provider_method_id}",
            ttl_seconds=300,
        ):
            return

        method_label = _payment_method_label(method)
        default_text = " This card is now your default payment method." if set_as_default else ""
        title = "Payment method added"
        message = f"{method_label} was added to your billing profile.{default_text}".strip()
        action_url = build_absolute_related_link(_PAYMENT_METHOD_NOTIFICATION_RELATED_LINK) or _PAYMENT_METHOD_NOTIFICATION_RELATED_LINK
        email_html = build_generic_notification_email_html(
            title=title,
            greeting=f"Hi {str(getattr(user, 'username', '') or getattr(user, 'email', '') or 'Trader').strip()},",
            message=(
                f"You added {method_label} to your SmarfRobotTrade billing profile."
                f"{default_text} Open Subscription & Billing to review saved cards and billing mode."
            ),
            related_link=action_url,
            action_label="Open billing",
        )

        await dispatch_notification_to_user(
            user,
            title=title,
            message=message,
            related_link=_PAYMENT_METHOD_NOTIFICATION_RELATED_LINK,
            email_subject=f"Payment method added ({_escape_text(method_label)}) - SmarfRobotTrade",
            email_html=email_html,
            action_label="Open billing",
            send_discord_channel=False,
        )
    except Exception:
        return


def _map_subscription_response(
    subscription,
    *,
    status_override: str | None = None,
    next_billing_date: str | None = None,
    promptpay_exchange_rate: float | None = None,
):
    fee_type = _enum_value(subscription.feeType) or "percentage"
    fee_value = _to_float(subscription.feeValue)
    min_profit_threshold = _to_float(subscription.minProfitThreshold)
    return SubscriptionResponse(
        id=str(subscription.id),
        status=status_override or _enum_value(subscription.status) or "active",
        collection_mode=_enum_value(getattr(subscription, "collectionMode", None)) or "automatic",
        fee_type=fee_type,
        fee_value=round(fee_value, 2),
        min_profit_threshold=round(min_profit_threshold, 2),
        next_billing_date=next_billing_date if next_billing_date is not None else _to_date_string(subscription.nextBillingDate),
        default_payment_method_id=subscription.defaultPaymentMethodId,
        billing_currency=get_billing_base_currency(),
        billing_exchange_rate=1.0,
        promptpay_enabled=_promptpay_enabled(),
        promptpay_currency=get_promptpay_currency() if _promptpay_enabled() else None,
        promptpay_exchange_rate=(
            promptpay_exchange_rate
            if _promptpay_enabled() and promptpay_exchange_rate is not None
            else (get_promptpay_exchange_rate() if _promptpay_enabled() else None)
        ),
    )


async def _get_or_create_subscription(user_id: str):
    subscription = await db.subscription.find_first(where={"userId": user_id})
    if subscription:
        return subscription

    config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})
    default_fee_type = _enum_value(config.defaultFeeType) if config else None
    if not default_fee_type:
        default_fee_type = "percentage"
    default_collection_mode = _enum_value(getattr(config, "defaultCollectionMode", None)) if config else None
    if default_collection_mode not in COLLECTION_MODES:
        default_collection_mode = "automatic"
    default_fee_value = config.defaultFeeValue if config and config.defaultFeeValue is not None else Decimal("20.00")
    default_min_threshold = (
        config.defaultMinThreshold if config and config.defaultMinThreshold is not None else Decimal("0.00")
    )
    next_billing_date = _resolve_default_next_billing_date(config=config, today=date.today())

    return await db.subscription.create(
        data={
            "userId": user_id,
            "status": "active",
            "collectionMode": default_collection_mode,
            "feeType": default_fee_type,
            "feeValue": default_fee_value,
            "minProfitThreshold": default_min_threshold,
            "nextBillingDate": datetime.combine(next_billing_date, datetime.min.time()),
        }
    )


async def _get_or_create_stripe_customer(current_user):
    _ensure_stripe_configured()

    user_id = str(current_user.id)
    customer_id = current_user.stripeCustomerId

    if customer_id:
        try:
            customer = await _stripe_call(stripe.Customer.retrieve, customer_id)
            if customer and not bool(_stripe_object_get(customer, "deleted", False)):
                return customer_id
        except Exception:
            pass

    customer = await _stripe_call(
        stripe.Customer.create,
        email=current_user.email,
        name=current_user.username,
        metadata={"user_id": user_id},
    )
    customer_id = _extract_stripe_object_id(customer)
    if not customer_id:
        raise HTTPException(status_code=500, detail="Failed to create Stripe customer")

    await db.user.update(
        where={"id": user_id},
        data={"stripeCustomerId": customer_id},
    )
    return customer_id


async def _upsert_local_payment_method(
    user_id: str,
    stripe_payment_method,
    is_default: bool,
):
    provider_method_id = _extract_stripe_object_id(stripe_payment_method)
    card = _stripe_object_get(stripe_payment_method, "card", {}) or {}

    existing = await db.userpaymentmethod.find_first(
        where={
            "userId": user_id,
            "providerMethodId": provider_method_id,
        }
    )

    update_data = {
        "type": _stripe_object_get(stripe_payment_method, "type"),
        "cardLast4": _stripe_object_get(card, "last4"),
        "cardBrand": _stripe_object_get(card, "brand"),
        "expiryMonth": _stripe_object_get(card, "exp_month"),
        "expiryYear": _stripe_object_get(card, "exp_year"),
        "isActive": True,
        "isDefault": is_default,
    }

    if existing:
        return await db.userpaymentmethod.update(
            where={"id": str(existing.id)},
            data=update_data,
        )

    return await db.userpaymentmethod.create(
        data={
            "userId": user_id,
            "providerMethodId": provider_method_id,
            **update_data,
        }
    )


async def _sync_customer_payment_methods(user_id: str, stripe_customer_id: str):
    if not _stripe_enabled():
        return

    customer = await _stripe_call(stripe.Customer.retrieve, stripe_customer_id)
    invoice_settings = _stripe_object_get(customer, "invoice_settings", {}) or {}
    default_payment_method_id = _extract_stripe_object_id(_stripe_object_get(invoice_settings, "default_payment_method"))

    methods_response = await _stripe_call(
        stripe.PaymentMethod.list,
        customer=stripe_customer_id,
        type="card",
        limit=20,
    )
    methods = _stripe_object_list(methods_response, "data")
    stripe_ids = {_extract_stripe_object_id(method) for method in methods if _extract_stripe_object_id(method)}

    for method in methods:
        provider_method_id = _extract_stripe_object_id(method)
        await _upsert_local_payment_method(
            user_id=user_id,
            stripe_payment_method=method,
            is_default=provider_method_id == default_payment_method_id,
        )

    local_active_methods = await db.userpaymentmethod.find_many(
        where={"userId": user_id, "isActive": True}
    )
    for local_method in local_active_methods:
        provider_id = local_method.providerMethodId
        if provider_id and provider_id not in stripe_ids:
            await db.userpaymentmethod.update(
                where={"id": str(local_method.id)},
                data={"isActive": False, "isDefault": False},
            )


async def _sync_checkout_card_payment_method(
    *,
    user_id: str,
    stripe_customer_id: str | None,
    stripe_payment_intent,
    subscription_id: str,
):
    if not stripe_customer_id or not _stripe_enabled():
        return None

    provider_method_id = _extract_stripe_object_id(_stripe_object_get(stripe_payment_intent, "payment_method"))
    existing_local_method = None
    if provider_method_id:
        existing_local_method = await db.userpaymentmethod.find_first(
            where={
                "userId": user_id,
                "providerMethodId": provider_method_id,
                "isActive": True,
            }
        )

    had_active_methods = await db.userpaymentmethod.count(where={"userId": user_id, "isActive": True}) > 0
    await _sync_customer_payment_methods(user_id, stripe_customer_id)

    if not provider_method_id:
        return None

    local_method = await db.userpaymentmethod.find_first(
        where={
            "userId": user_id,
            "providerMethodId": provider_method_id,
            "isActive": True,
        }
    )
    if not local_method:
        return None

    subscription = await db.subscription.find_unique(where={"id": subscription_id})
    if not subscription:
        return local_method

    should_set_default = not had_active_methods or not getattr(subscription, "defaultPaymentMethodId", None)
    if should_set_default:
        await _set_default_payment_method(
            user_id=user_id,
            method_id=str(local_method.id),
            stripe_customer_id=stripe_customer_id,
        )
        local_method = await db.userpaymentmethod.find_unique(where={"id": str(local_method.id)}) or local_method

    if existing_local_method is None:
        user = await db.user.find_unique(where={"id": user_id})
        if user:
            await _notify_payment_method_added(
                user=user,
                method=local_method,
                set_as_default=bool(getattr(local_method, "isDefault", False)),
            )

    return local_method


async def _set_default_payment_method(
    user_id: str,
    method_id: str,
    stripe_customer_id: Optional[str] = None,
):
    method = await db.userpaymentmethod.find_first(
        where={"id": method_id, "userId": user_id, "isActive": True}
    )
    if not method:
        raise HTTPException(status_code=404, detail="Payment method not found")

    if stripe_customer_id and method.providerMethodId and _stripe_enabled():
        try:
            await _stripe_call(
                stripe.Customer.modify,
                stripe_customer_id,
                invoice_settings={"default_payment_method": method.providerMethodId},
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(error)}")

    await db.userpaymentmethod.update_many(
        where={"userId": user_id},
        data={"isDefault": False},
    )
    await db.userpaymentmethod.update(
        where={"id": method_id},
        data={"isDefault": True},
    )

    subscription = await _get_or_create_subscription(user_id)
    await db.subscription.update(
        where={"id": str(subscription.id)},
        data={"defaultPaymentMethodId": method_id},
    )


def _map_payment_method(method) -> PaymentMethodResponse:
    return PaymentMethodResponse(
        id=str(method.id),
        type=method.type,
        card_last4=method.cardLast4,
        card_brand=method.cardBrand,
        expiry_month=method.expiryMonth,
        expiry_year=method.expiryYear,
        is_default=bool(method.isDefault),
    )


def _extract_invoice_payment_display(invoice) -> tuple[float | None, str | None]:
    amount_details = get_invoice_payment_amount_details(invoice)
    payment_amount = amount_details.get("actual_payment_amount")
    payment_currency = amount_details.get("actual_payment_currency")

    if payment_amount is None or not payment_currency:
        payment_amount = amount_details.get("expected_payment_amount")
        payment_currency = amount_details.get("expected_payment_currency")

    if payment_amount is None:
        payment_amount = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
    if not payment_currency:
        payment_currency = get_billing_base_currency()

    return (round(_to_float(payment_amount, 0.0), 2), str(payment_currency).upper())


def _map_invoice_response(invoice) -> InvoiceResponse:
    payment_amount, payment_currency = _extract_invoice_payment_display(invoice)
    return InvoiceResponse(
        id=str(invoice.id),
        billing_start_date=_to_date_string(invoice.billingStartDate),
        billing_end_date=_to_date_string(invoice.billingEndDate),
        total_period_profit=round(_to_float(invoice.totalPeriodProfit), 2),
        calculated_fee=round(_to_float(invoice.calculatedFee), 2),
        payment_amount=payment_amount,
        payment_currency=payment_currency,
        status=_enum_value(invoice.status),
        payment_method_used=invoice.paymentMethodUsed,
        payment_method_label=_format_invoice_payment_method_display(invoice),
        paid_at=_to_datetime_string(invoice.paidAt),
        created_at=_to_datetime_string(invoice.createdAt),
    )


def _invoice_period(invoice) -> tuple[date | None, date | None]:
    return (
        _extract_date(getattr(invoice, "billingStartDate", None)),
        _extract_date(getattr(invoice, "billingEndDate", None)),
    )


def _invoice_periods_overlap(left_invoice, right_invoice) -> bool:
    left_start, left_end = _invoice_period(left_invoice)
    right_start, right_end = _invoice_period(right_invoice)
    if not left_start or not left_end or not right_start or not right_end:
        return False
    return left_start <= right_end and right_start <= left_end


def _filter_user_visible_invoices(invoices: list[Any]) -> list[Any]:
    visible_invoices: list[Any] = []
    for invoice in invoices:
        invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "").strip().lower()
        invoice_amount = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
        overlaps_another_invoice = any(
            str(getattr(other_invoice, "id", "") or "") != str(getattr(invoice, "id", "") or "")
            and _invoice_periods_overlap(invoice, other_invoice)
            for other_invoice in invoices
        )
        if invoice_status == "skipped" and invoice_amount <= 0 and overlaps_another_invoice:
            continue
        visible_invoices.append(invoice)
    return visible_invoices


def _checkout_session_is_paid(session: Any) -> bool:
    if not session:
        return False
    return str(_stripe_object_get(session, "payment_status") or "").strip().lower() == "paid"


async def _find_paid_checkout_session_for_invoice(
    *,
    invoice_id: str,
    stripe_customer_id: str | None,
    preferred_session_id: str | None = None,
) -> Any | None:
    seen_session_ids: set[str] = set()

    async def try_session(session_id: str | None) -> Any | None:
        if not session_id or not session_id.startswith("cs_") or session_id in seen_session_ids:
            return None
        seen_session_ids.add(session_id)
        try:
            session = await _stripe_call(
                stripe.checkout.Session.retrieve,
                session_id,
                expand=["payment_intent"],
            )
        except Exception:
            return None
        metadata = _stripe_object_get(session, "metadata", {}) or {}
        if str(_stripe_object_get(metadata, "invoice_id") or "").strip() != invoice_id:
            return None
        if _checkout_session_is_paid(session):
            return session
        return None

    paid_session = await try_session(preferred_session_id)
    if paid_session:
        return paid_session

    if not stripe_customer_id:
        return None

    try:
        payment_intents = await _stripe_call(
            stripe.PaymentIntent.list,
            customer=stripe_customer_id,
            limit=25,
        )
    except Exception:
        return None

    for payment_intent in _stripe_object_list(payment_intents, "data"):
        metadata = _stripe_object_get(payment_intent, "metadata", {}) or {}
        if str(_stripe_object_get(metadata, "invoice_id") or "").strip() != invoice_id:
            continue
        if str(_stripe_object_get(payment_intent, "status") or "").strip().lower() != "succeeded":
            continue

        payment_details = _stripe_object_get(payment_intent, "payment_details", {}) or {}
        order_reference = str(_stripe_object_get(payment_details, "order_reference") or "").strip()
        paid_session = await try_session(order_reference)
        if paid_session:
            return paid_session

        return {
            "id": order_reference or f"checkout-{_stripe_object_get(payment_intent, 'id')}",
            "customer": stripe_customer_id,
            "payment_intent": _stripe_object_get(payment_intent, "id"),
            "payment_status": "paid",
            "status": "complete",
            "metadata": metadata,
        }

    return None


async def _finalize_paid_checkout_invoice_if_available(
    *,
    invoice,
    user,
    preferred_session_id: str | None = None,
    notify_user: bool = False,
):
    if not _stripe_enabled():
        return invoice

    invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "").strip().lower()
    if invoice_status == "paid":
        return invoice

    stripe_customer_id = str(getattr(user, "stripeCustomerId", "") or "").strip() or None
    paid_session = await _find_paid_checkout_session_for_invoice(
        invoice_id=str(invoice.id),
        stripe_customer_id=stripe_customer_id,
        preferred_session_id=preferred_session_id or str(getattr(invoice, "processorRequestId", "") or "").strip() or None,
    )
    if not paid_session:
        return invoice

    return await _finalize_checkout_invoice(
        invoice=invoice,
        user=user,
        stripe_session=paid_session,
        target_status="paid",
        notify_user=notify_user,
    )


async def _build_checkout_charge_result(
    *,
    invoice,
    stripe_session,
    stripe_payment_intent,
    local_payment_method,
    status: str,
):
    latest_charge = _stripe_object_get(stripe_payment_intent, "latest_charge") or {}
    if not isinstance(latest_charge, dict) and not hasattr(latest_charge, "id"):
        latest_charge = {}
    invoice_amount_usd = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
    amount_details = get_invoice_payment_amount_details(invoice)
    expected_payment_amount = amount_details.get("expected_payment_amount")
    expected_payment_currency = amount_details.get("expected_payment_currency")
    configured_exchange_rate = amount_details.get("configured_exchange_rate")

    payment_breakdown = _build_payment_breakdown(stripe_payment_intent)
    presentment_breakdown = _build_presentment_breakdown(stripe_session) or _build_presentment_breakdown(
        stripe_payment_intent
    )
    if presentment_breakdown:
        payment_breakdown = {**(payment_breakdown or {}), **presentment_breakdown}

    return ChargeAttemptResult(
        status=status,
        payment_intent_id=_extract_stripe_object_id(stripe_payment_intent),
        paid_at=datetime.utcnow() if status == "paid" else None,
        note=_checkout_note_for_status(status),
        local_payment_method_id=str(getattr(local_payment_method, "id", "") or "").strip() or None,
        provider_payment_method_id=_extract_stripe_object_id(_stripe_object_get(stripe_payment_intent, "payment_method")),
        request_id=_extract_request_id(stripe_payment_intent) or getattr(invoice, "processorRequestId", None),
        charge_id=_extract_stripe_object_id(latest_charge),
        balance_transaction_id=_extract_stripe_object_id(_stripe_object_get(latest_charge, "balance_transaction")),
        payment_breakdown=merge_payment_breakdown_with_expected_amount(
            payment_breakdown,
            invoice_amount_usd=invoice_amount_usd,
            expected_payment_amount=expected_payment_amount,
            expected_payment_currency=expected_payment_currency,
            configured_exchange_rate=configured_exchange_rate,
        ),
        payment_method_details=_build_payment_method_details(
            local_payment_method=local_payment_method,
            stripe_payment_method=_stripe_object_get(stripe_payment_intent, "payment_method"),
        ),
        payment_error_details=_build_payment_error_details(None, payment_intent=stripe_payment_intent),
    )


async def _finalize_checkout_invoice(
    *,
    invoice,
    user,
    stripe_session,
    target_status: str,
    notify_user: bool,
):
    subscription = getattr(invoice, "subscription", None)
    if subscription is None:
        subscription = await db.subscription.find_unique(where={"id": str(invoice.subId)})
    if subscription is None:
        raise HTTPException(status_code=404, detail="Subscription not found for invoice")

    stripe_payment_intent = None
    payment_intent_id = _extract_stripe_object_id(_stripe_object_get(stripe_session, "payment_intent"))
    if payment_intent_id:
        try:
            stripe_payment_intent = await _stripe_call(
                stripe.PaymentIntent.retrieve,
                payment_intent_id,
                expand=["latest_charge.balance_transaction", "payment_method"],
            )
        except Exception:
            stripe_payment_intent = None

    local_payment_method = None
    stripe_customer_id = _extract_stripe_object_id(_stripe_object_get(stripe_session, "customer")) or str(
        getattr(user, "stripeCustomerId", "") or ""
    ).strip() or None
    payment_method_data = _stripe_object_get(stripe_payment_intent, "payment_method", {}) or {}
    payment_method_type = ""
    if payment_method_data:
        payment_method_type = str(_stripe_object_get(payment_method_data, "type") or "").strip().lower()

    if stripe_payment_intent and stripe_customer_id and payment_method_type == "card":
        local_payment_method = await _sync_checkout_card_payment_method(
            user_id=str(user.id),
            stripe_customer_id=stripe_customer_id,
            stripe_payment_intent=stripe_payment_intent,
            subscription_id=str(subscription.id),
        )

    if stripe_payment_intent:
        charge_result = await _build_checkout_charge_result(
            invoice=invoice,
            stripe_session=stripe_session,
            stripe_payment_intent=stripe_payment_intent,
            local_payment_method=local_payment_method,
            status=target_status,
        )
        update_payload = _invoice_charge_update_payload(
            charge_result,
            payment_method_used=charge_result.local_payment_method_id,
            status=target_status,
        )
    else:
        invoice_amount_usd = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
        amount_details = get_invoice_payment_amount_details(invoice)
        payment_breakdown = merge_payment_breakdown_with_expected_amount(
            _build_presentment_breakdown(stripe_session),
            invoice_amount_usd=invoice_amount_usd,
            expected_payment_amount=amount_details.get("expected_payment_amount"),
            expected_payment_currency=amount_details.get("expected_payment_currency"),
            configured_exchange_rate=amount_details.get("configured_exchange_rate"),
        )
        update_payload = {
            "status": target_status,
            "paidAt": datetime.utcnow() if target_status == "paid" else None,
            "stripePaymentIntentId": payment_intent_id,
            "processorRequestId": _extract_stripe_object_id(stripe_session),
            "paymentBreakdown": _to_prisma_json(payment_breakdown) if payment_breakdown else None,
        }

    updated_invoice = await db.invoice.update(
        where={"id": str(invoice.id)},
        data=update_payload,
        include={"subscription": True},
    )
    await sync_daily_aggregate_status_for_invoice(str(updated_invoice.id), target_status)
    await sync_subscription_status_from_invoices(str(subscription.id))

    if notify_user:
        await notify_invoice_event(
            invoice=updated_invoice,
            user=user,
            subscription=updated_invoice.subscription,
            event_type="payment_received" if target_status == "paid" else "payment_failed",
            note=_checkout_note_for_status(target_status),
            source="manual",
            event_token=payment_intent_id or _extract_stripe_object_id(stripe_session) or target_status,
        )

    return updated_invoice


@subscription_router.get("/summary", response_model=SubscriptionSummaryResponse)
async def get_subscription_summary(
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    user_id = str(current_user.id)
    today = date.today()

    subscription = await _get_or_create_subscription(user_id)
    config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})

    if _stripe_enabled() and current_user.stripeCustomerId:
        try:
            await _sync_customer_payment_methods(user_id, current_user.stripeCustomerId)
        except Exception:
            pass

    await process_all_due_billing(today=today, user_id=user_id)
    await sync_subscription_status_from_invoices(str(subscription.id), allow_reactivate=False)
    subscription = await db.subscription.find_unique(where={"id": str(subscription.id)}) or subscription
    fallback_next_billing_date = _extract_date(subscription.nextBillingDate)
    if not fallback_next_billing_date:
        fallback_next_billing_date = _resolve_subscription_next_billing_date(
            subscription.nextBillingDate,
            config=config,
            today=today,
        )
    subscription, resolved_next_billing_date = await normalize_subscription_next_billing_date(
        subscription,
        fallback_due_date=fallback_next_billing_date,
    )
    payment_methods = await db.userpaymentmethod.find_many(
        where={"userId": user_id, "isActive": True}
    )
    payment_methods_sorted = sorted(
        payment_methods,
        key=lambda method: (not bool(method.isDefault), str(method.id)),
    )

    invoices = await db.invoice.find_many(
        where={"subId": str(subscription.id)},
        order={"createdAt": "desc"},
        take=20,
    )
    invoices = _filter_user_visible_invoices(invoices)
    unresolved_invoice = next(
        (
            invoice
            for invoice in invoices
            if str(_enum_value(getattr(invoice, "status", None)) or "").strip().lower() in {"pending", "failed"}
        ),
        None,
    )
    if unresolved_invoice:
        unresolved_invoice = await _finalize_paid_checkout_invoice_if_available(
            invoice=unresolved_invoice,
            user=current_user,
            notify_user=False,
        )
        unresolved_invoice = await reconcile_open_invoice_amount(
            unresolved_invoice,
            subscription=subscription,
            user_id=user_id,
        )
        invoices = [
            unresolved_invoice if str(invoice.id) == str(unresolved_invoice.id) else invoice
            for invoice in invoices
        ]
        if str(_enum_value(getattr(unresolved_invoice, "status", None)) or "").strip().lower() not in {"pending", "failed"}:
            unresolved_invoice = None

    fee_type = _enum_value(subscription.feeType) or "percentage"
    fee_value = _to_float(subscription.feeValue)
    min_profit_threshold = _to_float(subscription.minProfitThreshold)
    if unresolved_invoice:
        billing_period_start = _extract_date(getattr(unresolved_invoice, "billingStartDate", None)) or today
        billing_period_end = _extract_date(getattr(unresolved_invoice, "billingEndDate", None)) or billing_period_start
        weekly_aggregates = await _load_billing_preview_rows(
            user_id=user_id,
            period_start=billing_period_start,
            period_end=billing_period_end,
            invoice_id=str(unresolved_invoice.id),
        )
        gross_profit, gross_loss, net_profit = _summarize_daily_aggregate_rows(weekly_aggregates)
        net_profit = round(_to_float(getattr(unresolved_invoice, "totalPeriodProfit", None), net_profit), 2)
        estimated_fee = round(_to_float(getattr(unresolved_invoice, "calculatedFee", None), 0.0), 2)
        if not weekly_aggregates:
            gross_profit = max(net_profit, 0.0)
            gross_loss = abs(min(net_profit, 0.0))
    else:
        billing_period_start, billing_period_end = billing_period_for_due_date(resolved_next_billing_date or today)
        weekly_aggregates = await _load_billing_preview_rows(
            user_id=user_id,
            period_start=billing_period_start,
            period_end=billing_period_end,
        )
        gross_profit, gross_loss, net_profit = _summarize_daily_aggregate_rows(weekly_aggregates)
        estimated_fee = _calculate_estimated_fee(
            net_profit=net_profit,
            fee_type=fee_type,
            fee_value=fee_value,
            min_profit_threshold=min_profit_threshold,
        )
    next_billing_date = resolved_next_billing_date.isoformat() if resolved_next_billing_date else None
    promptpay_exchange_rate = None
    if _promptpay_enabled():
        try:
            promptpay_exchange_rate = await resolve_promptpay_exchange_rate(require_config=False)
        except Exception:
            promptpay_exchange_rate = get_promptpay_exchange_rate()
    subscription = await db.subscription.find_unique(where={"id": str(subscription.id)}) or subscription
    access_state = await get_user_subscription_access_state(user_id)

    mapped_subscription = _map_subscription_response(
        subscription,
        status_override=access_state.subscription_status or _enum_value(subscription.status) or "active",
        next_billing_date=next_billing_date,
        promptpay_exchange_rate=promptpay_exchange_rate,
    )

    mapped_invoices = [_map_invoice_response(invoice) for invoice in invoices]

    mapped_payment_methods = [_map_payment_method(method) for method in payment_methods_sorted]

    weekly_preview = WeeklyPreviewResponse(
        week_start=billing_period_start.isoformat(),
        week_end=billing_period_end.isoformat(),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        net_profit=round(net_profit, 2),
        estimated_fee=round(estimated_fee, 2),
        estimated_fee_payment=round(estimated_fee, 2),
        estimated_fee_payment_currency=get_billing_base_currency(),
    )

    return SubscriptionSummaryResponse(
        subscription=mapped_subscription,
        invoices=mapped_invoices,
        payment_methods=mapped_payment_methods,
        weekly_preview=weekly_preview,
    )


@subscription_router.patch("/collection-mode", response_model=SubscriptionResponse)
async def update_subscription_collection_mode(
    data: UpdateCollectionModeRequest,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    next_collection_mode = str(data.collection_mode or "").strip().lower()
    if next_collection_mode not in COLLECTION_MODES:
        raise HTTPException(status_code=400, detail="collection_mode must be automatic or manual")

    subscription = await _get_or_create_subscription(str(current_user.id))
    if next_collection_mode == "automatic":
        has_active_payment_method = await db.userpaymentmethod.find_first(
            where={
                "userId": str(current_user.id),
                "isActive": True,
            }
        )
        if has_active_payment_method is None:
            raise HTTPException(
                status_code=400,
                detail="Add a saved card before switching billing to automatic",
            )

    updated_subscription = await db.subscription.update(
        where={"id": str(subscription.id)},
        data={"collectionMode": next_collection_mode},
    )
    access_state = await get_user_subscription_access_state(str(current_user.id))
    return _map_subscription_response(
        updated_subscription,
        status_override=access_state.subscription_status or _enum_value(updated_subscription.status) or "active",
    )


@subscription_router.post("/setup-intent", response_model=CreateSetupIntentResponse)
async def create_setup_intent(
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    customer_id = await _get_or_create_stripe_customer(current_user)

    try:
        setup_intent = await _stripe_call(
            stripe.SetupIntent.create,
            customer=customer_id,
            payment_method_types=["card"],
            usage="off_session",
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(error)}")

    client_secret = _stripe_object_get(setup_intent, "client_secret")
    if not client_secret:
        raise HTTPException(status_code=500, detail="Failed to create Stripe setup intent")

    return CreateSetupIntentResponse(client_secret=client_secret)


@subscription_router.post(
    "/payment-methods",
    response_model=PaymentMethodResponse,
    status_code=status.HTTP_201_CREATED,
)
async def attach_payment_method(
    data: AttachPaymentMethodRequest,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    customer_id = await _get_or_create_stripe_customer(current_user)
    user_id = str(current_user.id)
    payment_method_id = data.paymentMethodId.strip()

    if not payment_method_id:
        raise HTTPException(status_code=400, detail="paymentMethodId is required")

    try:
        stripe_payment_method = await _stripe_call(stripe.PaymentMethod.retrieve, payment_method_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(error)}")

    if _stripe_object_get(stripe_payment_method, "type") != "card":
        raise HTTPException(status_code=400, detail="Only card payment methods are supported")

    attached_customer = _extract_stripe_object_id(_stripe_object_get(stripe_payment_method, "customer")) or _stripe_object_get(
        stripe_payment_method,
        "customer",
    )
    if attached_customer and attached_customer != customer_id:
        raise HTTPException(status_code=400, detail="Payment method belongs to another customer")

    if not attached_customer:
        try:
            stripe_payment_method = await _stripe_call(
                stripe.PaymentMethod.attach,
                payment_method_id,
                customer=customer_id,
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(error)}")

    local_method = await _upsert_local_payment_method(
        user_id=user_id,
        stripe_payment_method=stripe_payment_method,
        is_default=False,
    )

    active_methods_count = await db.userpaymentmethod.count(where={"userId": user_id, "isActive": True})
    set_as_default = data.setAsDefault or active_methods_count == 1

    if set_as_default:
        await _set_default_payment_method(
            user_id=user_id,
            method_id=str(local_method.id),
            stripe_customer_id=customer_id,
        )
        local_method = await db.userpaymentmethod.find_unique(where={"id": str(local_method.id)})
        if not local_method:
            raise HTTPException(status_code=500, detail="Failed to fetch payment method")

    await _notify_payment_method_added(
        user=current_user,
        method=local_method,
        set_as_default=bool(getattr(local_method, "isDefault", False)),
    )

    return _map_payment_method(local_method)


@subscription_router.patch("/payment-methods/{method_id}/default")
async def set_default_payment_method(
    method_id: str,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    await _set_default_payment_method(
        user_id=str(current_user.id),
        method_id=method_id,
        stripe_customer_id=current_user.stripeCustomerId,
    )
    return {"message": "Default payment method updated"}


@subscription_router.delete("/payment-methods/{method_id}")
async def remove_payment_method(
    method_id: str,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    user_id = str(current_user.id)

    method = await db.userpaymentmethod.find_first(
        where={"id": method_id, "userId": user_id, "isActive": True}
    )
    if not method:
        raise HTTPException(status_code=404, detail="Payment method not found")

    if method.providerMethodId and _stripe_enabled():
        try:
            await _stripe_call(stripe.PaymentMethod.detach, method.providerMethodId)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(error)}")

    await db.userpaymentmethod.update(
        where={"id": method_id},
        data={"isActive": False, "isDefault": False},
    )

    subscription = await db.subscription.find_first(where={"userId": user_id})
    if subscription and subscription.defaultPaymentMethodId == method_id:
        replacement = await db.userpaymentmethod.find_first(
            where={"userId": user_id, "isActive": True}
        )
        if replacement:
            await _set_default_payment_method(
                user_id=user_id,
                method_id=str(replacement.id),
                stripe_customer_id=current_user.stripeCustomerId,
            )
        else:
            await db.subscription.update(
                where={"id": str(subscription.id)},
                data={"defaultPaymentMethodId": None},
            )
            if current_user.stripeCustomerId and _stripe_enabled():
                try:
                    await _stripe_call(
                        stripe.Customer.modify,
                        current_user.stripeCustomerId,
                        invoice_settings={"default_payment_method": None},
                    )
                except Exception:
                    pass

    return {"message": "Payment method removed"}


@subscription_router.get("/invoices/{invoice_id}/download")
async def download_invoice(
    invoice_id: str,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    invoice = await db.invoice.find_unique(
        where={"id": invoice_id},
        include={"subscription": True},
    )
    if not invoice or str(invoice.subscription.userId) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Invoice not found")
    invoice_status = (_enum_value(invoice.status) or "").lower()
    if invoice_status != "paid":
        raise HTTPException(status_code=400, detail="Only paid invoices can be downloaded")

    payment_method = None
    payment_method_id = str(invoice.paymentMethodUsed or "").strip()
    if payment_method_id:
        payment_method = await db.userpaymentmethod.find_unique(where={"id": payment_method_id})
    if not payment_method and invoice.subscription.defaultPaymentMethodId:
        payment_method = await db.userpaymentmethod.find_unique(
            where={"id": str(invoice.subscription.defaultPaymentMethodId)}
        )

    try:
        pdf_bytes = build_invoice_pdf(
            invoice,
            user=current_user,
            subscription=invoice.subscription,
            payment_method=payment_method,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))

    filename = f"invoice-{str(invoice.id)[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@subscription_router.post(
    "/invoices/{invoice_id}/checkout-session",
    response_model=CreateCheckoutSessionResponse,
)
async def create_invoice_checkout_session(
    invoice_id: str,
    current_user: Annotated[any, Depends(get_current_active_user)],
    data: CreateCheckoutSessionRequest | None = None,
):
    _ensure_stripe_configured()

    invoice = await db.invoice.find_unique(
        where={"id": invoice_id},
        include={"subscription": True},
    )
    if not invoice or str(invoice.subscription.userId) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Invoice not found")

    invoice_status = (_enum_value(invoice.status) or "").lower()
    if invoice_status == "paid":
        raise HTTPException(status_code=400, detail="Invoice already paid")
    if invoice_status == "skipped":
        raise HTTPException(status_code=400, detail="Skipped invoice cannot be paid")
    invoice = await _finalize_paid_checkout_invoice_if_available(
        invoice=invoice,
        user=current_user,
        notify_user=False,
    )
    invoice_status = (_enum_value(invoice.status) or "").lower()
    if invoice_status == "paid":
        raise HTTPException(status_code=400, detail="Invoice already paid")
    if invoice_status in {"pending", "failed"}:
        invoice = await reconcile_open_invoice_amount(
            invoice,
            subscription=invoice.subscription,
            user_id=str(current_user.id),
        )
        invoice = await db.invoice.find_unique(
            where={"id": invoice_id},
            include={"subscription": True},
        ) or invoice
        invoice_status = (_enum_value(invoice.status) or "").lower()
        if invoice_status == "skipped":
            raise HTTPException(status_code=400, detail="Invoice no longer requires payment")

    payment_flow = _normalize_checkout_payment_flow(getattr(data, "payment_flow", "card"))

    existing_session_id = str(getattr(invoice, "processorRequestId", "") or "").strip()
    if existing_session_id.startswith("cs_"):
        try:
            existing_session = await _stripe_call(stripe.checkout.Session.retrieve, existing_session_id)
            existing_metadata = _stripe_object_get(existing_session, "metadata", {}) or {}
            existing_flow = str(_stripe_object_get(existing_metadata, "payment_flow") or "").strip().lower()
            if _checkout_session_is_reusable(existing_session) and existing_flow == payment_flow:
                return CreateCheckoutSessionResponse(
                    session_id=str(_stripe_object_get(existing_session, "id")),
                    url=_stripe_object_get(existing_session, "url"),
                )
        except Exception:
            pass

    customer_id = await _get_or_create_stripe_customer(current_user)
    amount = round(_to_float(invoice.calculatedFee), 2)
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Invoice amount must be greater than zero")

    period_label = None
    billing_start = _to_date_string(invoice.billingStartDate)
    billing_end = _to_date_string(invoice.billingEndDate)
    if billing_start and billing_end:
        period_label = f"{billing_start} to {billing_end}"
    elif billing_start:
        period_label = billing_start
    elif billing_end:
        period_label = billing_end
    else:
        period_label = _to_datetime_string(invoice.createdAt) or str(invoice.id)

    currency = _resolve_checkout_currency()
    amount_minor = max(0, int(round(amount * 100)))
    expected_payment_breakdown = build_expected_payment_breakdown(
        amount,
        expected_payment_amount=amount,
        expected_payment_currency=currency.upper(),
        configured_exchange_rate=1.0,
    )
    payment_method_options = None

    if payment_flow == "promptpay":
        currency = get_promptpay_currency().strip().lower() or "thb"
        try:
            promptpay_exchange_rate = await resolve_promptpay_exchange_rate(require_config=True)
            payment_amount = convert_invoice_amount_to_promptpay_currency(
                amount,
                require_config=True,
                exchange_rate=promptpay_exchange_rate,
            )
            amount_minor = promptpay_amount_to_minor_units(
                amount,
                require_config=True,
                exchange_rate=promptpay_exchange_rate,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=500, detail=str(error))

        if payment_amount < PROMPTPAY_MIN_AMOUNT_THB:
            raise HTTPException(
                status_code=400,
                detail=f"PromptPay requires a minimum checkout amount of THB {PROMPTPAY_MIN_AMOUNT_THB:,.2f}",
            )

        expected_payment_breakdown = build_expected_payment_breakdown(
            amount,
            expected_payment_amount=payment_amount,
            expected_payment_currency=currency.upper(),
            configured_exchange_rate=promptpay_exchange_rate,
        )
    else:
        payment_method_options = {
            "card": {
                "setup_future_usage": "off_session",
            }
        }

    session_metadata = {
        "invoice_id": str(invoice.id),
        "subscription_id": str(invoice.subscription.id),
        "user_id": str(current_user.id),
        "payment_flow": payment_flow,
        "invoice_currency": get_billing_base_currency(),
        "invoice_amount": f"{amount:.2f}",
        "charge_currency": currency.upper(),
        "charge_amount": f"{round(_to_float(expected_payment_breakdown.get('expected_payment_amount')), 2):.2f}",
        "configured_exchange_rate": f"{_to_float(expected_payment_breakdown.get('configured_exchange_rate'), 1.0):.6f}",
    }

    session_payload = {
        "mode": "payment",
        "customer": customer_id,
        "success_url": _build_checkout_success_url(str(invoice.id)),
        "cancel_url": _build_checkout_cancel_url(str(invoice.id)),
        "line_items": [
            {
                "quantity": 1,
                "price_data": {
                    "currency": currency,
                    "unit_amount": amount_minor,
                    "product_data": {
                        "name": "SmarfRobotTrade subscription invoice",
                        "description": f"Billing period {period_label}",
                    },
                },
            }
        ],
        "metadata": session_metadata,
        "payment_intent_data": {
            "metadata": session_metadata,
        },
        "locale": "auto",
        "payment_method_types": _resolve_checkout_payment_method_types(payment_flow),
    }
    if payment_method_options:
        session_payload["payment_method_options"] = payment_method_options

    try:
        session = await _stripe_call(stripe.checkout.Session.create, **session_payload)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(error)}")

    session_id = str(_stripe_object_get(session, "id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=500, detail="Stripe did not return a checkout session id")

    await db.invoice.update(
        where={"id": str(invoice.id)},
        data={
            "processorRequestId": session_id,
            "paymentBreakdown": _to_prisma_json(expected_payment_breakdown),
        },
    )

    return CreateCheckoutSessionResponse(
        session_id=session_id,
        url=_stripe_object_get(session, "url"),
    )


@subscription_router.post(
    "/invoices/{invoice_id}/checkout-confirm",
    response_model=InvoiceResponse,
)
async def confirm_invoice_checkout(
    invoice_id: str,
    current_user: Annotated[any, Depends(get_current_active_user)],
    session_id: str | None = None,
):
    invoice = await db.invoice.find_unique(
        where={"id": invoice_id},
        include={"subscription": True},
    )
    if not invoice or str(invoice.subscription.userId) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Invoice not found")

    invoice = await _finalize_paid_checkout_invoice_if_available(
        invoice=invoice,
        user=current_user,
        preferred_session_id=session_id,
        notify_user=False,
    )

    if str(_enum_value(getattr(invoice, "status", None)) or "").strip().lower() in {"pending", "failed"}:
        invoice = await reconcile_open_invoice_amount(
            invoice,
            subscription=invoice.subscription,
            user_id=str(current_user.id),
        )

    invoice = await db.invoice.find_unique(
        where={"id": invoice_id},
        include={"subscription": True},
    ) or invoice
    return _map_invoice_response(invoice)


@subscription_router.post("/invoices/{invoice_id}/pay")
async def pay_invoice(
    invoice_id: str,
    current_user: Annotated[any, Depends(get_current_active_user)],
    data: PayInvoiceRequest | None = None,
):
    invoice = await db.invoice.find_unique(
        where={"id": invoice_id},
        include={"subscription": True},
    )
    if not invoice or str(invoice.subscription.userId) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Invoice not found")

    invoice_status = (_enum_value(invoice.status) or "").lower()
    if invoice_status == "paid":
        return {"message": "Invoice already paid"}
    if invoice_status == "skipped":
        raise HTTPException(status_code=400, detail="Skipped invoice cannot be paid")

    try:
        await pay_invoice_now(
            invoice,
            user=current_user,
            selected_payment_method_id=(data.payment_method_id if data else None),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {"message": "Invoice paid successfully"}


@subscription_router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    _ensure_stripe_configured()
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret is not configured")

    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")

    try:
        event = stripe.Webhook.construct_event(payload, signature, STRIPE_WEBHOOK_SECRET)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid Stripe webhook payload: {str(error)}")

    event_type = str(_stripe_object_get(event, "type") or "").strip()
    if event_type not in {
        "checkout.session.completed",
        "checkout.session.async_payment_succeeded",
        "checkout.session.async_payment_failed",
        "checkout.session.expired",
    }:
        return {"received": True}

    session = _stripe_object_get(_stripe_object_get(event, "data", {}) or {}, "object", {}) or {}
    session_metadata = _stripe_object_get(session, "metadata", {}) or {}
    invoice_id = str(_stripe_object_get(session_metadata, "invoice_id") or "").strip()
    if not invoice_id:
        return {"received": True}

    invoice = await db.invoice.find_unique(
        where={"id": invoice_id},
        include={"subscription": True},
    )
    if not invoice:
        return {"received": True}
    if str(_enum_value(invoice.status) or "").strip().lower() == "paid":
        return {"received": True}

    user = await db.user.find_unique(where={"id": str(invoice.subscription.userId)})
    if not user:
        return {"received": True}

    if event_type in {"checkout.session.async_payment_failed", "checkout.session.expired"}:
        await _finalize_checkout_invoice(
            invoice=invoice,
            user=user,
            stripe_session=session,
            target_status="failed",
            notify_user=event_type == "checkout.session.async_payment_failed",
        )
        return {"received": True}

    payment_status = str(_stripe_object_get(session, "payment_status") or "").strip().lower()
    if event_type == "checkout.session.completed" and payment_status != "paid":
        await _finalize_checkout_invoice(
            invoice=invoice,
            user=user,
            stripe_session=session,
            target_status="pending",
            notify_user=False,
        )
        return {"received": True}

    await _finalize_checkout_invoice(
        invoice=invoice,
        user=user,
        stripe_session=session,
        target_status="paid",
        notify_user=True,
    )
    return {"received": True}


def _map_billing_config(config) -> AdminBillingConfigResponse:
    return AdminBillingConfigResponse(
        config_id=int(config.configId) if config else None,
        default_fee_type=_enum_value(config.defaultFeeType) if config and config.defaultFeeType else "percentage",
        default_collection_mode=(
            _enum_value(getattr(config, "defaultCollectionMode", None))
            if config and getattr(config, "defaultCollectionMode", None)
            else "automatic"
        ),
        default_fee_value=round(_to_float(config.defaultFeeValue), 2) if config else 20.0,
        default_min_threshold=round(_to_float(config.defaultMinThreshold), 2) if config else 0.0,
        default_next_billing_date=_to_date_string(getattr(config, "defaultNextBillingDate", None)) if config else None,
        updated_at=_to_datetime_string(config.updatedAt) if config else None,
    )


def _map_admin_invoice_detail(invoice) -> AdminInvoiceDetailResponse:
    return AdminInvoiceDetailResponse(
        id=str(invoice.id),
        billing_start_date=_to_date_string(invoice.billingStartDate),
        billing_end_date=_to_date_string(invoice.billingEndDate),
        total_period_profit=round(_to_float(invoice.totalPeriodProfit), 2),
        calculated_fee=round(_to_float(invoice.calculatedFee), 2),
        status=_enum_value(invoice.status),
        payment_method_used=invoice.paymentMethodUsed,
        stripe_payment_intent_id=invoice.stripePaymentIntentId,
        stripe_charge_id=getattr(invoice, "stripeChargeId", None),
        stripe_balance_txn_id=getattr(invoice, "stripeBalanceTxnId", None),
        processor_request_id=getattr(invoice, "processorRequestId", None),
        payment_breakdown=getattr(invoice, "paymentBreakdown", None),
        payment_method_details=getattr(invoice, "paymentMethodDetails", None),
        payment_error_details=getattr(invoice, "paymentErrorDetails", None),
        paid_at=_to_datetime_string(invoice.paidAt),
        created_at=_to_datetime_string(invoice.createdAt),
    )


@subscription_router.get("/admin/management", response_model=AdminSubscriptionManagementResponse)
async def get_admin_subscription_management(
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    today = date.today()
    config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})
    subscriptions = await db.subscription.find_many(
        order={"createdAt": "desc"},
        include={"user": True},
    )

    normalized_subscriptions = []
    for sub in subscriptions:
        current_next_billing_date = _extract_date(sub.nextBillingDate)
        if not current_next_billing_date:
            resolved_next_billing_date = _resolve_subscription_next_billing_date(
                sub.nextBillingDate,
                config=config,
                today=today,
            )
            sub = await db.subscription.update(
                where={"id": str(sub.id)},
                data={"nextBillingDate": datetime.combine(resolved_next_billing_date, datetime.min.time())},
                include={"user": True},
            )
        normalized_subscriptions.append(sub)

    mapped_subscriptions = [
        AdminSubscriptionItemResponse(
            id=str(sub.id),
            user_id=str(sub.userId),
            user_email=sub.user.email if sub.user else None,
            status=_enum_value(sub.status) or "active",
            collection_mode=_enum_value(getattr(sub, "collectionMode", None)) or "automatic",
            fee_type=_enum_value(sub.feeType) or "percentage",
            fee_value=round(_to_float(sub.feeValue), 2),
            min_profit_threshold=round(_to_float(sub.minProfitThreshold), 2),
            next_billing_date=_to_date_string(sub.nextBillingDate),
            created_at=_to_datetime_string(sub.createdAt),
        )
        for sub in normalized_subscriptions
    ]

    return AdminSubscriptionManagementResponse(
        billing_config=_map_billing_config(config),
        subscriptions=mapped_subscriptions,
    )


@subscription_router.get(
    "/admin/subscriptions/{subscription_id}/invoices",
    response_model=AdminSubscriptionInvoiceListResponse,
)
async def get_admin_subscription_invoices(
    subscription_id: str,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    subscription = await db.subscription.find_unique(
        where={"id": subscription_id},
        include={"user": True},
    )
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    invoices = await db.invoice.find_many(
        where={"subId": subscription_id},
        order={"createdAt": "desc"},
        take=12,
    )

    return AdminSubscriptionInvoiceListResponse(
        subscription_id=str(subscription.id),
        user_id=str(subscription.userId),
        user_email=subscription.user.email if subscription.user else None,
        invoices=[_map_admin_invoice_detail(invoice) for invoice in invoices],
    )


@subscription_router.put("/admin/config", response_model=AdminBillingConfigResponse)
async def update_admin_billing_config(
    data: UpdateBillingConfigRequest,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    fee_type = data.default_fee_type.strip().lower()
    if fee_type not in ADMIN_FEE_TYPES:
        raise HTTPException(status_code=400, detail="default_fee_type must be percentage or fixed")
    collection_mode = data.default_collection_mode.strip().lower()
    if collection_mode not in COLLECTION_MODES:
        raise HTTPException(status_code=400, detail="default_collection_mode must be automatic or manual")
    if data.default_fee_value < 0:
        raise HTTPException(status_code=400, detail="default_fee_value must be greater than or equal to 0")
    if data.default_min_threshold < 0:
        raise HTTPException(status_code=400, detail="default_min_threshold must be greater than or equal to 0")
    default_next_billing_in_payload = "default_next_billing_date" in getattr(data, "model_fields_set", set())
    parsed_default_next_billing_date = None
    if default_next_billing_in_payload:
        parsed_default_next_billing_date = _parse_iso_date_or_none(
            data.default_next_billing_date,
            field_name="default_next_billing_date",
        )
        if parsed_default_next_billing_date and parsed_default_next_billing_date < date.today():
            raise HTTPException(status_code=400, detail="default_next_billing_date must be today or in the future")

    config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})
    payload = {
        "defaultFeeType": fee_type,
        "defaultCollectionMode": collection_mode,
        "defaultFeeValue": Decimal(str(data.default_fee_value)),
        "defaultMinThreshold": Decimal(str(data.default_min_threshold)),
    }
    if default_next_billing_in_payload:
        payload["defaultNextBillingDate"] = (
            datetime.combine(parsed_default_next_billing_date, datetime.min.time())
            if parsed_default_next_billing_date
            else None
        )

    if config:
        updated_config = await db.systembillingconfig.update(
            where={"configId": int(config.configId)},
            data=payload,
        )
    else:
        updated_config = await db.systembillingconfig.create(data=payload)

    return _map_billing_config(updated_config)


@subscription_router.patch("/admin/subscriptions/{subscription_id}/status")
async def update_admin_subscription_status(
    subscription_id: str,
    data: UpdateSubscriptionStatusRequest,
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    next_status = data.status.strip().lower()
    if next_status not in ADMIN_SUB_STATUSES:
        raise HTTPException(status_code=400, detail="status must be active, past_due, or canceled")

    subscription = await db.subscription.find_unique(where={"id": subscription_id})
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    await db.subscription.update(
        where={"id": subscription_id},
        data={"status": next_status},
    )
    if next_status in {"past_due", "canceled"}:
        await suspend_user_bot_runtime(
            str(subscription.userId),
            reason=f"admin_subscription_status_{next_status}",
        )
    return {"message": f"Subscription updated to {next_status}"}


@subscription_router.get("/admin/stats", response_model=AdminSubscriptionStatsResponse)
async def get_admin_subscription_stats(
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    total_users = await db.user.count()
    active_subscriptions = await db.subscription.count(where={"status": "active"})
    total_bot_versions = await db.botversion.count()
    pending_tickets = await db.supportticket.count(where={"status": "open"})

    today = date.today()
    month_start = datetime.combine(today.replace(day=1), datetime.min.time())
    if today.month == 12:
        next_month_date = date(today.year + 1, 1, 1)
    else:
        next_month_date = date(today.year, today.month + 1, 1)
    next_month_start = datetime.combine(next_month_date, datetime.min.time())

    paid_invoices = await db.invoice.find_many(
        where={
            "status": "paid",
            "paidAt": {
                "gte": month_start,
                "lt": next_month_start,
            },
        }
    )
    monthly_revenue = round(sum(_to_float(invoice.calculatedFee) for invoice in paid_invoices), 2)

    return AdminSubscriptionStatsResponse(
        total_users=total_users,
        active_subscriptions=active_subscriptions,
        total_bot_versions=total_bot_versions,
        pending_tickets=pending_tickets,
        monthly_revenue=monthly_revenue,
    )


@subscription_router.post("/admin/process-due", response_model=ProcessDueBillingResponse)
async def process_due_billing_now(
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    summary = await process_all_due_billing(today=date.today())
    return ProcessDueBillingResponse(
        processed_subscriptions=summary.processed_subscriptions,
        created_invoices=summary.created_invoices,
        paid_invoices=summary.paid_invoices,
        pending_invoices=summary.pending_invoices,
        skipped_invoices=summary.skipped_invoices,
        failed_invoices=summary.failed_invoices,
    )
