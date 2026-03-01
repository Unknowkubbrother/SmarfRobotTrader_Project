import asyncio
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..database.client import db
from ..models.subscription_model import (
    AdminBillingConfigResponse,
    AdminSubscriptionItemResponse,
    AdminSubscriptionManagementResponse,
    AdminSubscriptionStatsResponse,
    AttachPaymentMethodRequest,
    CreateSetupIntentResponse,
    InvoiceResponse,
    PaymentMethodResponse,
    SubscriptionResponse,
    SubscriptionSummaryResponse,
    UpdateBillingConfigRequest,
    UpdateSubscriptionStatusRequest,
    WeeklyPreviewResponse,
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


def _to_float(value: Optional[Decimal]) -> float:
    return float(value) if value is not None else 0.0


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


async def _get_or_create_subscription(user_id: str):
    subscription = await db.subscription.find_first(where={"userId": user_id})
    if subscription:
        return subscription

    config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})
    default_fee_type = _enum_value(config.defaultFeeType) if config else None
    if not default_fee_type:
        default_fee_type = "percentage"
    default_fee_value = config.defaultFeeValue if config and config.defaultFeeValue is not None else Decimal("20.00")
    default_min_threshold = (
        config.defaultMinThreshold if config and config.defaultMinThreshold is not None else Decimal("0.00")
    )

    return await db.subscription.create(
        data={
            "userId": user_id,
            "status": "active",
            "feeType": default_fee_type,
            "feeValue": default_fee_value,
            "minProfitThreshold": default_min_threshold,
            "nextBillingDate": datetime.combine(_next_monday(date.today()), datetime.min.time()),
        }
    )


async def _get_or_create_stripe_customer(current_user):
    _ensure_stripe_configured()

    user_id = str(current_user.id)
    customer_id = current_user.stripeCustomerId

    if customer_id:
        try:
            customer = await _stripe_call(stripe.Customer.retrieve, customer_id)
            if customer and not customer.get("deleted", False):
                return customer_id
        except Exception:
            pass

    customer = await _stripe_call(
        stripe.Customer.create,
        email=current_user.email,
        name=current_user.username,
        metadata={"user_id": user_id},
    )
    customer_id = customer.get("id")
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
    provider_method_id = stripe_payment_method.get("id")
    card = stripe_payment_method.get("card") or {}

    existing = await db.userpaymentmethod.find_first(
        where={
            "userId": user_id,
            "providerMethodId": provider_method_id,
        }
    )

    update_data = {
        "type": stripe_payment_method.get("type"),
        "cardLast4": card.get("last4"),
        "cardBrand": card.get("brand"),
        "expiryMonth": card.get("exp_month"),
        "expiryYear": card.get("exp_year"),
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
    invoice_settings = customer.get("invoice_settings") or {}
    default_payment_method_id = invoice_settings.get("default_payment_method")
    if isinstance(default_payment_method_id, dict):
        default_payment_method_id = default_payment_method_id.get("id")

    methods_response = await _stripe_call(
        stripe.PaymentMethod.list,
        customer=stripe_customer_id,
        type="card",
        limit=20,
    )
    methods = methods_response.get("data") or []
    stripe_ids = {method.get("id") for method in methods if method.get("id")}

    for method in methods:
        provider_method_id = method.get("id")
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


@subscription_router.get("/summary", response_model=SubscriptionSummaryResponse)
async def get_subscription_summary(
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    user_id = str(current_user.id)
    today = date.today()

    subscription = await _get_or_create_subscription(user_id)

    if _stripe_enabled() and current_user.stripeCustomerId:
        try:
            await _sync_customer_payment_methods(user_id, current_user.stripeCustomerId)
        except Exception:
            pass

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

    week_start = _start_of_week(today)
    weekly_aggregates = await db.dailyaggregate.find_many(
        where={
            "account": {"userId": user_id},
            "date": {
                "gte": datetime.combine(week_start, datetime.min.time()),
                "lte": datetime.combine(today, datetime.max.time()),
            },
        }
    )

    gross_profit = 0.0
    gross_loss = 0.0

    for aggregate in weekly_aggregates:
        pnl = _to_float(aggregate.dailyNetProfit)
        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

    net_profit = gross_profit - gross_loss
    fee_type = _enum_value(subscription.feeType) or "percentage"
    fee_value = _to_float(subscription.feeValue)
    min_profit_threshold = _to_float(subscription.minProfitThreshold)
    estimated_fee = _calculate_estimated_fee(
        net_profit=net_profit,
        fee_type=fee_type,
        fee_value=fee_value,
        min_profit_threshold=min_profit_threshold,
    )

    mapped_subscription = SubscriptionResponse(
        id=str(subscription.id),
        status=_enum_value(subscription.status) or "active",
        fee_type=fee_type,
        fee_value=round(fee_value, 2),
        min_profit_threshold=round(min_profit_threshold, 2),
        next_billing_date=_to_date_string(subscription.nextBillingDate) or _next_monday(today).isoformat(),
        default_payment_method_id=subscription.defaultPaymentMethodId,
    )

    mapped_invoices = [
        InvoiceResponse(
            id=str(invoice.id),
            billing_start_date=_to_date_string(invoice.billingStartDate),
            billing_end_date=_to_date_string(invoice.billingEndDate),
            total_period_profit=round(_to_float(invoice.totalPeriodProfit), 2),
            calculated_fee=round(_to_float(invoice.calculatedFee), 2),
            status=_enum_value(invoice.status),
            payment_method_used=invoice.paymentMethodUsed,
            paid_at=_to_datetime_string(invoice.paidAt),
            created_at=_to_datetime_string(invoice.createdAt),
        )
        for invoice in invoices
    ]

    mapped_payment_methods = [_map_payment_method(method) for method in payment_methods_sorted]

    weekly_preview = WeeklyPreviewResponse(
        week_start=week_start.isoformat(),
        week_end=today.isoformat(),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        net_profit=round(net_profit, 2),
        estimated_fee=round(estimated_fee, 2),
    )

    return SubscriptionSummaryResponse(
        subscription=mapped_subscription,
        invoices=mapped_invoices,
        payment_methods=mapped_payment_methods,
        weekly_preview=weekly_preview,
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

    client_secret = setup_intent.get("client_secret")
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

    if stripe_payment_method.get("type") != "card":
        raise HTTPException(status_code=400, detail="Only card payment methods are supported")

    attached_customer = stripe_payment_method.get("customer")
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


def _map_billing_config(config) -> AdminBillingConfigResponse:
    return AdminBillingConfigResponse(
        config_id=int(config.configId) if config else None,
        default_fee_type=_enum_value(config.defaultFeeType) if config and config.defaultFeeType else "percentage",
        default_fee_value=round(_to_float(config.defaultFeeValue), 2) if config else 20.0,
        default_min_threshold=round(_to_float(config.defaultMinThreshold), 2) if config else 0.0,
        updated_at=_to_datetime_string(config.updatedAt) if config else None,
    )


@subscription_router.get("/admin/management", response_model=AdminSubscriptionManagementResponse)
async def get_admin_subscription_management(
    current_user: Annotated[any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})
    subscriptions = await db.subscription.find_many(
        order={"createdAt": "desc"},
        include={"user": True},
    )

    mapped_subscriptions = [
        AdminSubscriptionItemResponse(
            id=str(sub.id),
            user_id=str(sub.userId),
            user_email=sub.user.email if sub.user else None,
            status=_enum_value(sub.status) or "active",
            fee_type=_enum_value(sub.feeType) or "percentage",
            fee_value=round(_to_float(sub.feeValue), 2),
            min_profit_threshold=round(_to_float(sub.minProfitThreshold), 2),
            next_billing_date=_to_date_string(sub.nextBillingDate),
            created_at=_to_datetime_string(sub.createdAt),
        )
        for sub in subscriptions
    ]

    return AdminSubscriptionManagementResponse(
        billing_config=_map_billing_config(config),
        subscriptions=mapped_subscriptions,
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
    if data.default_fee_value < 0:
        raise HTTPException(status_code=400, detail="default_fee_value must be greater than or equal to 0")
    if data.default_min_threshold < 0:
        raise HTTPException(status_code=400, detail="default_min_threshold must be greater than or equal to 0")

    config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})
    payload = {
        "defaultFeeType": fee_type,
        "defaultFeeValue": Decimal(str(data.default_fee_value)),
        "defaultMinThreshold": Decimal(str(data.default_min_threshold)),
    }

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
