from __future__ import annotations

import asyncio
import html
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any

from ..database.client import db

try:
    import stripe
except Exception:
    stripe = None

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if stripe is not None and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

logger = logging.getLogger(__name__)

_BILLING_POLL_SECONDS = max(60, int(os.getenv("SUBSCRIPTION_BILLING_POLL_SECONDS", "900")))
_PROCESSABLE_SUB_STATUSES = {"active", "past_due"}


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


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


async def _get_period_profit(user_id: str, period_start: date, period_end: date) -> float:
    rows = await db.dailyaggregate.find_many(
        where={
            "account": {
                "userId": user_id,
                "recordStatus": "active",
            },
            "date": {
                "gte": _datetime_at_start(period_start),
                "lte": _datetime_at_end(period_end),
            },
        }
    )
    return round(sum(_to_float(getattr(row, "dailyNetProfit", None)) for row in rows), 2)


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
            "paidAt": paid_at,
        }
    )


async def _charge_invoice(
    *,
    user: Any,
    subscription: Any,
    local_payment_method: Any,
    amount: float,
    period_start: date,
    period_end: date,
) -> tuple[str, str | None, datetime | None, str]:
    if amount <= 0:
        return ("skipped", None, None, "No billable fee for this period")

    if not _stripe_enabled():
        return ("pending", None, None, "Stripe is not configured on the server")

    provider_method_id = str(getattr(local_payment_method, "providerMethodId", "") or "").strip()
    if not provider_method_id:
        return ("pending", None, None, "Default payment method is missing Stripe provider id")

    stripe_customer_id = str(getattr(user, "stripeCustomerId", "") or "").strip()
    if not stripe_customer_id:
        return ("pending", None, None, "User is missing Stripe customer id")

    amount_cents = max(0, int(round(amount * 100)))
    if amount_cents <= 0:
        return ("skipped", None, None, "Calculated fee rounded down to zero")

    try:
        payment_intent = await _stripe_call(
            stripe.PaymentIntent.create,
            amount=amount_cents,
            currency="usd",
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
            },
        )
    except Exception as error:
        return ("failed", None, None, str(error))

    payment_status = str(payment_intent.get("status") or "").strip().lower()
    payment_intent_id = payment_intent.get("id")

    if payment_status == "succeeded":
        return ("paid", payment_intent_id, datetime.now(timezone.utc), "Charge succeeded")

    if payment_status in {"processing", "requires_capture"}:
        return ("pending", payment_intent_id, None, f"Stripe payment intent status: {payment_status}")

    if payment_status in {"requires_payment_method", "requires_action", "canceled"}:
        return ("failed", payment_intent_id, None, f"Stripe payment intent status: {payment_status}")

    return ("pending", payment_intent_id, None, f"Unhandled Stripe payment intent status: {payment_status or 'unknown'}")


async def process_due_subscription(subscription: Any, *, today: date | None = None) -> list[BillingCycleResult]:
    today = today or date.today()
    subscription_status = str(_enum_value(getattr(subscription, "status", None)) or "active").lower()
    if subscription_status not in _PROCESSABLE_SUB_STATUSES:
        return []

    due_date = _extract_date(getattr(subscription, "nextBillingDate", None))
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

        period_start, period_end = billing_period_for_due_date(next_due_date)
        existing_invoice = await _find_existing_invoice(str(subscription.id), period_start, period_end)
        if existing_invoice:
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(existing_invoice.id),
                    invoice_created=False,
                    status=str(_enum_value(existing_invoice.status) or "pending"),
                    amount=round(_to_float(existing_invoice.calculatedFee), 2),
                    period_start=period_start,
                    period_end=period_end,
                    note="Existing invoice reused",
                )
            )
            next_due_date += timedelta(days=7)
            continue

        net_profit = await _get_period_profit(user_id, period_start, period_end)
        fee_type = str(_enum_value(getattr(subscription, "feeType", None)) or "percentage")
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

        local_payment_method = await _get_default_payment_method(
            user_id=user_id,
            default_method_id=str(getattr(subscription, "defaultPaymentMethodId", "") or "") or None,
        )

        if calculated_fee <= 0:
            invoice = await _create_invoice_record(
                subscription_id=str(subscription.id),
                period_start=period_start,
                period_end=period_end,
                total_period_profit=net_profit,
                calculated_fee=0.0,
                status="skipped",
                payment_method_used=str(getattr(local_payment_method, "id", "") or "") or None,
            )
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(invoice.id),
                    invoice_created=True,
                    status="skipped",
                    amount=0.0,
                    period_start=period_start,
                    period_end=period_end,
                    note="Net profit did not exceed billing threshold",
                )
            )
            next_due_date += timedelta(days=7)
            continue

        if not local_payment_method:
            invoice = await _create_invoice_record(
                subscription_id=str(subscription.id),
                period_start=period_start,
                period_end=period_end,
                total_period_profit=net_profit,
                calculated_fee=calculated_fee,
                status="pending",
            )
            results.append(
                BillingCycleResult(
                    subscription_id=str(subscription.id),
                    invoice_id=str(invoice.id),
                    invoice_created=True,
                    status="pending",
                    amount=calculated_fee,
                    period_start=period_start,
                    period_end=period_end,
                    note="No default payment method available",
                )
            )
            next_due_date += timedelta(days=7)
            continue

        charge_status, payment_intent_id, paid_at, note = await _charge_invoice(
            user=user,
            subscription=subscription,
            local_payment_method=local_payment_method,
            amount=calculated_fee,
            period_start=period_start,
            period_end=period_end,
        )

        invoice = await _create_invoice_record(
            subscription_id=str(subscription.id),
            period_start=period_start,
            period_end=period_end,
            total_period_profit=net_profit,
            calculated_fee=calculated_fee,
            status=charge_status,
            payment_method_used=str(local_payment_method.id),
            stripe_payment_intent_id=payment_intent_id,
            paid_at=paid_at,
        )
        results.append(
            BillingCycleResult(
                subscription_id=str(subscription.id),
                invoice_id=str(invoice.id),
                invoice_created=True,
                status=charge_status,
                amount=calculated_fee,
                period_start=period_start,
                period_end=period_end,
                note=note,
            )
        )
        next_due_date += timedelta(days=7)

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


async def pay_invoice_now(invoice: Any, *, user: Any) -> Any:
    invoice_status = str(_enum_value(getattr(invoice, "status", None)) or "").lower()
    if invoice_status == "paid":
        return invoice

    subscription = getattr(invoice, "subscription", None)
    if not subscription:
        subscription = await db.subscription.find_unique(where={"id": str(invoice.subId)})
    if not subscription:
        raise ValueError("Subscription not found for invoice")

    local_payment_method = await _get_default_payment_method(
        user_id=str(getattr(user, "id", "") or ""),
        default_method_id=str(getattr(subscription, "defaultPaymentMethodId", "") or "") or None,
    )
    if not local_payment_method:
        raise ValueError("No default payment method available")

    amount = round(_to_float(getattr(invoice, "calculatedFee", None), 0.0), 2)
    period_start = _extract_date(getattr(invoice, "billingStartDate", None))
    period_end = _extract_date(getattr(invoice, "billingEndDate", None))
    if not period_start or not period_end:
        raise ValueError("Invoice billing period is incomplete")

    charge_status, payment_intent_id, paid_at, note = await _charge_invoice(
        user=user,
        subscription=subscription,
        local_payment_method=local_payment_method,
        amount=amount,
        period_start=period_start,
        period_end=period_end,
    )
    if charge_status != "paid":
        updated = await db.invoice.update(
            where={"id": str(invoice.id)},
            data={
                "status": "failed" if charge_status == "failed" else "pending",
                "paymentMethodUsed": str(local_payment_method.id),
                "stripePaymentIntentId": payment_intent_id,
            },
        )
        raise ValueError(note or "Unable to collect payment")

    return await db.invoice.update(
        where={"id": str(invoice.id)},
        data={
            "status": "paid",
            "paymentMethodUsed": str(local_payment_method.id),
            "stripePaymentIntentId": payment_intent_id,
            "paidAt": paid_at,
        },
    )


def _safe_text(value: Any) -> str:
    return html.escape(str(value or "").strip())


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
    fee_type = str(_enum_value(getattr(subscription, "feeType", None)) or "percentage")
    fee_value = _to_float(getattr(subscription, "feeValue", None), 0.0)

    if payment_method:
        card_brand = str(getattr(payment_method, "cardBrand", "") or "").upper()
        last4 = str(getattr(payment_method, "cardLast4", "") or "")
        payment_method_text = f"{card_brand} ending in {last4}".strip()
    else:
        payment_method_text = "Not available"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_safe_text(invoice_label)}</title>
</head>
<body style="margin:0; padding:32px; background:#f8fafc; font-family:Arial, sans-serif; color:#0f172a;">
  <div style="max-width:760px; margin:0 auto; background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; overflow:hidden;">
    <div style="padding:28px 32px; background:linear-gradient(135deg, #0f172a, #1d4ed8); color:#ffffff;">
      <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start;">
        <div>
          <div style="font-size:12px; letter-spacing:0.12em; text-transform:uppercase; opacity:0.8;">SmarfRobotTrade</div>
          <h1 style="margin:8px 0 0 0; font-size:30px;">Invoice</h1>
        </div>
        <div style="text-align:right;">
          <div style="font-size:18px; font-weight:700;">{_safe_text(invoice_label)}</div>
          <div style="margin-top:8px; font-size:13px;">Status: {_safe_text(status)}</div>
        </div>
      </div>
    </div>

    <div style="padding:32px;">
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
            <th style="padding:12px; border:1px solid #e2e8f0; text-align:left;">Notes</th>
            <th style="padding:12px; border:1px solid #e2e8f0; text-align:right;">Amount</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding:12px; border:1px solid #e2e8f0;">Weekly performance fee</td>
            <td style="padding:12px; border:1px solid #e2e8f0;">{_safe_text(fee_type)} fee at {fee_value:.2f}{'%' if fee_type == 'percentage' else ' USD'}</td>
            <td style="padding:12px; border:1px solid #e2e8f0; text-align:right;">${calculated_fee:,.2f}</td>
          </tr>
        </tbody>
      </table>

      <div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:20px;">
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:12px; background:#f8fafc;">
          <div style="font-size:12px; text-transform:uppercase; color:#64748b;">Billing Summary</div>
          <div style="margin-top:10px;">Net profit for period: <strong>${total_period_profit:,.2f}</strong></div>
          <div style="margin-top:6px;">Payment method: <strong>{_safe_text(payment_method_text)}</strong></div>
        </div>
        <div style="padding:18px; border:1px solid #e2e8f0; border-radius:12px; background:#eff6ff;">
          <div style="font-size:12px; text-transform:uppercase; color:#1d4ed8;">Total Due</div>
          <div style="margin-top:10px; font-size:28px; font-weight:800; color:#0f172a;">${calculated_fee:,.2f}</div>
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
