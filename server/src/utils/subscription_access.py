from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from ..database.client import db
from .mt5_bot_runner import BotRunnerError, run_bot_instance_action

logger = logging.getLogger(__name__)

ACTIVE_RECORD_STATUS = "active"
BLOCKING_SUB_STATUSES = {"past_due", "canceled"}
UNRESOLVED_INVOICE_STATUSES = {"pending", "failed"}


def _enum_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _normalize_status(value: Any, default: str) -> str:
    normalized = str(_enum_value(value) or default).strip().lower()
    return normalized or default


def _build_block_message(subscription_status: str, invoice_status: str | None = None) -> str:
    if subscription_status == "canceled":
        return "Your subscription is canceled. Reactivate billing before using bots again."
    if invoice_status == "failed":
        return "Your subscription is past due because the latest invoice payment failed. Pay the outstanding invoice or ask admin to skip it before using bots again."
    return "Your subscription is past due. Pay the outstanding invoice or ask admin to skip it before using bots again."


def _build_missing_payment_method_message(invoice_status: str | None = None) -> str:
    if invoice_status in {"pending", "failed"}:
        return "Add a payment method, then pay the outstanding invoice or ask admin to skip it before connecting trading accounts or using bots."
    return "Add a payment method before connecting trading accounts or using bots."


@dataclass
class SubscriptionAccessState:
    subscription_id: str | None
    subscription_status: str | None
    blocked: bool
    block_message: str | None
    unpaid_invoice_id: str | None
    unpaid_invoice_status: str | None
    has_active_payment_method: bool


async def get_latest_subscription_for_user(user_id: str, *, include: dict[str, Any] | None = None):
    return await db.subscription.find_first(
        where={"userId": str(user_id)},
        include=include,
        order={"createdAt": "desc"},
    )


async def get_latest_unresolved_invoice(subscription_id: str):
    return await db.invoice.find_first(
        where={
            "subId": str(subscription_id),
            "status": {"in": list(UNRESOLVED_INVOICE_STATUSES)},
        },
        order={"createdAt": "desc"},
    )


async def user_has_active_payment_method(user_id: str) -> bool:
    method = await db.userpaymentmethod.find_first(
        where={
            "userId": str(user_id),
            "isActive": True,
        }
    )
    return method is not None


async def get_user_subscription_access_state(user_id: str) -> SubscriptionAccessState:
    has_active_payment_method = await user_has_active_payment_method(str(user_id))
    subscription = await get_latest_subscription_for_user(str(user_id))
    if not subscription:
        return SubscriptionAccessState(
            subscription_id=None,
            subscription_status=None,
            blocked=not has_active_payment_method,
            block_message=None if has_active_payment_method else _build_missing_payment_method_message(),
            unpaid_invoice_id=None,
            unpaid_invoice_status=None,
            has_active_payment_method=has_active_payment_method,
        )

    current_status = _normalize_status(getattr(subscription, "status", None), "active")
    unresolved_invoice = await get_latest_unresolved_invoice(str(subscription.id))
    unresolved_status = (
        _normalize_status(getattr(unresolved_invoice, "status", None), "pending")
        if unresolved_invoice
        else None
    )

    if current_status == "canceled":
        effective_status = "canceled"
    elif unresolved_invoice is not None or current_status == "past_due":
        effective_status = "past_due"
    else:
        effective_status = "active"

    blocked = effective_status in BLOCKING_SUB_STATUSES or not has_active_payment_method
    if effective_status in BLOCKING_SUB_STATUSES:
        block_message = _build_block_message(effective_status, unresolved_status)
    elif not has_active_payment_method:
        block_message = _build_missing_payment_method_message(unresolved_status)
    else:
        block_message = None

    return SubscriptionAccessState(
        subscription_id=str(subscription.id),
        subscription_status=effective_status,
        blocked=blocked,
        block_message=block_message,
        unpaid_invoice_id=str(getattr(unresolved_invoice, "id", "") or "") or None,
        unpaid_invoice_status=unresolved_status,
        has_active_payment_method=has_active_payment_method,
    )


async def assert_user_subscription_allows_bot_usage(user_id: str, *, action_label: str = "use bots") -> SubscriptionAccessState:
    access_state = await get_user_subscription_access_state(user_id)
    if not access_state.blocked:
        return access_state

    message = access_state.block_message or "Your subscription does not allow bot access right now."
    raise HTTPException(
        status_code=403 if access_state.subscription_status == "canceled" else 402,
        detail=f"Cannot {action_label}. {message}",
    )


async def suspend_user_bot_runtime(user_id: str, *, reason: str) -> dict[str, int]:
    bot_configs = await db.botconfiguration.find_many(
        where={
            "recordStatus": ACTIVE_RECORD_STATUS,
            "account": {
                "userId": str(user_id),
                "recordStatus": ACTIVE_RECORD_STATUS,
            },
            "OR": [
                {"containerStatus": "running"},
                {"containerStatus": "starting"},
                {"isActive": True},
                {"dockerContainerId": {"not": None}},
            ],
        }
    )

    if not bot_configs:
        return {"stopped": 0, "failed": 0}

    stopped_ids: list[str] = []
    failed = 0
    for bot_config in bot_configs:
        bot_config_id = str(getattr(bot_config, "id", "") or "")
        container_status = _normalize_status(getattr(bot_config, "containerStatus", None), "stopped")
        docker_container_id = str(getattr(bot_config, "dockerContainerId", "") or "").strip() or None
        try:
            await asyncio.to_thread(
                run_bot_instance_action,
                action="stop",
                instance_name=bot_config_id,
                timeout_sec=300,
            )
            stopped_ids.append(bot_config_id)
        except BotRunnerError as error:
            if container_status != "running" and not docker_container_id:
                stopped_ids.append(bot_config_id)
                continue
            failed += 1
            logger.error(
                "failed to stop bot runtime for config %s while suspending user %s: %s",
                bot_config_id,
                user_id,
                error,
            )
        except Exception:
            if container_status != "running" and not docker_container_id:
                stopped_ids.append(bot_config_id)
                continue
            failed += 1
            logger.exception(
                "unexpected error stopping bot runtime for config %s while suspending user %s",
                bot_config_id,
                user_id,
            )

    if stopped_ids:
        await db.botconfiguration.update_many(
            where={"id": {"in": stopped_ids}},
            data={
                "containerStatus": "stopped",
                "isActive": False,
                "dockerContainerId": None,
            },
        )

    if stopped_ids:
        logger.info(
            "suspended %s bot runtime(s) for user %s (%s)",
            len(stopped_ids),
            user_id,
            reason,
        )

    return {"stopped": len(stopped_ids), "failed": failed}


async def sync_subscription_status_from_invoices(
    subscription_id: str,
    *,
    allow_reactivate: bool = True,
):
    subscription = await db.subscription.find_unique(where={"id": str(subscription_id)})
    if not subscription:
        return None

    current_status = _normalize_status(getattr(subscription, "status", None), "active")
    if current_status == "canceled":
        return subscription

    unresolved_invoice = await get_latest_unresolved_invoice(str(subscription.id))
    if unresolved_invoice:
        next_status = "past_due"
    elif current_status == "past_due" and not allow_reactivate:
        return subscription
    else:
        next_status = "active"

    if next_status != current_status:
        subscription = await db.subscription.update(
            where={"id": str(subscription.id)},
            data={"status": next_status},
        )

    if next_status == "past_due" and current_status != "past_due":
        await suspend_user_bot_runtime(
            str(getattr(subscription, "userId", "") or ""),
            reason=f"subscription_status_changed_to_{next_status}",
        )

    return subscription
