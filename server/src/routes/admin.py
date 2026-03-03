import asyncio
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from lib.untils import send_email

from ..database.client import db
from ..models.admin_model import (
    AdminBotVersionItemResponse,
    AdminStatsResponse,
    AdminUserBillingSummaryResponse,
    AdminUserBotConfigurationItemResponse,
    AdminUserDetailResponse,
    AdminUserInvoiceItemResponse,
    AdminUserItemResponse,
    AdminUserSubscriptionItemResponse,
    AdminUserTradingAccountItemResponse,
    CreateAdminBotVersionRequest,
    PublishAdminBotUpdateRequest,
    UpdateAdminBotConfigurationStatusRequest,
    UpdateAdminBotVersionActiveRequest,
    UpdateAdminBotVersionRequest,
    UpdateAdminUserSubscriptionBillingRequest,
    UpdateAdminUserRoleRequest,
    UpdateAdminUserStatusRequest,
)
from ..utils.mt5_bot_runner import (
    BotRunnerError,
    build_bot_runtime_env,
    build_profile_name,
    decrypt_mt5_password,
    run_bot_instance_action,
)
from ..utils.bot_operation_events import emit_and_store_bot_operation_event
from .authentication import get_current_active_user

admin_router = APIRouter(tags=["Admin"])
logger = logging.getLogger(__name__)

ALLOWED_USER_STATUSES = {"active", "banned", "pending"}
ALLOWED_USER_ROLES = {"user", "admin"}
ALLOWED_BOT_STATUSES = {"running", "stopped"}
ALLOWED_SUB_STATUSES = {"active", "past_due", "canceled"}
ALLOWED_FEE_TYPES = {"percentage", "fixed"}


def _to_float(value: Optional[Decimal]) -> float:
    return float(value) if value is not None else 0.0


def _enum_value(value):
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _to_datetime_string(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    return value.isoformat()


def _to_date_string(value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _parse_iso_date_or_none(raw_value: Optional[str], field_name: str) -> Optional[date]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must be in YYYY-MM-DD format",
        )


def _next_monday(today: date) -> date:
    days_until = (7 - today.weekday()) % 7
    if days_until == 0:
        days_until = 7
    return today + timedelta(days=days_until)


def _extract_date(value) -> Optional[date]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _roll_weekly_forward(base_date: date, today: date) -> date:
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


def _require_admin(current_user):
    role = _enum_value(getattr(current_user, "role", None))
    if role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")


def _map_user_subscription(subscription) -> AdminUserSubscriptionItemResponse:
    return AdminUserSubscriptionItemResponse(
        id=str(subscription.id),
        status=_enum_value(subscription.status) or "active",
        fee_type=_enum_value(subscription.feeType) or "percentage",
        fee_value=round(_to_float(subscription.feeValue), 2),
        min_profit_threshold=round(_to_float(subscription.minProfitThreshold), 2),
        next_billing_date=_to_date_string(subscription.nextBillingDate),
        created_at=_to_datetime_string(subscription.createdAt),
    )


def _clean_release_notes(release_notes: Optional[List[str]]) -> List[str]:
    if not release_notes:
        return []
    return [note.strip() for note in release_notes if note and note.strip()]


def _map_bot_version(version, usage_count: int) -> AdminBotVersionItemResponse:
    runner_profile = None
    try:
        if version.symbol and version.timeframe:
            runner_profile = build_profile_name(version.symbol, version.timeframe)
    except Exception:
        runner_profile = None

    return AdminBotVersionItemResponse(
        id=str(version.modelId),
        label=version.label,
        version_tag=version.versionTag,
        symbol=version.symbol,
        timeframe=version.timeframe,
        runner_profile=runner_profile,
        is_active=bool(getattr(version, "isActive", True)),
        release_notes=version.releaseNotes or [],
        release_date=_to_datetime_string(version.releaseDate),
        usage_count=usage_count,
    )


def _safe_runner_profile(symbol: Optional[str], timeframe: Optional[str]) -> str:
    if not symbol or not timeframe:
        return "-"
    try:
        return build_profile_name(symbol, timeframe)
    except Exception:
        return "-"


def _is_notification_allowed(notification_config) -> bool:
    if not notification_config:
        return True
    value = getattr(notification_config, "emailNotificationEnable", None)
    if value is None:
        return True
    return bool(value)


def _runner_error_message(prefix: str, exc: BotRunnerError) -> str:
    stderr = str(getattr(exc, "stderr", "") or "").strip()
    stdout = str(getattr(exc, "stdout", "") or "").strip()
    if stderr:
        return f"{prefix}: {exc}. stderr={stderr}"
    if stdout:
        return f"{prefix}: {exc}. stdout={stdout}"
    return f"{prefix}: {exc}"


def _extract_admin_runtime_context(
    bot_configuration,
    image_override: str | None = None,
    bot_version_override=None,
) -> dict[str, str | None]:
    account = getattr(bot_configuration, "account", None)
    if not account:
        raise HTTPException(status_code=400, detail="Trading account is missing for this bot.")

    mt5_login = str(getattr(account, "mt5LoginId", "") or "").strip()
    mt5_server = str(getattr(account, "serverName", "") or "").strip()
    encrypted_password = str(getattr(account, "mt5Password", "") or "").strip()
    if not mt5_login or not mt5_server or not encrypted_password:
        raise HTTPException(
            status_code=400,
            detail="Trading account credentials are incomplete. Please update account login/password/server first.",
        )

    bot_version = bot_version_override or getattr(bot_configuration, "botVersion", None)
    if not bot_version:
        raise HTTPException(status_code=400, detail="Bot version is missing for this bot.")

    live_symbol = str(getattr(bot_version, "symbol", "") or "").strip().upper()
    live_timeframe = str(getattr(bot_version, "timeframe", "") or "").strip().upper()
    if not live_symbol or not live_timeframe:
        raise HTTPException(
            status_code=400,
            detail="Bot version must include symbol and timeframe to run docker profile.",
        )

    try:
        mt5_password = decrypt_mt5_password(encrypted_password)
    except BotRunnerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    image_ref = (
        str(image_override).strip()
        if image_override and str(image_override).strip()
        else str(getattr(bot_version, "dockerImageId", "") or "").strip() or None
    )

    try:
        profile_name = build_profile_name(live_symbol, live_timeframe)
    except BotRunnerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runtime_env = build_bot_runtime_env(
        bot_config_id=str(bot_configuration.id),
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        live_symbol=live_symbol,
        live_timeframe=live_timeframe,
        docker_image_id=image_ref,
    )

    return {
        "profile_name": profile_name,
        "docker_image_id": image_ref,
        "runtime_env": runtime_env,
    }


async def _emit_admin_lifecycle_event(
    bot_config_id: str,
    action: str,
    phase: str,
    detail: str,
    status: str | None = None,
    metadata: dict | None = None,
    owner_user_id: str | None = None,
) -> None:
    try:
        await emit_and_store_bot_operation_event(
            bot_config_id=bot_config_id,
            action=action,
            phase=phase,
            detail=detail,
            status=status,
            source="admin",
            metadata=metadata,
            owner_user_id=owner_user_id,
        )
    except Exception as exc:
        logger.warning(
            "failed to broadcast admin bot lifecycle event bot=%s action=%s phase=%s: %s",
            bot_config_id,
            action,
            phase,
            exc,
        )


def _build_bot_update_email_html(
    bot_label: str,
    version_tag: str,
    runner_profile: str,
    release_notes: List[str],
) -> str:
    notes_html = "".join(f"<li>{note}</li>" for note in release_notes) if release_notes else "<li>General improvements</li>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bot update available</title>
</head>
<body style="font-family: Arial, sans-serif; background:#f5f7fb; padding:24px;">
  <div style="max-width:620px; margin:0 auto; background:#ffffff; border-radius:12px; padding:24px; border:1px solid #e5e7eb;">
    <h2 style="margin:0 0 10px 0; color:#0f172a;">Bot update available</h2>
    <p style="margin:0 0 14px 0; color:#334155;">
      {bot_label} has a new release <strong>{version_tag}</strong>.
    </p>
    <p style="margin:0 0 10px 0; color:#334155;">Runner profile: <code>{runner_profile}</code></p>
    <p style="margin:0 0 8px 0; color:#0f172a;"><strong>Release notes</strong></p>
    <ul style="margin:0 0 18px 16px; color:#334155;">
      {notes_html}
    </ul>
    <p style="margin:0; color:#334155;">
      Go to Bot Control to apply the update. The system will restart your bot to use the latest version.
    </p>
  </div>
</body>
</html>"""


async def _notify_users_for_bot_update(
    model_id: str,
    version,
    release_notes: List[str],
) -> tuple[int, int]:
    source_where = {"modelId": {"not": model_id}}
    if version.symbol:
        source_where["symbol"] = version.symbol
    if version.timeframe:
        source_where["timeframe"] = version.timeframe
    if not version.symbol and not version.timeframe and version.label:
        source_where["label"] = version.label

    old_versions = []
    if version.symbol or version.timeframe or version.label:
        old_versions = await db.botversion.find_many(where=source_where)

    old_model_ids = [str(item.modelId) for item in old_versions]

    notify_filters = []
    if old_model_ids:
        notify_filters.append({"modelId": {"in": old_model_ids}})

    latest_version_tag = getattr(version, "versionTag", None)
    if latest_version_tag:
        notify_filters.append(
            {
                "modelId": model_id,
                "OR": [
                    {"installedVersionTag": None},
                    {"installedVersionTag": {"not": latest_version_tag}},
                ],
            }
        )
    else:
        notify_filters.append({"modelId": model_id})

    if not notify_filters:
        return 0, 0

    bot_configurations = await db.botconfiguration.find_many(
        where={"OR": notify_filters},
        include={
            "account": {
                "include": {
                    "user": {
                        "include": {
                            "notificationConfig": True,
                        }
                    }
                }
            }
        },
    )

    user_map = {}
    for config in bot_configurations:
        account = getattr(config, "account", None)
        user = getattr(account, "user", None) if account else None
        if not user:
            continue
        user_status = _enum_value(getattr(user, "status", None))
        if user_status == "banned":
            continue
        if not _is_notification_allowed(getattr(user, "notificationConfig", None)):
            continue
        user_map[str(user.id)] = user

    if not user_map:
        return 0, 0

    bot_label = version.label or "Trading Bot"
    version_tag = version.versionTag or "-"
    note_excerpt = release_notes[0] if release_notes else "A new bot update is ready."
    notification_title = f"Bot update: {bot_label}"
    notification_message = (
        f"New version {version_tag} is available. {note_excerpt} "
        f"Open Bot Control and click Update Bot."
    )

    emails_sent = 0
    for user in user_map.values():
        await db.notification.create(
            data={
                "userId": str(user.id),
                "title": notification_title,
                "message": notification_message,
                "relatedLink": "/bot-control",
            }
        )

        try:
            if getattr(user, "email", None):
                send_email(
                    to_email=user.email,
                    subject=f"Bot update available ({version_tag}) - SmarfRobotTrade",
                    html_content=_build_bot_update_email_html(
                        bot_label=bot_label,
                        version_tag=version_tag,
                        runner_profile=_safe_runner_profile(version.symbol, version.timeframe),
                        release_notes=release_notes,
                    ),
                )
                emails_sent += 1
        except Exception as exc:
            print(f"[admin] failed to send bot update email to {getattr(user, 'email', None)}: {exc}")

    return len(user_map), emails_sent


async def _stop_active_bots_using_version(model_id: str) -> int:
    active_bots_count = await db.botconfiguration.count(
        where={
            "modelId": model_id,
            "OR": [
                {"containerStatus": "running"},
                {"isActive": True},
            ],
        }
    )

    if active_bots_count > 0:
        await db.botconfiguration.update_many(
            where={
                "modelId": model_id,
                "OR": [
                    {"containerStatus": "running"},
                    {"isActive": True},
                ],
            },
            data={
                "containerStatus": "stopped",
                "isActive": False,
            },
        )

    return active_bots_count


@admin_router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    total_users = await db.user.count()
    total_mt5_accounts = await db.tradingaccount.count()
    total_bot_versions = await db.botversion.count()
    active_subscriptions = await db.subscription.count(where={"status": "active"})
    pending_tickets = await db.supportticket.count(where={"status": "open"})
    running_bots = await db.botconfiguration.count(
        where={
            "OR": [
                {"containerStatus": "running"},
                {"isActive": True},
            ]
        }
    )

    today = date.today()
    month_start = datetime.combine(today.replace(day=1), datetime.min.time())
    if today.month == 12:
        next_month = date(today.year + 1, 1, 1)
    else:
        next_month = date(today.year, today.month + 1, 1)
    next_month_start = datetime.combine(next_month, datetime.min.time())

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

    return AdminStatsResponse(
        total_users=total_users,
        total_mt5_accounts=total_mt5_accounts,
        total_bot_versions=total_bot_versions,
        active_subscriptions=active_subscriptions,
        pending_tickets=pending_tickets,
        monthly_revenue=monthly_revenue,
        running_bots=running_bots,
    )


@admin_router.get("/users", response_model=List[AdminUserItemResponse])
async def get_admin_users(
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    users = await db.user.find_many(order={"createdAt": "desc"})
    return [
        AdminUserItemResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=_enum_value(user.role) or "user",
            status=_enum_value(user.status) or "active",
            created_at=_to_datetime_string(user.createdAt) or "",
            is_onboarding_completed=bool(user.isOnboardingCompleted),
        )
        for user in users
    ]


@admin_router.get("/users/{user_id}/detail", response_model=AdminUserDetailResponse)
async def get_admin_user_detail(
    user_id: str,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    target_user = await db.user.find_unique(where={"id": user_id})
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    trading_accounts = await db.tradingaccount.find_many(
        where={"userId": user_id},
        order={"createdAt": "desc"},
        include={
            "botConfigurations": {
                "include": {
                    "botVersion": True,
                }
            }
        },
    )

    mapped_accounts: List[AdminUserTradingAccountItemResponse] = []
    total_balance = 0.0

    for account in trading_accounts:
        balance = _to_float(account.balance)
        total_balance += balance

        mapped_bots: List[AdminUserBotConfigurationItemResponse] = []
        running_bots = 0
        active_bots = 0

        for config in account.botConfigurations:
            container_status = _enum_value(config.containerStatus)
            is_active = bool(config.isActive)

            if container_status in {"running", "starting"}:
                running_bots += 1
            if is_active:
                active_bots += 1

            mapped_bots.append(
                AdminUserBotConfigurationItemResponse(
                    id=str(config.id),
                    bot_instance_id=int(config.botInstanceId),
                    model_id=str(config.modelId),
                    label=config.botVersion.label if config.botVersion else None,
                    symbol=config.botVersion.symbol if config.botVersion else None,
                    timeframe=config.botVersion.timeframe if config.botVersion else None,
                    container_status=container_status,
                    is_active=is_active,
                    updated_at=_to_datetime_string(config.updatedAt),
                )
            )

        mapped_accounts.append(
            AdminUserTradingAccountItemResponse(
                id=str(account.id),
                mt5_login_id=account.mt5LoginId,
                broker_name=account.brokerName,
                server_name=account.serverName,
                balance=round(balance, 2),
                equity=round(_to_float(account.equity), 2),
                running_bots=running_bots,
                active_bots=active_bots,
                bots=mapped_bots,
            )
        )

    subscriptions = await db.subscription.find_many(
        where={"userId": user_id},
        order={"createdAt": "desc"},
    )
    today = date.today()
    billing_config = await db.systembillingconfig.find_first(order={"updatedAt": "desc"})
    normalized_subscriptions = []
    for sub in subscriptions:
        current_next_billing_date = _extract_date(sub.nextBillingDate)
        resolved_next_billing_date = _resolve_subscription_next_billing_date(
            sub.nextBillingDate,
            config=billing_config,
            today=today,
        )
        if current_next_billing_date != resolved_next_billing_date:
            sub = await db.subscription.update(
                where={"id": str(sub.id)},
                data={"nextBillingDate": datetime.combine(resolved_next_billing_date, datetime.min.time())},
            )
        normalized_subscriptions.append(sub)

    mapped_subscriptions = [_map_user_subscription(subscription) for subscription in normalized_subscriptions]
    subscription_ids = [str(sub.id) for sub in normalized_subscriptions]

    pending_count = 0
    paid_count = 0
    pending_amount = 0.0
    paid_amount = 0.0
    recent_invoices: List[AdminUserInvoiceItemResponse] = []

    if subscription_ids:
        pending_invoices = await db.invoice.find_many(
            where={"subId": {"in": subscription_ids}, "status": "pending"}
        )
        paid_invoices = await db.invoice.find_many(
            where={"subId": {"in": subscription_ids}, "status": "paid"}
        )
        latest_invoices = await db.invoice.find_many(
            where={"subId": {"in": subscription_ids}},
            order={"createdAt": "desc"},
            take=12,
        )

        pending_count = len(pending_invoices)
        paid_count = len(paid_invoices)
        pending_amount = round(sum(_to_float(invoice.calculatedFee) for invoice in pending_invoices), 2)
        paid_amount = round(sum(_to_float(invoice.calculatedFee) for invoice in paid_invoices), 2)

        recent_invoices = [
            AdminUserInvoiceItemResponse(
                id=str(invoice.id),
                subscription_id=str(invoice.subId),
                status=_enum_value(invoice.status),
                amount=round(_to_float(invoice.calculatedFee), 2),
                created_at=_to_datetime_string(invoice.createdAt),
                paid_at=_to_datetime_string(invoice.paidAt),
                billing_start_date=_to_date_string(invoice.billingStartDate),
                billing_end_date=_to_date_string(invoice.billingEndDate),
            )
            for invoice in latest_invoices
        ]

    return AdminUserDetailResponse(
        id=str(target_user.id),
        username=target_user.username,
        email=target_user.email,
        role=_enum_value(target_user.role) or "user",
        status=_enum_value(target_user.status) or "active",
        created_at=_to_datetime_string(target_user.createdAt) or "",
        is_onboarding_completed=bool(target_user.isOnboardingCompleted),
        total_accounts=len(mapped_accounts),
        total_balance=round(total_balance, 2),
        pending_bills=pending_count,
        trading_accounts=mapped_accounts,
        subscriptions=mapped_subscriptions,
        billing=AdminUserBillingSummaryResponse(
            pending_count=pending_count,
            paid_count=paid_count,
            pending_amount=pending_amount,
            paid_amount=paid_amount,
            recent_invoices=recent_invoices,
        ),
    )


@admin_router.patch("/users/{user_id}/status")
async def update_admin_user_status(
    user_id: str,
    data: UpdateAdminUserStatusRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    status_value = data.status.strip().lower()
    if status_value not in ALLOWED_USER_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="status must be active, banned, or pending",
        )

    target_user = await db.user.find_unique(where={"id": user_id})
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.user.update(
        where={"id": user_id},
        data={"status": status_value},
    )
    return {"message": f"User status updated to {status_value}"}


@admin_router.patch("/users/{user_id}/role")
async def update_admin_user_role(
    user_id: str,
    data: UpdateAdminUserRoleRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    role_value = data.role.strip().lower()
    if role_value not in ALLOWED_USER_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="role must be user or admin",
        )

    target_user = await db.user.find_unique(where={"id": user_id})
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.user.update(
        where={"id": user_id},
        data={"role": role_value},
    )
    return {"message": f"User role updated to {role_value}"}


@admin_router.patch(
    "/users/{user_id}/subscriptions/{subscription_id}/billing",
    response_model=AdminUserSubscriptionItemResponse,
)
async def update_admin_user_subscription_billing(
    user_id: str,
    subscription_id: str,
    data: UpdateAdminUserSubscriptionBillingRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    fee_type = data.fee_type.strip().lower()
    if fee_type not in ALLOWED_FEE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="fee_type must be percentage or fixed",
        )
    if data.fee_value < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="fee_value must be greater than or equal to 0",
        )
    if data.min_profit_threshold < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_profit_threshold must be greater than or equal to 0",
        )
    next_billing_in_payload = "next_billing_date" in getattr(data, "model_fields_set", set())
    parsed_next_billing_date = None
    if next_billing_in_payload:
        parsed_next_billing_date = _parse_iso_date_or_none(
            data.next_billing_date,
            field_name="next_billing_date",
        )

    subscription = await db.subscription.find_first(
        where={"id": subscription_id, "userId": user_id}
    )
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found for this user",
        )

    current_status = _enum_value(subscription.status) or "active"
    if current_status not in ALLOWED_SUB_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subscription status is invalid for update",
        )

    update_payload = {
        "feeType": fee_type,
        "feeValue": Decimal(str(data.fee_value)),
        "minProfitThreshold": Decimal(str(data.min_profit_threshold)),
    }
    if next_billing_in_payload:
        update_payload["nextBillingDate"] = (
            datetime.combine(parsed_next_billing_date, datetime.min.time())
            if parsed_next_billing_date
            else None
        )

    updated_subscription = await db.subscription.update(
        where={"id": subscription_id},
        data=update_payload,
    )

    return _map_user_subscription(updated_subscription)


@admin_router.patch("/users/{user_id}/invoices/{invoice_id}/skip")
async def skip_admin_user_invoice(
    user_id: str,
    invoice_id: str,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    invoice = await db.invoice.find_unique(
        where={"id": invoice_id},
        include={"subscription": True},
    )
    if not invoice or str(invoice.subscription.userId) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice not found for this user",
        )

    invoice_status = _enum_value(invoice.status)
    if invoice_status == "paid":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Paid invoice cannot be skipped",
        )

    if invoice_status == "skipped":
        return {"message": "Invoice already skipped"}

    await db.invoice.update(
        where={"id": invoice_id},
        data={"status": "skipped"},
    )
    return {"message": "Invoice skipped"}


@admin_router.patch("/users/{user_id}/bot-configurations/{bot_configuration_id}/status")
async def update_admin_user_bot_configuration_status(
    user_id: str,
    bot_configuration_id: str,
    data: UpdateAdminBotConfigurationStatusRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    status_value = data.status.strip().lower()
    if status_value not in ALLOWED_BOT_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="status must be running or stopped",
        )

    bot_configuration = await db.botconfiguration.find_unique(
        where={"id": bot_configuration_id},
        include={"account": True, "botVersion": True},
    )
    if not bot_configuration or str(bot_configuration.account.userId) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot configuration not found for this user",
        )
    if status_value == "running" and bot_configuration.botVersion and not bool(getattr(bot_configuration.botVersion, "isActive", True)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot run bot. Its version is inactive.",
        )

    runner_result = None
    if status_value == "running":
        runtime = _extract_admin_runtime_context(bot_configuration)
        current_status = _enum_value(getattr(bot_configuration, "containerStatus", None))
        action = "restart" if current_status == "running" or bool(bot_configuration.isActive) else "start"
        await db.botconfiguration.update(
            where={"id": bot_configuration_id},
            data={
                "containerStatus": "starting",
            },
        )
        await _emit_admin_lifecycle_event(
            bot_config_id=str(bot_configuration_id),
            action=action,
            phase="requested",
            detail="Bot runtime start requested by admin",
            status="running",
            metadata={"admin_user_id": str(getattr(current_user, "id", "") or "").strip() or None},
            owner_user_id=str(user_id),
        )

        try:
            runner_result = await asyncio.to_thread(
                run_bot_instance_action,
                action=action,
                instance_name=str(bot_configuration_id),
                profile_name=str(runtime["profile_name"]),
                env_overrides=dict(runtime["runtime_env"] or {}),
            )
        except BotRunnerError as exc:
            await db.botconfiguration.update(
                where={"id": bot_configuration_id},
                data={
                    "containerStatus": "error",
                    "isActive": False,
                },
            )
            await _emit_admin_lifecycle_event(
                bot_config_id=str(bot_configuration_id),
                action=action,
                phase="failed",
                detail=str(exc),
                status="error",
                metadata={"admin_user_id": str(getattr(current_user, "id", "") or "").strip() or None},
                owner_user_id=str(user_id),
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to start bot docker instance (admin action)", exc),
            ) from exc

        update_payload = {
            "containerStatus": "running",
            "isActive": True,
            "dockerContainerId": runner_result.container_id,
        }
        current_version_tag = str(getattr(bot_configuration.botVersion, "versionTag", "") or "").strip()
        if current_version_tag:
            update_payload["installedVersionTag"] = current_version_tag
        if runtime.get("docker_image_id"):
            update_payload["installedDockerImageId"] = runtime["docker_image_id"]

        await db.botconfiguration.update(
            where={"id": bot_configuration_id},
            data=update_payload,
        )
        await _emit_admin_lifecycle_event(
            bot_config_id=str(bot_configuration_id),
            action=action,
            phase="succeeded",
            detail="Bot runtime is running",
            status="running",
            metadata={
                "admin_user_id": str(getattr(current_user, "id", "") or "").strip() or None,
                "docker_project_name": getattr(runner_result, "project_name", None),
                "docker_container_id": getattr(runner_result, "container_id", None),
            },
            owner_user_id=str(user_id),
        )
    else:
        await _emit_admin_lifecycle_event(
            bot_config_id=str(bot_configuration_id),
            action="stop",
            phase="requested",
            detail="Bot runtime stop requested by admin",
            status="stopped",
            metadata={"admin_user_id": str(getattr(current_user, "id", "") or "").strip() or None},
            owner_user_id=str(user_id),
        )
        try:
            runner_result = await asyncio.to_thread(
                run_bot_instance_action,
                action="stop",
                instance_name=str(bot_configuration_id),
                timeout_sec=300,
            )
        except BotRunnerError as exc:
            await _emit_admin_lifecycle_event(
                bot_config_id=str(bot_configuration_id),
                action="stop",
                phase="failed",
                detail=str(exc),
                status="error",
                metadata={"admin_user_id": str(getattr(current_user, "id", "") or "").strip() or None},
                owner_user_id=str(user_id),
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to stop bot docker instance (admin action)", exc),
            ) from exc

        await db.botconfiguration.update(
            where={"id": bot_configuration_id},
            data={
                "containerStatus": "stopped",
                "isActive": False,
                "dockerContainerId": None,
            },
        )
        await _emit_admin_lifecycle_event(
            bot_config_id=str(bot_configuration_id),
            action="stop",
            phase="succeeded",
            detail="Bot runtime stopped",
            status="stopped",
            metadata={
                "admin_user_id": str(getattr(current_user, "id", "") or "").strip() or None,
                "docker_project_name": getattr(runner_result, "project_name", None),
                "docker_container_id": getattr(runner_result, "container_id", None),
            },
            owner_user_id=str(user_id),
        )

    return {
        "message": f"Bot status updated to {status_value}",
        "docker_project_name": getattr(runner_result, "project_name", None),
        "docker_container_id": getattr(runner_result, "container_id", None),
    }


@admin_router.get("/bot-versions", response_model=List[AdminBotVersionItemResponse])
async def get_admin_bot_versions(
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    versions = await db.botversion.find_many(order={"releaseDate": "desc"})
    results: List[AdminBotVersionItemResponse] = []

    for version in versions:
        usage_count = await db.botconfiguration.count(where={"modelId": str(version.modelId)})
        results.append(_map_bot_version(version, usage_count=usage_count))

    return results


@admin_router.post(
    "/bot-versions",
    response_model=AdminBotVersionItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_admin_bot_version(
    data: CreateAdminBotVersionRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    if not data.label.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="label is required")
    if not data.version_tag.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="version_tag is required")

    created = await db.botversion.create(
        data={
            "label": data.label.strip(),
            "versionTag": data.version_tag.strip(),
            "symbol": data.symbol.strip() if data.symbol else None,
            "timeframe": data.timeframe.strip() if data.timeframe else None,
            "isActive": bool(data.is_active),
            "releaseNotes": _clean_release_notes(data.release_notes),
        }
    )

    return _map_bot_version(created, usage_count=0)


@admin_router.patch("/bot-versions/{model_id}", response_model=AdminBotVersionItemResponse)
async def update_admin_bot_version(
    model_id: str,
    data: UpdateAdminBotVersionRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    existing = await db.botversion.find_unique(where={"modelId": model_id})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot version not found")

    update_payload = {}

    if data.label is not None:
        label = data.label.strip()
        if not label:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="label cannot be empty")
        update_payload["label"] = label

    if data.version_tag is not None:
        version_tag = data.version_tag.strip()
        if not version_tag:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="version_tag cannot be empty")
        if existing.versionTag and version_tag != existing.versionTag:
            await db.botconfiguration.update_many(
                where={
                    "modelId": model_id,
                    "installedVersionTag": None,
                },
                data={"installedVersionTag": existing.versionTag},
            )
        update_payload["versionTag"] = version_tag

    if data.symbol is not None:
        update_payload["symbol"] = data.symbol.strip() if data.symbol else None

    if data.timeframe is not None:
        update_payload["timeframe"] = data.timeframe.strip() if data.timeframe else None

    if data.release_notes is not None:
        update_payload["releaseNotes"] = _clean_release_notes(data.release_notes)

    if data.is_active is not None:
        next_active = bool(data.is_active)
        update_payload["isActive"] = next_active
        if not next_active:
            await _stop_active_bots_using_version(model_id)

    if update_payload:
        updated = await db.botversion.update(
            where={"modelId": model_id},
            data=update_payload,
        )
    else:
        updated = existing

    usage_count = await db.botconfiguration.count(where={"modelId": model_id})
    return _map_bot_version(updated, usage_count=usage_count)


@admin_router.post("/bot-versions/{model_id}/publish-update")
async def publish_admin_bot_update(
    model_id: str,
    data: PublishAdminBotUpdateRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    existing = await db.botversion.find_unique(where={"modelId": model_id})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot version not found")

    release_notes = _clean_release_notes(data.release_notes) if data.release_notes is not None else None
    next_version_tag = data.version_tag.strip() if data.version_tag is not None else None
    if next_version_tag is not None and not next_version_tag:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="version_tag cannot be empty",
        )

    has_version_tag_change = bool(
        next_version_tag is not None and next_version_tag != (existing.versionTag or "")
    )
    has_release_notes_change = (
        release_notes is not None and release_notes != (existing.releaseNotes or [])
    )

    if not has_version_tag_change and not has_release_notes_change:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No update fields changed. Provide version_tag or release_notes.",
        )

    if has_version_tag_change and existing.versionTag:
        await db.botconfiguration.update_many(
            where={
                "modelId": model_id,
                "installedVersionTag": None,
            },
            data={"installedVersionTag": existing.versionTag},
        )

    update_payload = {
        "releaseDate": datetime.utcnow(),
    }

    if next_version_tag is not None:
        update_payload["versionTag"] = next_version_tag

    if data.release_notes is not None:
        update_payload["releaseNotes"] = release_notes or []

    updated_version = await db.botversion.update(
        where={"modelId": model_id},
        data=update_payload,
    )

    affected_bots = await db.botconfiguration.count(where={"modelId": model_id})
    users_notified = 0
    emails_sent = 0
    if bool(data.notify_users):
        users_notified, emails_sent = await _notify_users_for_bot_update(
            model_id=model_id,
            version=updated_version,
            release_notes=updated_version.releaseNotes or [],
        )

    return {
        "message": "Bot update published",
        "model_id": model_id,
        "version_tag": updated_version.versionTag,
        "affected_bots": affected_bots,
        "users_notified": users_notified,
        "emails_sent": emails_sent,
    }


@admin_router.patch("/bot-versions/{model_id}/active")
async def update_admin_bot_version_active(
    model_id: str,
    data: UpdateAdminBotVersionActiveRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    existing = await db.botversion.find_unique(where={"modelId": model_id})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot version not found")

    next_active = bool(data.is_active)
    stopped_bots_count = 0
    if not next_active:
        stopped_bots_count = await _stop_active_bots_using_version(model_id)

    await db.botversion.update(
        where={"modelId": model_id},
        data={"isActive": next_active},
    )

    return {
        "message": f"Bot version {'activated' if next_active else 'deactivated'}",
        "is_active": next_active,
        "stopped_bots": stopped_bots_count,
    }


@admin_router.post("/bot-versions/{model_id}/rollout")
async def rollout_admin_bot_version(
    model_id: str,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    target_version = await db.botversion.find_unique(where={"modelId": model_id})
    if not target_version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot version not found")

    if not bool(getattr(target_version, "isActive", True)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot rollout an inactive bot version",
        )

    source_where = {"modelId": {"not": model_id}}
    if target_version.symbol:
        source_where["symbol"] = target_version.symbol
    if target_version.timeframe:
        source_where["timeframe"] = target_version.timeframe
    if not target_version.symbol and not target_version.timeframe and target_version.label:
        source_where["label"] = target_version.label
    if not target_version.symbol and not target_version.timeframe and not target_version.label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target version needs symbol/timeframe or label to rollout safely",
        )

    source_versions = await db.botversion.find_many(where=source_where)
    source_model_ids = [str(version.modelId) for version in source_versions]

    if not source_model_ids:
        return {
            "message": "No older bot versions found for rollout",
            "updated_bots": 0,
            "source_versions": 0,
        }

    affected_bots = await db.botconfiguration.count(
        where={"modelId": {"in": source_model_ids}}
    )

    if affected_bots > 0:
        await db.botconfiguration.update_many(
            where={"modelId": {"in": source_model_ids}},
            data={
                "modelId": model_id,
                "installedDockerImageId": target_version.dockerImageId,
                "installedVersionTag": target_version.versionTag,
            },
        )

    return {
        "message": "Bot rollout completed",
        "updated_bots": affected_bots,
        "source_versions": len(source_model_ids),
        "target_version": model_id,
    }


@admin_router.delete("/bot-versions/{model_id}")
async def delete_admin_bot_version(
    model_id: str,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    existing = await db.botversion.find_unique(where={"modelId": model_id})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot version not found")

    usage_count = await db.botconfiguration.count(where={"modelId": model_id})
    if usage_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a bot version that is currently in use",
        )

    await db.botversion.delete(where={"modelId": model_id})
    return {"message": "Bot version deleted"}
