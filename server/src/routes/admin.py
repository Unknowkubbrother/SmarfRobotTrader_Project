from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..database.client import db
from ..models.admin import (
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
    UpdateAdminBotConfigurationStatusRequest,
    UpdateAdminUserSubscriptionBillingRequest,
    UpdateAdminUserRoleRequest,
    UpdateAdminUserStatusRequest,
)
from .authentication import get_current_active_user

admin_router = APIRouter(tags=["Admin"])

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

            if container_status == "running":
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
    mapped_subscriptions = [_map_user_subscription(subscription) for subscription in subscriptions]
    subscription_ids = [str(sub.id) for sub in subscriptions]

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

    updated_subscription = await db.subscription.update(
        where={"id": subscription_id},
        data={
            "feeType": fee_type,
            "feeValue": Decimal(str(data.fee_value)),
            "minProfitThreshold": Decimal(str(data.min_profit_threshold)),
        },
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
        include={"account": True},
    )
    if not bot_configuration or str(bot_configuration.account.userId) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot configuration not found for this user",
        )

    await db.botconfiguration.update(
        where={"id": bot_configuration_id},
        data={
            "containerStatus": status_value,
            "isActive": status_value == "running",
        },
    )

    return {"message": f"Bot status updated to {status_value}"}


@admin_router.get("/bot-versions", response_model=List[AdminBotVersionItemResponse])
async def get_admin_bot_versions(
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    versions = await db.botversion.find_many(order={"releaseDate": "desc"})
    results: List[AdminBotVersionItemResponse] = []

    for version in versions:
        usage_count = await db.botconfiguration.count(where={"modelId": str(version.modelId)})
        results.append(
            AdminBotVersionItemResponse(
                id=str(version.modelId),
                label=version.label,
                version_tag=version.versionTag,
                symbol=version.symbol,
                timeframe=version.timeframe,
                docker_image_id=version.dockerImageId,
                release_notes=version.releaseNotes or [],
                release_date=_to_datetime_string(version.releaseDate),
                usage_count=usage_count,
            )
        )

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
            "dockerImageId": data.docker_image_id.strip() if data.docker_image_id else None,
            "releaseNotes": data.release_notes,
        }
    )

    return AdminBotVersionItemResponse(
        id=str(created.modelId),
        label=created.label,
        version_tag=created.versionTag,
        symbol=created.symbol,
        timeframe=created.timeframe,
        docker_image_id=created.dockerImageId,
        release_notes=created.releaseNotes or [],
        release_date=_to_datetime_string(created.releaseDate),
        usage_count=0,
    )


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
