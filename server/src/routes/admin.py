from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..database.client import db
from ..models.admin import (
    AdminBotVersionItemResponse,
    AdminStatsResponse,
    AdminUserItemResponse,
    CreateAdminBotVersionRequest,
    UpdateAdminUserRoleRequest,
    UpdateAdminUserStatusRequest,
)
from .authentication import get_current_active_user

admin_router = APIRouter(tags=["Admin"])

ALLOWED_USER_STATUSES = {"active", "banned", "pending"}
ALLOWED_USER_ROLES = {"user", "admin"}


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


def _require_admin(current_user):
    role = _enum_value(getattr(current_user, "role", None))
    if role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")


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


@admin_router.get("/bot-versions", response_model=List[AdminBotVersionItemResponse])
async def get_admin_bot_versions(
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    _require_admin(current_user)

    versions = await db.botversion.find_many(order={"releaseDate": "desc"})
    results: List[AdminBotVersionItemResponse] = []

    for version in versions:
        usage_count = await db.botconfiguration.count(
            where={"modelId": str(version.modelId)}
        )
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
