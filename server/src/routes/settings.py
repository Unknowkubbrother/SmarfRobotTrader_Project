from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import Annotated, List, Optional
import bcrypt
import os
import shutil
import time
from urllib.parse import urlparse

from ..database.client import db, r_cache
from lib.untils import random_with_N_digits, send_otp_email
from ..models.settings_model import (
    UserProfileResponse, 
    NotificationConfigResponse, 
    SecurityOtpRequest,
    UpdateProfileRequest, 
    UpdatePasswordRequest, 
    UpdateNotificationsRequest, 
    ActivityLogResponse
)
from .authentication import get_current_active_user

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads", "avatars")
os.makedirs(UPLOADS_DIR, exist_ok=True)

settings_router = APIRouter()


def _security_otp_cache_key(user_id: str, purpose: str) -> str:
    return f"security_otp:{user_id}:{purpose}"


def _mask_discord_webhook_url(value: Optional[str]) -> Optional[str]:
    normalized = str(value or "").strip()
    if not normalized:
        return None

    parsed = urlparse(normalized)
    host = parsed.netloc or "discord"
    suffix = normalized[-4:] if len(normalized) >= 4 else normalized
    return f"{host} ••••{suffix}"


def _build_notification_config_response(notify_config) -> NotificationConfigResponse:
    if not notify_config:
        return NotificationConfigResponse()

    webhook_url = str(getattr(notify_config, "discordWebhookUrl", None) or "").strip()
    return NotificationConfigResponse(
        emailNotificationEnable=notify_config.emailNotificationEnable,
        alertMarginLevelThreshold=notify_config.alertMarginLevelThreshold,
        alertProfitTarget=notify_config.alertProfitTarget,
        alertLossLimit=notify_config.alertLossLimit,
        enableWeeklySummary=notify_config.enableWeeklySummary,
        enableMonthlySummary=notify_config.enableMonthlySummary,
        discordWebhookUrl=None,
        discordWebhookDisplay=_mask_discord_webhook_url(webhook_url),
        hasDiscordWebhook=bool(webhook_url),
    )

@settings_router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    # Fetch notification config
    notify_config = await db.notificationconfig.find_unique(
        where={"userId": str(current_user.id)}
    )
    
    notify_response = _build_notification_config_response(notify_config)

    return UserProfileResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        recoveryEmail=current_user.recoveryEmail,
        avatarUrl=current_user.avatarUrl,
        notificationConfig=notify_response,
        hasPassword=bool(current_user.password)
    )

@settings_router.post("/security/otp")
async def request_security_otp(
    current_user: Annotated[any, Depends(get_current_active_user)],
    data: Optional[SecurityOtpRequest] = None,
):
    if not current_user.recoveryEmail:
        raise HTTPException(status_code=400, detail="No recovery email set")

    purpose = data.purpose if data else "password_change"
    otp = str(random_with_N_digits(6))
    r_cache.setex(_security_otp_cache_key(str(current_user.id), purpose), 300, otp)
    
    send_otp_email(current_user.recoveryEmail, otp, purpose=purpose)

    return {
        "message": "OTP sent to recovery email",
        "recovery_email_hint": f"{current_user.recoveryEmail[:4]}****@{current_user.recoveryEmail.split('@')[1]}"
    }

@settings_router.post("/profile/avatar")
async def upload_avatar(
    file: Annotated[UploadFile, File()],
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    extension = os.path.splitext(file.filename)[1]
    filename = f"user_{current_user.id}_{int(time.time())}{extension}"
    file_path = os.path.join(UPLOADS_DIR, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save image")
        
    # Construct URL (assuming server is at root/static)
    avatar_url = f"/static/avatars/{filename}"
    
    # Delete old avatar if exists and is local
    if current_user.avatarUrl and current_user.avatarUrl.startswith("/static/avatars/"):
        old_filename = current_user.avatarUrl.split("/")[-1]
        old_path = os.path.join(UPLOADS_DIR, old_filename)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Update user record
    await db.user.update(
        where={"id": str(current_user.id)},
        data={"avatarUrl": avatar_url}
    )
    
    return {"avatarUrl": avatar_url}

@settings_router.patch("/profile", response_model=UserProfileResponse)
async def update_profile(
    update_data: UpdateProfileRequest,
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    provided_fields = set(getattr(update_data, "model_fields_set", getattr(update_data, "__fields_set__", set())))
    data_to_update = {}
    sensitive_change_requested = False

    if "username" in provided_fields:
        username = update_data.username.strip() if update_data.username else None
        if username != current_user.username:
            if not username:
                raise HTTPException(status_code=400, detail="Username is required")

        # Check if username exists
            existing = await db.user.find_first(
                where={
                    "username": username,
                    "NOT": {"id": str(current_user.id)}
                }
            )
            if existing:
                raise HTTPException(status_code=400, detail="Username already taken")

            data_to_update["username"] = username
            sensitive_change_requested = True

    if "email" in provided_fields:
        email = str(update_data.email).strip() if update_data.email is not None else None
        if email != current_user.email:
        # Check if email exists
            if not email:
                raise HTTPException(status_code=400, detail="Email is required")

            existing = await db.user.find_first(
                where={
                    "email": email,
                    "NOT": {"id": str(current_user.id)}
                }
            )
            if existing:
                raise HTTPException(status_code=400, detail="Email already taken")

            data_to_update["email"] = email
            sensitive_change_requested = True

    if "recoveryEmail" in provided_fields:
        recovery_email = str(update_data.recoveryEmail).strip() if update_data.recoveryEmail is not None else None
        if recovery_email != current_user.recoveryEmail:
            data_to_update["recoveryEmail"] = recovery_email
            sensitive_change_requested = True

    if "avatarUrl" in provided_fields:
        avatar_url = update_data.avatarUrl.strip() if isinstance(update_data.avatarUrl, str) else update_data.avatarUrl
        if avatar_url != current_user.avatarUrl:
            data_to_update["avatarUrl"] = avatar_url

    if not data_to_update:
         return await get_profile(current_user)

    if sensitive_change_requested:
        if not update_data.otp:
            raise HTTPException(status_code=400, detail="OTP is required to update account information")

        stored_otp = r_cache.get(_security_otp_cache_key(str(current_user.id), "profile_change"))
        if not stored_otp:
            raise HTTPException(status_code=400, detail="OTP expired or not requested")

        if stored_otp != update_data.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")

    updated_user = await db.user.update(
        where={"id": str(current_user.id)},
        data=data_to_update
    )

    if sensitive_change_requested:
        r_cache.delete(_security_otp_cache_key(str(current_user.id), "profile_change"))
    
    # Re-fetch profile to include config
    return await get_profile(updated_user)

@settings_router.patch("/password")
async def update_password(
    data: UpdatePasswordRequest,
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    # Verify OTP
    stored_otp = r_cache.get(_security_otp_cache_key(str(current_user.id), "password_change"))
    if not stored_otp:
        raise HTTPException(status_code=400, detail="OTP expired or not requested")
    
    if stored_otp != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed = bcrypt.hashpw(data.newPassword.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    await db.user.update(
        where={"id": str(current_user.id)},
        data={"password": hashed}
    )
    
    # Clear OTP
    r_cache.delete(_security_otp_cache_key(str(current_user.id), "password_change"))
    
    return {"message": "Password updated successfully"}

@settings_router.patch("/notifications", response_model=NotificationConfigResponse)
async def update_notifications(
    data: UpdateNotificationsRequest,
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    # Prepare update data types, filtering None
    update_dict = data.dict(exclude_unset=True)

    if not update_dict:
        return (await get_profile(current_user)).notificationConfig

    if "alertProfitTarget" in update_dict and update_dict["alertProfitTarget"] is not None:
        update_dict["alertProfitTarget"] = abs(update_dict["alertProfitTarget"])

    if "alertLossLimit" in update_dict and update_dict["alertLossLimit"] is not None:
        update_dict["alertLossLimit"] = -abs(update_dict["alertLossLimit"])

    if "alertMarginLevelThreshold" in update_dict and update_dict["alertMarginLevelThreshold"] is not None:
        update_dict["alertMarginLevelThreshold"] = abs(update_dict["alertMarginLevelThreshold"])

    if "discordWebhookUrl" in update_dict and update_dict["discordWebhookUrl"] is not None:
        normalized_url = str(update_dict["discordWebhookUrl"]).strip()
        update_dict["discordWebhookUrl"] = normalized_url or None

    # Check if config exists
    config = await db.notificationconfig.find_unique(where={"userId": str(current_user.id)})
    
    if config:
        await db.notificationconfig.update(
            where={"userId": str(current_user.id)},
            data=update_dict
        )
    else:
        # Create new config with defaults + updates
        create_data = {
            "userId": str(current_user.id),
            **update_dict
        }
        await db.notificationconfig.create(data=create_data)

    refreshed_config = await db.notificationconfig.find_unique(where={"userId": str(current_user.id)})
    return _build_notification_config_response(refreshed_config)

@settings_router.get("/activity-logs", response_model=List[ActivityLogResponse])
async def get_activity_logs(
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    logs = await db.activitylog.find_many(
        where={"userId": str(current_user.id)},
        order={"createdAt": "desc"},
        take=10
    )
    
    return [
        ActivityLogResponse(
            id=str(log.id),
            date=log.createdAt,
            ip=log.ipAddress,
            device=log.deviceInfo,
            topic=log.topic
        ) for log in logs
    ]
