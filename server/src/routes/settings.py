from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import Annotated, List, Optional
import bcrypt
import os
import shutil
import time

from ..database.client import db, r_cache
from lib.untils import random_with_N_digits, send_otp_email
from ..models.settings import (
    UserProfileResponse, 
    NotificationConfigResponse, 
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

@settings_router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    # Fetch notification config
    notify_config = await db.notificationconfig.find_unique(
        where={"userId": str(current_user.id)}
    )
    
    notify_response = None
    if notify_config:
        notify_response = NotificationConfigResponse(
            emailNotificationEnable=notify_config.emailNotificationEnable,
            alertMarginLevelThreshold=notify_config.alertMarginLevelThreshold,
            alertProfitTarget=notify_config.alertProfitTarget,
            alertLossLimit=notify_config.alertLossLimit,
            enableWeeklySummary=notify_config.enableWeeklySummary,
            enableMonthlySummary=notify_config.enableMonthlySummary,
            lineNotifyToken=notify_config.lineNotifyToken,
            discordWebhookUrl=notify_config.discordWebhookUrl
        )
    else:
        # Return defaults if no config exists
        notify_response = NotificationConfigResponse()

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
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    if not current_user.recoveryEmail:
        raise HTTPException(status_code=400, detail="No recovery email set")

    otp = str(random_with_N_digits(6))
    r_cache.setex(f"security_otp:{current_user.id}", 300, otp)
    
    send_otp_email(current_user.recoveryEmail, otp, purpose="password_change")

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
    data_to_update = {}
    if update_data.username:
        # Check if username exists
        existing = await db.user.find_first(
            where={
                "username": update_data.username,
                "NOT": {"id": str(current_user.id)}
            }
        )
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
        data_to_update["username"] = update_data.username
        
    if update_data.email:
        # Check if email exists
        existing = await db.user.find_first(
            where={
                "email": update_data.email,
                "NOT": {"id": str(current_user.id)}
            }
        )
        if existing:
            raise HTTPException(status_code=400, detail="Email already taken")
        data_to_update["email"] = update_data.email
        
    if update_data.recoveryEmail:
        data_to_update["recoveryEmail"] = update_data.recoveryEmail

    if update_data.avatarUrl is not None:
        data_to_update["avatarUrl"] = update_data.avatarUrl

    if not data_to_update:
         return await get_profile(current_user)

    updated_user = await db.user.update(
        where={"id": str(current_user.id)},
        data=data_to_update
    )
    
    # Re-fetch profile to include config
    return await get_profile(updated_user)

@settings_router.patch("/password")
async def update_password(
    data: UpdatePasswordRequest,
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    # Verify OTP
    stored_otp = r_cache.get(f"security_otp:{current_user.id}")
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
    r_cache.delete(f"security_otp:{current_user.id}")
    
    return {"message": "Password updated successfully"}

@settings_router.patch("/notifications")
async def update_notifications(
    data: UpdateNotificationsRequest,
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    # Prepare update data types, filtering None
    update_dict = data.dict(exclude_unset=True)
    
    if not update_dict:
        return {"message": "No changes provided"}

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
        
    return {"message": "Notification settings updated"}

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
