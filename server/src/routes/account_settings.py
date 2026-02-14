from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, List, Optional
import bcrypt

from ..database.client import db
from ..models.settings import (
    UserProfileResponse, 
    NotificationConfigResponse, 
    UpdateProfileRequest, 
    UpdatePasswordRequest, 
    UpdateNotificationsRequest, 
    ActivityLogResponse
)
from .authentication import get_current_active_user

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
        notificationConfig=notify_response
    )

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
    if not current_user.password:
        raise HTTPException(status_code=400, detail="Please set a password first via forgot password or login flow")

    if not bcrypt.checkpw(data.currentPassword.encode('utf-8'), current_user.password.encode('utf-8')):
         raise HTTPException(status_code=400, detail="Current password incorrect")
         
    hashed = bcrypt.hashpw(data.newPassword.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    await db.user.update(
        where={"id": str(current_user.id)},
        data={"password": hashed}
    )
    
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
