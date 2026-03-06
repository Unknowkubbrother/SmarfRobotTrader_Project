from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class ActivityLogResponse(BaseModel):
    id: str
    date: datetime
    ip: Optional[str] = None
    device: Optional[str] = None
    location: Optional[str] = None  # We might not have this in DB, maybe infer or leave null
    topic: Optional[str] = None

class NotificationConfigResponse(BaseModel):
    emailNotificationEnable: bool = True
    alertMarginLevelThreshold: Optional[Decimal] = None
    alertProfitTarget: Optional[Decimal] = None
    alertLossLimit: Optional[Decimal] = None
    enableWeeklySummary: bool = True
    enableMonthlySummary: bool = True
    discordWebhookUrl: Optional[str] = None
    discordWebhookDisplay: Optional[str] = None
    hasDiscordWebhook: bool = False

class UserProfileResponse(BaseModel):
    id: str
    username: str
    email: str
    recoveryEmail: Optional[str] = None
    avatarUrl: Optional[str] = None
    # Settings that are part of profile
    notificationConfig: Optional[NotificationConfigResponse] = None
    hasPassword: bool = False

class UpdateProfileRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=2, max_length=50)
    email: Optional[EmailStr] = None
    recoveryEmail: Optional[EmailStr] = None
    avatarUrl: Optional[str] = None

class UpdatePasswordRequest(BaseModel):
    otp: str
    newPassword: str = Field(..., min_length=6)

class UpdateNotificationsRequest(BaseModel):
    emailNotificationEnable: Optional[bool] = None
    alertMarginLevelThreshold: Optional[Decimal] = None
    alertProfitTarget: Optional[Decimal] = None
    alertLossLimit: Optional[Decimal] = None
    enableWeeklySummary: Optional[bool] = None
    enableMonthlySummary: Optional[bool] = None
    discordWebhookUrl: Optional[str] = None
