from fastapi import APIRouter, Request, HTTPException
from prisma.models import Notification
from ..database.client import db
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class NotificationResponse(BaseModel):
    id: str
    title: Optional[str]
    message: Optional[str]
    isRead: bool
    relatedLink: Optional[str]
    createdAt: datetime

notification_router = APIRouter(tags=["Notifications"])

@notification_router.get("/", response_model=List[NotificationResponse])
async def get_notifications(request: Request):
    user_id = request.state.user_id
    
    notifications = await db.notification.find_many(
        where={
            "userId": user_id
        },
        order={
            "createdAt": "desc"
        },
        take=20
    )
    
    return notifications

@notification_router.patch("/{notification_id}/read")
async def mark_as_read(notification_id: str, request: Request):
    user_id = request.state.user_id
    
    notification = await db.notification.find_first(
        where={
            "id": notification_id,
            "userId": user_id
        }
    )
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
        
    updated = await db.notification.update(
        where={
            "id": notification_id
        },
        data={
            "isRead": True
        }
    )
    
    return updated

@notification_router.patch("/read-all")
async def mark_all_as_read(request: Request):
    user_id = request.state.user_id
    
    await db.notification.update_many(
        where={
            "userId": user_id,
            "isRead": False
        },
        data={
            "isRead": True
        }
    )
    
    return {"status": "success"}
