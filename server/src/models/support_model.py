from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CreateSupportTicketRequest(BaseModel):
    subject: str
    category: Optional[str] = None
    message: str


class AdminReplySupportTicketRequest(BaseModel):
    reply: str
    status: str = "resolved"


class UserReplySupportTicketRequest(BaseModel):
    message: str


class SupportTicketMessageItem(BaseModel):
    role: str
    text: str
    created_at: Optional[str] = None
    sender_name: Optional[str] = None
    sender_email: Optional[str] = None


class SupportTicketItemResponse(BaseModel):
    id: str
    user_id: str
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    subject: str
    category: Optional[str] = None
    user_message: str
    admin_reply: Optional[str] = None
    messages: list[SupportTicketMessageItem] = Field(default_factory=list)
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
