from typing import List, Optional

from pydantic import BaseModel, Field


class AdminStatsResponse(BaseModel):
    total_users: int
    total_mt5_accounts: int
    total_bot_versions: int
    active_subscriptions: int
    pending_tickets: int
    monthly_revenue: float
    running_bots: int


class AdminUserItemResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    status: str
    created_at: str
    is_onboarding_completed: bool


class UpdateAdminUserStatusRequest(BaseModel):
    status: str


class UpdateAdminUserRoleRequest(BaseModel):
    role: str


class AdminBotVersionItemResponse(BaseModel):
    id: str
    label: Optional[str] = None
    version_tag: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    docker_image_id: Optional[str] = None
    release_notes: List[str] = Field(default_factory=list)
    release_date: Optional[str] = None
    usage_count: int = 0


class CreateAdminBotVersionRequest(BaseModel):
    label: str
    version_tag: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    docker_image_id: Optional[str] = None
    release_notes: List[str] = Field(default_factory=list)

