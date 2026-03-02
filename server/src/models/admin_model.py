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


class UpdateAdminBotConfigurationStatusRequest(BaseModel):
    status: str


class UpdateAdminUserSubscriptionBillingRequest(BaseModel):
    fee_type: str
    fee_value: float
    min_profit_threshold: float
    next_billing_date: Optional[str] = None


class UpdateAdminBotVersionActiveRequest(BaseModel):
    is_active: bool


class UpdateAdminBotVersionRequest(BaseModel):
    label: Optional[str] = None
    version_tag: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    docker_image_id: Optional[str] = None
    release_notes: Optional[List[str]] = None
    is_active: Optional[bool] = None


class AdminUserBotConfigurationItemResponse(BaseModel):
    id: str
    bot_instance_id: int
    model_id: str
    label: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    container_status: Optional[str] = None
    is_active: bool = False
    updated_at: Optional[str] = None


class AdminUserTradingAccountItemResponse(BaseModel):
    id: str
    mt5_login_id: Optional[str] = None
    broker_name: Optional[str] = None
    server_name: Optional[str] = None
    balance: float = 0.0
    equity: float = 0.0
    running_bots: int = 0
    active_bots: int = 0
    bots: List[AdminUserBotConfigurationItemResponse] = Field(default_factory=list)


class AdminUserSubscriptionItemResponse(BaseModel):
    id: str
    status: str
    fee_type: str
    fee_value: float = 0.0
    min_profit_threshold: float = 0.0
    next_billing_date: Optional[str] = None
    created_at: Optional[str] = None


class AdminUserInvoiceItemResponse(BaseModel):
    id: str
    subscription_id: str
    status: Optional[str] = None
    amount: float = 0.0
    created_at: Optional[str] = None
    paid_at: Optional[str] = None
    billing_start_date: Optional[str] = None
    billing_end_date: Optional[str] = None


class AdminUserBillingSummaryResponse(BaseModel):
    pending_count: int = 0
    paid_count: int = 0
    pending_amount: float = 0.0
    paid_amount: float = 0.0
    recent_invoices: List[AdminUserInvoiceItemResponse] = Field(default_factory=list)


class AdminUserDetailResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    status: str
    created_at: str
    is_onboarding_completed: bool
    total_accounts: int = 0
    total_balance: float = 0.0
    pending_bills: int = 0
    trading_accounts: List[AdminUserTradingAccountItemResponse] = Field(default_factory=list)
    subscriptions: List[AdminUserSubscriptionItemResponse] = Field(default_factory=list)
    billing: AdminUserBillingSummaryResponse


class AdminBotVersionItemResponse(BaseModel):
    id: str
    label: Optional[str] = None
    version_tag: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    docker_image_id: Optional[str] = None
    is_active: bool = True
    release_notes: List[str] = Field(default_factory=list)
    release_date: Optional[str] = None
    usage_count: int = 0


class CreateAdminBotVersionRequest(BaseModel):
    label: str
    version_tag: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    docker_image_id: Optional[str] = None
    is_active: bool = True
    release_notes: List[str] = Field(default_factory=list)


class PublishAdminBotUpdateRequest(BaseModel):
    docker_image_id: str
    version_tag: Optional[str] = None
    release_notes: List[str] = Field(default_factory=list)
    notify_users: bool = True
