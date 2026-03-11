from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CreateSetupIntentResponse(BaseModel):
    client_secret: str


class CreateCheckoutSessionResponse(BaseModel):
    session_id: str
    url: Optional[str] = None


class CreateCheckoutSessionRequest(BaseModel):
    payment_flow: str = "card"


class AttachPaymentMethodRequest(BaseModel):
    paymentMethodId: str
    setAsDefault: bool = False


class PayInvoiceRequest(BaseModel):
    payment_method_id: Optional[str] = None


class PaymentMethodResponse(BaseModel):
    id: str
    type: Optional[str] = None
    card_last4: Optional[str] = None
    card_brand: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    is_default: bool = False


class InvoiceResponse(BaseModel):
    id: str
    billing_start_date: Optional[str] = None
    billing_end_date: Optional[str] = None
    total_period_profit: float = 0.0
    calculated_fee: float = 0.0
    payment_amount: Optional[float] = None
    payment_currency: Optional[str] = None
    status: Optional[str] = None
    payment_method_used: Optional[str] = None
    payment_method_label: Optional[str] = None
    paid_at: Optional[str] = None
    created_at: Optional[str] = None


class SubscriptionResponse(BaseModel):
    id: str
    status: str
    collection_mode: str = "automatic"
    fee_type: str
    fee_value: float
    min_profit_threshold: float
    next_billing_date: Optional[str] = None
    default_payment_method_id: Optional[str] = None
    billing_currency: str = "USD"
    billing_exchange_rate: float = 1.0
    promptpay_enabled: bool = False
    promptpay_currency: Optional[str] = None
    promptpay_exchange_rate: Optional[float] = None


class WeeklyPreviewResponse(BaseModel):
    week_start: str
    week_end: str
    gross_profit: float
    gross_loss: float
    net_profit: float
    estimated_fee: float
    estimated_fee_payment: float
    estimated_fee_payment_currency: str = "USD"


class SubscriptionSummaryResponse(BaseModel):
    subscription: SubscriptionResponse
    invoices: List[InvoiceResponse]
    payment_methods: List[PaymentMethodResponse]
    weekly_preview: WeeklyPreviewResponse


class AdminBillingConfigResponse(BaseModel):
    config_id: Optional[int] = None
    default_fee_type: str = "percentage"
    default_collection_mode: str = "automatic"
    default_fee_value: float = 20.0
    default_min_threshold: float = 0.0
    default_next_billing_date: Optional[str] = None
    updated_at: Optional[str] = None


class AdminSubscriptionItemResponse(BaseModel):
    id: str
    user_id: str
    user_email: Optional[str] = None
    status: str
    collection_mode: str = "automatic"
    fee_type: str
    fee_value: float
    min_profit_threshold: float
    next_billing_date: Optional[str] = None
    created_at: Optional[str] = None


class AdminSubscriptionManagementResponse(BaseModel):
    billing_config: AdminBillingConfigResponse
    subscriptions: List[AdminSubscriptionItemResponse]


class AdminInvoiceDetailResponse(BaseModel):
    id: str
    billing_start_date: Optional[str] = None
    billing_end_date: Optional[str] = None
    total_period_profit: float = 0.0
    calculated_fee: float = 0.0
    status: Optional[str] = None
    payment_method_used: Optional[str] = None
    stripe_payment_intent_id: Optional[str] = None
    stripe_charge_id: Optional[str] = None
    stripe_balance_txn_id: Optional[str] = None
    processor_request_id: Optional[str] = None
    payment_breakdown: Optional[Dict[str, Any]] = None
    payment_method_details: Optional[Dict[str, Any]] = None
    payment_error_details: Optional[Dict[str, Any]] = None
    paid_at: Optional[str] = None
    created_at: Optional[str] = None


class AdminSubscriptionInvoiceListResponse(BaseModel):
    subscription_id: str
    user_id: str
    user_email: Optional[str] = None
    invoices: List[AdminInvoiceDetailResponse]


class UpdateBillingConfigRequest(BaseModel):
    default_fee_type: str
    default_collection_mode: str = "automatic"
    default_fee_value: float
    default_min_threshold: float
    default_next_billing_date: Optional[str] = None


class UpdateCollectionModeRequest(BaseModel):
    collection_mode: str = "automatic"


class UpdateSubscriptionStatusRequest(BaseModel):
    status: str


class AdminSubscriptionStatsResponse(BaseModel):
    total_users: int
    active_subscriptions: int
    total_bot_versions: int
    pending_tickets: int
    monthly_revenue: float


class ProcessDueBillingResponse(BaseModel):
    processed_subscriptions: int = 0
    created_invoices: int = 0
    paid_invoices: int = 0
    pending_invoices: int = 0
    skipped_invoices: int = 0
    failed_invoices: int = 0
