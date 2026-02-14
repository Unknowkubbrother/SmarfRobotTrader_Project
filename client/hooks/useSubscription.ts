import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";

import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";

export interface SubscriptionData {
  id: string;
  status: string;
  fee_type: string;
  fee_value: number;
  min_profit_threshold: number;
  next_billing_date: string | null;
  default_payment_method_id: string | null;
}

export interface InvoiceData {
  id: string;
  billing_start_date: string | null;
  billing_end_date: string | null;
  total_period_profit: number;
  calculated_fee: number;
  status: string | null;
  payment_method_used: string | null;
  paid_at: string | null;
  created_at: string | null;
}

export interface PaymentMethodData {
  id: string;
  type: string | null;
  card_last4: string | null;
  card_brand: string | null;
  expiry_month: number | null;
  expiry_year: number | null;
  is_default: boolean;
}

export interface WeeklyPreviewData {
  week_start: string;
  week_end: string;
  gross_profit: number;
  gross_loss: number;
  net_profit: number;
  estimated_fee: number;
}

interface SubscriptionSummaryResponse {
  subscription: SubscriptionData;
  invoices: InvoiceData[];
  payment_methods: PaymentMethodData[];
  weekly_preview: WeeklyPreviewData;
}

interface CreateSetupIntentResponse {
  client_secret: string;
}

interface AttachPaymentMethodPayload {
  paymentMethodId: string;
  setAsDefault?: boolean;
}

export function useSubscription() {
  const { user, loading: authLoading } = useAuth();

  const [subscription, setSubscription] = useState<SubscriptionData | null>(null);
  const [invoices, setInvoices] = useState<InvoiceData[]>([]);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethodData[]>([]);
  const [weeklyPreview, setWeeklyPreview] = useState<WeeklyPreviewData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    if (authLoading) return;

    if (!user) {
      setSubscription(null);
      setInvoices([]);
      setPaymentMethods([]);
      setWeeklyPreview(null);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const { data } = await api.get<SubscriptionSummaryResponse>("/subscription/summary");
      setSubscription(data.subscription ?? null);
      setInvoices(data.invoices ?? []);
      setPaymentMethods(data.payment_methods ?? []);
      setWeeklyPreview(data.weekly_preview ?? null);
    } catch (error) {
      console.error("Error fetching subscription data:", error);
      toast.error("Failed to load subscription data");
    } finally {
      setLoading(false);
    }
  }, [user, authLoading]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const createSetupIntent = useCallback(async () => {
    if (!user) return null;

    try {
      const { data } = await api.post<CreateSetupIntentResponse>("/subscription/setup-intent");
      return data.client_secret;
    } catch (error: any) {
      console.error("Error creating setup intent:", error);
      toast.error(error?.message || "Failed to initialize Stripe");
      return null;
    }
  }, [user]);

  const addPaymentMethod = async (payload: AttachPaymentMethodPayload) => {
    if (!user) return null;

    try {
      const { data } = await api.post<PaymentMethodData>("/subscription/payment-methods", payload);
      toast.success("Payment method added");
      await fetchData();
      return data;
    } catch (error: any) {
      console.error("Error adding payment method:", error);
      toast.error(error?.message || "Failed to add payment method");
      return null;
    }
  };

  const setDefaultPaymentMethod = async (methodId: string) => {
    if (!user) return false;

    try {
      await api.patch(`/subscription/payment-methods/${methodId}/default`);
      toast.success("Default payment method updated");
      await fetchData();
      return true;
    } catch (error: any) {
      console.error("Error setting default payment method:", error);
      toast.error(error?.message || "Failed to update default payment method");
      return false;
    }
  };

  const removePaymentMethod = async (methodId: string) => {
    if (!user) return false;

    try {
      await api.delete(`/subscription/payment-methods/${methodId}`);
      toast.success("Payment method removed");
      await fetchData();
      return true;
    } catch (error: any) {
      console.error("Error removing payment method:", error);
      toast.error(error?.message || "Failed to remove payment method");
      return false;
    }
  };

  return {
    subscription,
    invoices,
    paymentMethods,
    weeklyPreview,
    loading,
    refetch: fetchData,
    createSetupIntent,
    addPaymentMethod,
    setDefaultPaymentMethod,
    removePaymentMethod,
  };
}
