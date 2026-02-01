import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import type { Database } from "@/lib/integrations/supabase/types";

type Subscription = Database["public"]["Tables"]["subscriptions"]["Row"];
type Invoice = Database["public"]["Tables"]["invoices"]["Row"];
type PaymentMethod = Database["public"]["Tables"]["user_payment_methods"]["Row"];

export function useSubscription() {
  const { user } = useAuth();
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    if (!user) {
      setSubscription(null);
      setInvoices([]);
      setPaymentMethods([]);
      setLoading(false);
      return;
    }

    try {
      // Fetch subscription
      const { data: subData } = await supabase
        .from("subscriptions")
        .select("*")
        .eq("user_id", user.id)
        .maybeSingle();

      setSubscription(subData);

      // Fetch invoices
      if (subData) {
        const { data: invoicesData } = await supabase
          .from("invoices")
          .select("*")
          .eq("sub_id", subData.id)
          .order("created_at", { ascending: false });
        setInvoices(invoicesData || []);
      }

      // Fetch payment methods
      const { data: paymentData } = await supabase
        .from("user_payment_methods")
        .select("*")
        .eq("user_id", user.id)
        .eq("is_active", true);

      setPaymentMethods(paymentData || []);
    } catch (error) {
      console.error("Error fetching subscription data:", error);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const addPaymentMethod = async (data: {
    type: string;
    card_last4?: string;
    card_brand?: string;
    expiry_month?: number;
    expiry_year?: number;
  }) => {
    if (!user) return null;

    try {
      // If this is the first payment method, make it default
      const isDefault = paymentMethods.length === 0;

      const { data: method, error } = await supabase
        .from("user_payment_methods")
        .insert({
          user_id: user.id,
          is_default: isDefault,
          ...data,
        })
        .select()
        .single();

      if (error) throw error;

      toast.success("Payment method added");
      await fetchData();
      return method;
    } catch (error) {
      console.error("Error adding payment method:", error);
      toast.error("Failed to add payment method");
      return null;
    }
  };

  const setDefaultPaymentMethod = async (methodId: string) => {
    if (!user) return false;

    try {
      // Remove default from all
      await supabase
        .from("user_payment_methods")
        .update({ is_default: false })
        .eq("user_id", user.id);

      // Set new default
      const { error } = await supabase
        .from("user_payment_methods")
        .update({ is_default: true })
        .eq("id", methodId);

      if (error) throw error;

      toast.success("Default payment method updated");
      await fetchData();
      return true;
    } catch (error) {
      console.error("Error setting default payment method:", error);
      toast.error("Failed to update default payment method");
      return false;
    }
  };

  const removePaymentMethod = async (methodId: string) => {
    try {
      const { error } = await supabase
        .from("user_payment_methods")
        .update({ is_active: false })
        .eq("id", methodId);

      if (error) throw error;

      toast.success("Payment method removed");
      await fetchData();
      return true;
    } catch (error) {
      console.error("Error removing payment method:", error);
      toast.error("Failed to remove payment method");
      return false;
    }
  };

  return {
    subscription,
    invoices,
    paymentMethods,
    loading,
    refetch: fetchData,
    addPaymentMethod,
    setDefaultPaymentMethod,
    removePaymentMethod,
  };
}
