import { useCallback, useEffect, useRef, useState } from "react";
import {
  CheckCircle,
  Clock,
  CreditCard,
  Lock,
  Receipt,
  Shield,
  Trash2,
} from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useAuth } from "@/contexts/AuthContext";
import { useSubscription, type CheckoutPaymentFlow, type InvoiceData, type PaymentMethodData } from "@/hooks/useSubscription";
import { cn } from "@/lib/utils";

type SubscriptionTab = "overview" | "history" | "payment";

const STRIPE_PUBLISHABLE_KEY = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY || "";
const STRIPE_READY = Boolean(STRIPE_PUBLISHABLE_KEY);

let stripeJsPromise: Promise<void> | null = null;

declare global {
  interface Window {
    Stripe?: any;
  }
}

function ensureStripeJsLoaded() {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Stripe can only run in browser"));
  }

  if (window.Stripe) {
    return Promise.resolve();
  }

  if (stripeJsPromise) {
    return stripeJsPromise;
  }

  stripeJsPromise = new Promise<void>((resolve, reject) => {
    const scriptSrc = "https://js.stripe.com/v3/";
    const existingScript = document.querySelector<HTMLScriptElement>(`script[src="${scriptSrc}"]`);

    if (existingScript) {
      existingScript.addEventListener("load", () => resolve(), { once: true });
      existingScript.addEventListener("error", () => reject(new Error("Failed to load Stripe.js")), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = scriptSrc;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Stripe.js"));
    document.head.appendChild(script);
  });

  return stripeJsPromise;
}

function formatCurrency(amount: number, currency = "USD") {
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      minimumFractionDigits: 2,
    }).format(amount);
  } catch {
    return `${currency.toUpperCase()} ${amount.toFixed(2)}`;
  }
}

function formatUsdCurrency(amount: number) {
  return formatCurrency(amount, "USD");
}

function formatDate(dateValue: string | null | undefined) {
  if (!dateValue) return "-";
  const normalized = dateValue.includes("T") ? dateValue : `${dateValue}T00:00:00`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function addDaysToIsoDate(dateValue: string | null | undefined, days: number) {
  if (!dateValue) return null;
  const normalized = dateValue.includes("T") ? dateValue : `${dateValue}T00:00:00`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return null;
  date.setDate(date.getDate() + days);
  return date.toISOString();
}

function formatInvoicePeriod(start: string | null, end: string | null, fallback: string | null) {
  if (start && end) {
    return `${formatDate(start)} - ${formatDate(end)}`;
  }
  if (start) {
    return formatDate(start);
  }
  if (end) {
    return formatDate(end);
  }
  return formatDate(fallback);
}

function formatPaymentMethodLabel(method: PaymentMethodData) {
  const brand = method.card_brand?.toUpperCase() || "CARD";
  const last4 = method.card_last4 || "----";
  const hasExpiry = method.expiry_month && method.expiry_year;
  const expiry = hasExpiry
    ? `${String(method.expiry_month).padStart(2, "0")}/${String(method.expiry_year).slice(-2)}`
    : "N/A";
  return {
    brand,
    last4,
    expiry,
    label: `${brand} ending in ${last4}`,
  };
}

function formatInvoiceCharge(invoice: InvoiceData) {
  return formatCurrency(invoice.payment_amount ?? invoice.calculated_fee ?? 0, invoice.payment_currency || "USD");
}

function getInvoicePaymentMethodLabel(invoice: InvoiceData) {
  const label = invoice.payment_method_label?.trim();
  if (label && label.toLowerCase() !== "not available") {
    return label;
  }
  return invoice.status?.toLowerCase() === "paid" ? "Recorded by Stripe" : "-";
}

interface StripeAddCardFormProps {
  clientSecret: string;
  setAsDefault: boolean;
  submitting: boolean;
  onSetAsDefaultChange: (checked: boolean) => void;
  onCancel: () => void;
  onSubmit: (paymentMethodId: string, setAsDefault: boolean) => Promise<void>;
}

function StripeAddCardForm({
  publishableKey,
  clientSecret,
  setAsDefault,
  submitting,
  onSetAsDefaultChange,
  onCancel,
  onSubmit,
}: StripeAddCardFormProps & { publishableKey: string }) {
  const paymentElementContainerRef = useRef<HTMLDivElement | null>(null);
  const stripeRef = useRef<any>(null);
  const elementsRef = useRef<any>(null);
  const paymentElementRef = useRef<any>(null);
  const [ready, setReady] = useState(false);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    let isActive = true;

    const init = async () => {
      try {
        await ensureStripeJsLoaded();
        if (!isActive || !window.Stripe || !paymentElementContainerRef.current) return;

        const stripe = window.Stripe(publishableKey);
        const appearance = {
          theme: "stripe",
          variables: {
            colorPrimary: "#2563eb",
            colorText: "#0f172a",
            colorDanger: "#dc2626",
            borderRadius: "10px",
          },
          inputs: "spaced",
          labels: "above",
        } as const;

        const elements = stripe.elements({
          clientSecret,
          appearance,
          loader: "auto",
        });

        const paymentElement = elements.create("payment", {
          layout: {
            type: "tabs",
            defaultCollapsed: false,
          },
        });

        paymentElement.mount(paymentElementContainerRef.current);

        stripeRef.current = stripe;
        elementsRef.current = elements;
        paymentElementRef.current = paymentElement;
        setReady(true);
      } catch (error: any) {
        toast.error(error?.message || "Failed to initialize Stripe");
      }
    };

    init();

    return () => {
      isActive = false;
      if (paymentElementRef.current) {
        paymentElementRef.current.destroy();
        paymentElementRef.current = null;
      }
      stripeRef.current = null;
      elementsRef.current = null;
      setReady(false);
    };
  }, [publishableKey, clientSecret]);

  const disabled = submitting || confirming || !ready;
  const actionLabel = confirming ? "Verifying..." : submitting ? "Saving..." : "Add Card";

  const handleSubmit = async () => {
    if (!stripeRef.current || !elementsRef.current) {
      toast.error("Stripe is not ready yet");
      return;
    }

    setConfirming(true);
    const submitResult = await elementsRef.current.submit?.();
    if (submitResult?.error) {
      setConfirming(false);
      toast.error(submitResult.error.message || "Please check your payment details");
      return;
    }

    const result = await stripeRef.current.confirmSetup({
      elements: elementsRef.current,
      confirmParams: {
        return_url: window.location.href,
      },
      redirect: "if_required",
    });
    setConfirming(false);

    if (result.error) {
      toast.error(result.error.message || "Failed to verify card");
      return;
    }

    const paymentMethod = result.setupIntent?.payment_method;
    const paymentMethodId =
      typeof paymentMethod === "string" ? paymentMethod : paymentMethod?.id;
    if (typeof paymentMethodId !== "string" || !paymentMethodId) {
      toast.error("Stripe did not return a payment method");
      return;
    }

    await onSubmit(paymentMethodId, setAsDefault);
  };

  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-border bg-white p-4">
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-foreground">Card details</p>
            <p className="text-xs text-muted-foreground">Enter your card once for manual or automatic billing.</p>
          </div>
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted/30 px-2 py-1 text-[11px] font-medium text-muted-foreground">
            <Lock className="h-3 w-3" />
            Secure
          </span>
        </div>
        <div className="rounded-lg border border-border bg-muted/20 p-3">
          <div ref={paymentElementContainerRef} className="min-h-[112px]" />
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {["VISA", "MASTERCARD", "AMEX"].map((brand) => (
            <span
              key={brand}
              className="rounded-full border border-border/70 bg-background/80 px-2 py-0.5 text-[11px] font-medium text-muted-foreground"
            >
              {brand}
            </span>
          ))}
        </div>
      </div>

      <div className="flex items-start gap-3 rounded-lg border border-border bg-muted/20 p-3">
        <Checkbox
          id="setAsDefault"
          checked={setAsDefault}
          onCheckedChange={(checked) => onSetAsDefaultChange(checked === true)}
        />
        <div className="space-y-0.5">
          <Label htmlFor="setAsDefault" className="text-sm font-medium">
            Set as default payment method
          </Label>
          <p className="text-xs text-muted-foreground">Recommended for uninterrupted weekly billing.</p>
        </div>
      </div>

      <div className="flex flex-col-reverse gap-2 pt-1 sm:flex-row sm:justify-end">
        <Button variant="outline" onClick={onCancel} disabled={disabled}>
          Cancel
        </Button>
        <Button onClick={handleSubmit} disabled={disabled}>
          {actionLabel}
        </Button>
      </div>
    </div>
  );
}

export default function Subscription() {
  const { refreshUser } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [activeTab, setActiveTab] = useState<SubscriptionTab>("overview");
  const [addCardOpen, setAddCardOpen] = useState(false);
  const [setupIntentSecret, setSetupIntentSecret] = useState<string | null>(null);
  const [setupIntentLoading, setSetupIntentLoading] = useState(false);
  const [setAsDefault, setSetAsDefault] = useState(false);
  const [submittingCard, setSubmittingCard] = useState(false);
  const [methodActionId, setMethodActionId] = useState<string | null>(null);
  const [invoiceActionId, setInvoiceActionId] = useState<string | null>(null);
  const [invoiceActionType, setInvoiceActionType] = useState<"download" | "pay" | "promptpay" | null>(null);
  const [pendingCollectionMode, setPendingCollectionMode] = useState<"automatic" | "manual">("automatic");
  const [savingCollectionMode, setSavingCollectionMode] = useState(false);
  const setupIntentRequestedRef = useRef(false);

  const {
    subscription,
    invoices,
    paymentMethods,
    weeklyPreview,
    loading,
    createSetupIntent,
    addPaymentMethod,
    setDefaultPaymentMethod,
    removePaymentMethod,
    downloadInvoice,
    createInvoiceCheckoutSession,
    confirmInvoiceCheckout,
    refetch,
    updateCollectionMode,
  } = useSubscription();

  const weeklyProfit = weeklyPreview?.net_profit ?? 0;
  const estimatedFee = weeklyPreview?.estimated_fee ?? 0;
  const previewPeriodLabel =
    weeklyPreview?.week_start && weeklyPreview?.week_end
      ? `${formatDate(weeklyPreview.week_start)} - ${formatDate(weeklyPreview.week_end)}`
      : null;

  const nextBillingLabel = formatDate(subscription?.next_billing_date);
  const subscriptionStatus = subscription?.status ?? "active";
  const statusIsActive = subscriptionStatus === "active";
  const collectionMode = subscription?.collection_mode ?? "automatic";
  const collectionModeLabel = collectionMode === "manual" ? "Manual payment" : "Automatic charge";
  const hasActivePaymentMethod = paymentMethods.length > 0;
  const canEnableAutomaticCollection = hasActivePaymentMethod;
  const promptpayEnabled = Boolean(subscription?.promptpay_enabled);
  const promptpayCurrency = String(subscription?.promptpay_currency || "THB").toUpperCase();
  const promptpayExchangeRate = subscription?.promptpay_exchange_rate ?? null;
  const promptpayMinimumUsdAmount =
    promptpayEnabled && promptpayExchangeRate && promptpayExchangeRate > 0 ? 10 / promptpayExchangeRate : null;
  const unresolvedInvoice = invoices.find((invoice) => {
    const status = String(invoice.status || "").toLowerCase();
    return status === "pending" || status === "failed";
  });
  const overviewInvoice = unresolvedInvoice ?? null;
  const overviewProfit = overviewInvoice ? overviewInvoice.total_period_profit ?? 0 : weeklyProfit;
  const overviewCharge = overviewInvoice ? overviewInvoice.calculated_fee ?? 0 : estimatedFee;
  const overviewPeriodLabel = overviewInvoice
    ? formatInvoicePeriod(overviewInvoice.billing_start_date, overviewInvoice.billing_end_date, overviewInvoice.created_at)
    : previewPeriodLabel;
  const overviewDueDate = overviewInvoice
    ? formatDate(addDaysToIsoDate(overviewInvoice.billing_end_date, 1))
    : nextBillingLabel;
  const overviewProfitLabel = overviewInvoice ? "Outstanding Billing Period Profit" : "Current Billing Period Profit";
  const overviewChargeLabel = overviewInvoice ? "Outstanding Charge (USD)" : "Estimated Charge (USD)";
  const overviewDateLabel = overviewInvoice ? "Invoice Due" : "Next Billing";
  const sectionClass = "rounded-xl border border-border bg-white";
  const statCellClass = "p-5";
  const subscriptionAlertMessage =
    subscriptionStatus === "canceled"
      ? "Your subscription is canceled. Reactivate billing before using bots again."
      : unresolvedInvoice
        ? "Your bot access is paused until the outstanding invoice is paid or skipped by admin."
        : "Your bot access is paused until billing is resolved.";

  const initializeStripeSetup = useCallback(async () => {
    if (!STRIPE_READY) {
      toast.error("Stripe publishable key is missing");
      setupIntentRequestedRef.current = false;
      return;
    }

    setSetupIntentLoading(true);
    const clientSecret = await createSetupIntent();
    if (clientSecret) {
      setSetupIntentSecret(clientSecret);
    } else {
      setupIntentRequestedRef.current = false;
    }
    setSetupIntentLoading(false);
  }, [createSetupIntent]);

  useEffect(() => {
    if (!addCardOpen) {
      setSetupIntentSecret(null);
      setSetAsDefault(false);
      setupIntentRequestedRef.current = false;
      return;
    }

    setSetAsDefault(paymentMethods.length === 0);
    if (setupIntentSecret || setupIntentRequestedRef.current) {
      return;
    }

    setupIntentRequestedRef.current = true;
    initializeStripeSetup();
  }, [addCardOpen, paymentMethods.length, setupIntentSecret, initializeStripeSetup]);

  useEffect(() => {
    setPendingCollectionMode(collectionMode === "manual" ? "manual" : "automatic");
  }, [collectionMode]);

  useEffect(() => {
    const checkoutStatus = searchParams.get("checkout");
    const checkoutInvoiceId = searchParams.get("invoice_id");
    const checkoutSessionId = searchParams.get("session_id");
    if (!checkoutStatus) {
      return;
    }

    const handleCheckoutReturn = async () => {
      if (checkoutStatus === "success") {
        let confirmedStatus: string | null = null;
        if (checkoutInvoiceId) {
          const confirmedInvoice = await confirmInvoiceCheckout(checkoutInvoiceId, checkoutSessionId);
          confirmedStatus = confirmedInvoice?.status?.toLowerCase() ?? null;
        }

        if (confirmedStatus === "paid") {
          toast.success("Payment confirmed");
        } else {
          toast.success("Returned from Stripe Checkout. Refreshing billing status...");
          await Promise.all([refetch(), refreshUser()]);
          window.setTimeout(() => {
            void Promise.all([refetch(), refreshUser()]);
          }, 2500);
        }
        setActiveTab("history");
      } else if (checkoutStatus === "cancelled") {
        toast.message("Stripe Checkout was cancelled");
        setActiveTab("history");
      }

      router.replace(pathname || "/subscription");
    };

    handleCheckoutReturn();
  }, [confirmInvoiceCheckout, pathname, refetch, refreshUser, router, searchParams]);

  const onAttachPaymentMethod = async (paymentMethodId: string, makeDefault: boolean) => {
    setSubmittingCard(true);
    const added = await addPaymentMethod({
      paymentMethodId,
      setAsDefault: makeDefault,
    });
    setSubmittingCard(false);

    if (added) {
      setAddCardOpen(false);
      setSetupIntentSecret(null);
    }
  };

  const onSetDefault = async (methodId: string) => {
    setMethodActionId(methodId);
    await setDefaultPaymentMethod(methodId);
    setMethodActionId(null);
  };

  const onRemoveMethod = async (methodId: string) => {
    setMethodActionId(methodId);
    await removePaymentMethod(methodId);
    setMethodActionId(null);
  };

  const onDownloadInvoice = async (invoiceId: string) => {
    setInvoiceActionId(invoiceId);
    setInvoiceActionType("download");
    await downloadInvoice(invoiceId);
    setInvoiceActionId(null);
    setInvoiceActionType(null);
  };

  const onCheckoutInvoice = async (invoice: InvoiceData, paymentFlow: CheckoutPaymentFlow) => {
    setInvoiceActionId(invoice.id);
    setInvoiceActionType(paymentFlow === "promptpay" ? "promptpay" : "pay");

    try {
      const session = await createInvoiceCheckoutSession(invoice.id, paymentFlow);
      if (!session) {
        return;
      }

      if (STRIPE_READY && session.session_id) {
        try {
          await ensureStripeJsLoaded();
          if (window.Stripe) {
            const stripe = window.Stripe(STRIPE_PUBLISHABLE_KEY);
            const result = await stripe.redirectToCheckout({ sessionId: session.session_id });
            if (result?.error) {
              if (session.url) {
                window.location.href = session.url;
                return;
              }
              throw new Error(result.error.message || "Failed to redirect to Stripe Checkout");
            }
            return;
          }
        } catch {
          if (session.url) {
            window.location.href = session.url;
            return;
          }
          throw new Error("Failed to initialize Stripe.js");
        }
      }

      if (session.url) {
        window.location.href = session.url;
        return;
      }

      throw new Error("Stripe Checkout session is missing both session id and URL");
    } catch (error: any) {
      toast.error(error?.message || "Failed to open Stripe Checkout");
    } finally {
      setInvoiceActionId(null);
      setInvoiceActionType(null);
    }
  };

  const onSaveCollectionMode = async () => {
    if (pendingCollectionMode === "automatic" && !hasActivePaymentMethod) {
      toast.error("Add a saved card before switching billing to automatic");
      setActiveTab("payment");
      return;
    }

    if (pendingCollectionMode === collectionMode) {
      return;
    }

    setSavingCollectionMode(true);
    await updateCollectionMode(pendingCollectionMode);
    setSavingCollectionMode(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Subscription & Billing</h1>
        <p className="text-sm text-muted-foreground">Manage your weekly profit-based subscription</p>
      </div>

      <div className={sectionClass}>
        <div className="flex flex-col gap-4 border-b border-border p-5 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-10 w-10 items-center justify-center rounded-lg border",
                statusIsActive
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-amber-200 bg-amber-50 text-amber-700"
              )}
            >
              <CheckCircle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-lg font-semibold capitalize">{subscriptionStatus.replace("_", " ")} Subscription</h3>
              <p className="text-sm text-muted-foreground">
                Weekly profit-based billing • {collectionModeLabel}
              </p>
            </div>
          </div>
          <span
            className={cn(
              "inline-flex items-center rounded-full border px-3 py-1 text-sm font-medium capitalize",
              statusIsActive
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-amber-200 bg-amber-50 text-amber-700"
            )}
          >
            {subscriptionStatus.replace("_", " ")}
          </span>
        </div>

        <div className="grid divide-y divide-border md:grid-cols-4 md:divide-x md:divide-y-0">
          <div className={statCellClass}>
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">{overviewProfitLabel}</p>
            <p className={cn("mt-2 text-2xl font-semibold font-mono", overviewProfit >= 0 ? "profit-text" : "loss-text")}>
              {overviewProfit >= 0 ? "+" : ""}
              {formatUsdCurrency(overviewProfit)}
            </p>
          </div>
          <div className={statCellClass}>
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">Collection Mode</p>
            <p className="mt-2 text-2xl font-semibold">{collectionModeLabel}</p>
          </div>
          <div className={statCellClass}>
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">{overviewChargeLabel}</p>
            <p className="mt-2 text-2xl font-semibold font-mono">{formatUsdCurrency(overviewCharge)}</p>
          </div>
          <div className={statCellClass}>
            <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">{overviewDateLabel}</p>
            <p className="mt-2 text-2xl font-semibold font-mono">{overviewDueDate}</p>
          </div>
        </div>
      </div>

      {!statusIsActive && (
        <Alert className="border-amber-200 bg-amber-50 text-foreground [&>svg]:text-amber-700">
          <Shield className="h-4 w-4" />
          <AlertTitle className="capitalize">{subscriptionStatus.replace("_", " ")} subscription</AlertTitle>
          <AlertDescription>
            {subscriptionAlertMessage}
          </AlertDescription>
        </Alert>
      )}

      {!hasActivePaymentMethod && (
        <Alert className="border-border bg-white text-foreground [&>svg]:text-muted-foreground">
          <CreditCard className="h-4 w-4" />
          <AlertTitle>Saved card optional in manual mode</AlertTitle>
          <AlertDescription className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <span>
              Add a card for automatic billing, or switch to manual payment and choose either card checkout with card saving
              or PromptPay checkout in THB when you pay an invoice.
            </span>
            <Button size="sm" variant="outline" className="w-full md:w-auto" onClick={() => setActiveTab("payment")}>
              Open Payment Methods
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div className="inline-flex rounded-lg border border-border bg-white p-1">
        {[
          { id: "overview", label: "Overview", icon: Receipt },
          { id: "history", label: "Billing History", icon: Clock },
          { id: "payment", label: "Payment Methods", icon: CreditCard },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as SubscriptionTab)}
            className={cn(
              "flex items-center gap-2 rounded-md px-4 py-2.5 text-sm font-medium transition-colors",
              activeTab === tab.id
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "overview" && (
        <div className="space-y-6">
          <div className={`${sectionClass} p-6`}>
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-semibold">Billing Preferences</h3>
                <p className="text-sm text-muted-foreground">
                  Choose whether invoices are charged automatically or paid manually when due.
                </p>
              </div>
              <Button
                onClick={onSaveCollectionMode}
                disabled={
                  savingCollectionMode ||
                  pendingCollectionMode === collectionMode ||
                  (pendingCollectionMode === "automatic" && !canEnableAutomaticCollection)
                }
              >
                {savingCollectionMode ? "Saving..." : "Save Collection Mode"}
              </Button>
            </div>

            <RadioGroup
              value={pendingCollectionMode}
              onValueChange={(value) => setPendingCollectionMode(value as "automatic" | "manual")}
              className="mt-5 grid gap-3 md:grid-cols-2"
              disabled={savingCollectionMode}
            >
              <Label
                htmlFor="collection-mode-automatic"
                className={cn(
                  "flex cursor-pointer items-start gap-3 rounded-lg border p-4 transition-colors",
                  pendingCollectionMode === "automatic" ? "border-foreground bg-muted/20" : "border-border bg-white",
                  (!canEnableAutomaticCollection || savingCollectionMode) && "opacity-70"
                )}
              >
                <RadioGroupItem
                  value="automatic"
                  id="collection-mode-automatic"
                  className="mt-1"
                  disabled={!canEnableAutomaticCollection}
                />
                <div>
                  <p className="text-sm font-semibold text-foreground">Automatic charge</p>
                  <p className="text-xs text-muted-foreground">
                    On the billing date, the system charges your saved cards automatically and can fallback to other active cards.
                  </p>
                </div>
              </Label>

              <Label
                htmlFor="collection-mode-manual"
                className={cn(
                  "flex cursor-pointer items-start gap-3 rounded-lg border p-4 transition-colors",
                  pendingCollectionMode === "manual" ? "border-foreground bg-muted/20" : "border-border bg-white",
                  savingCollectionMode && "opacity-70"
                )}
              >
                <RadioGroupItem value="manual" id="collection-mode-manual" className="mt-1" />
                <div>
                  <p className="text-sm font-semibold text-foreground">Manual payment</p>
                  <p className="text-xs text-muted-foreground">
                    When an invoice becomes due, access pauses until you open Stripe Checkout and choose either card
                    payment with save-for-later support or PromptPay in THB.
                  </p>
                </div>
              </Label>
            </RadioGroup>

            {!hasActivePaymentMethod && (
              <p className="mt-4 text-xs text-muted-foreground">
                Automatic billing still requires at least one saved card. Manual billing can work without a saved card.
              </p>
            )}
            {promptpayEnabled && promptpayExchangeRate ? (
              <p className="mt-2 text-xs text-muted-foreground">
                PromptPay sessions are charged in {promptpayCurrency} using the latest fetched rate of {promptpayExchangeRate.toFixed(2)} {promptpayCurrency} per USD.
              </p>
            ) : null}
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <div className={`${sectionClass} p-6`}>
              <h3 className="text-lg font-semibold mb-4">How It Works</h3>
              <div className="divide-y divide-border">
                <div className="flex gap-4 py-4 first:pt-0">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-muted/20 text-sm font-semibold text-foreground">
                    1
                  </div>
                  <div>
                    <p className="font-medium">Billing Period Profit Calculation</p>
                    <p className="text-sm text-muted-foreground">
                      Net profit is calculated from your active connected accounts across the next billable 7-day period.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4 py-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-muted/20 text-sm font-semibold text-foreground">
                    2
                  </div>
                  <div>
                    <p className="font-medium">Profit-Based Fee</p>
                    <p className="text-sm text-muted-foreground">
                      Fee is only charged when net profit is above your subscription threshold.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4 py-4 last:pb-0">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-muted/20 text-sm font-semibold text-foreground">
                    3
                  </div>
                  <div>
                    <p className="font-medium">
                      {collectionMode === "manual" ? "Manual Payment Required" : "Automatic Billing"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {collectionMode === "manual"
                        ? "On the billing date, an invoice is created and bot access is paused until you pay it in Stripe Checkout or admin skips it."
                        : "On the billing date, that 7-day period is invoiced using your saved payment methods automatically."}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className={`${sectionClass} p-6`}>
              <div className="mb-4">
                <h3 className="text-lg font-semibold">
                  {overviewInvoice ? "Outstanding Invoice Summary" : "Billing Period Estimate"}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {overviewPeriodLabel ? `For ${overviewPeriodLabel}` : "For your next billing period"}
                </p>
              </div>
              <div className="divide-y divide-border">
                <div className="flex justify-between py-3">
                  <span className="text-muted-foreground">Gross Profit</span>
                  <span className="font-mono profit-text">+{formatUsdCurrency(weeklyPreview?.gross_profit ?? 0)}</span>
                </div>
                <div className="flex justify-between py-3">
                  <span className="text-muted-foreground">Gross Loss</span>
                  <span className="font-mono loss-text">-{formatUsdCurrency(weeklyPreview?.gross_loss ?? 0)}</span>
                </div>
                <div className="flex justify-between py-3">
                  <span className="text-muted-foreground">Net Profit</span>
                  <span className={cn("font-mono font-bold", overviewProfit >= 0 ? "profit-text" : "loss-text")}>
                    {overviewProfit >= 0 ? "+" : ""}
                    {formatUsdCurrency(overviewProfit)}
                  </span>
                </div>
                <div className="flex justify-between py-3">
                  <span className="text-muted-foreground">{overviewInvoice ? "Outstanding Charge" : "Estimated Charge"}</span>
                  <span className="font-mono font-bold">{formatUsdCurrency(overviewCharge)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "history" && (
        <div className={`${sectionClass} p-6`}>
          {invoices.length === 0 ? (
            <div className="text-center text-sm text-muted-foreground py-12">
              No invoices yet.
              {nextBillingLabel !== "-" ? ` Your next billing date is ${nextBillingLabel}.` : " Your billing history will appear here."}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-2 py-3 text-left text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Period</th>
                    <th className="px-2 py-3 text-right text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Net Profit</th>
                    <th className="px-2 py-3 text-right text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Charge</th>
                    <th className="px-2 py-3 text-left text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Method</th>
                    <th className="px-2 py-3 text-center text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Status</th>
                    <th className="px-2 py-3 text-right text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {invoices.map((invoice) => {
                    const profit = invoice.total_period_profit ?? 0;
                    const status = (invoice.status ?? "pending").toLowerCase();
                    const isActionLoading = invoiceActionId === invoice.id;
                    const canPayNow = status === "pending" || status === "failed";
                    const canDownload = status === "paid";
                    const hasVisibleAction = canPayNow || canDownload;

                    return (
                      <tr key={invoice.id} className="border-b border-border/70">
                        <td className="py-4 px-2 font-medium">
                          {formatInvoicePeriod(invoice.billing_start_date, invoice.billing_end_date, invoice.created_at)}
                        </td>
                        <td className="py-4 px-2 text-right">
                          <span className={cn("font-mono", profit >= 0 ? "profit-text" : "loss-text")}>
                            {profit >= 0 ? "+" : ""}
                            {formatUsdCurrency(profit)}
                          </span>
                        </td>
                        <td className="py-4 px-2 text-right">
                          <div className="font-mono">{formatInvoiceCharge(invoice)}</div>
                          {invoice.payment_currency && invoice.payment_currency.toUpperCase() !== "USD" && (
                            <div className="mt-1 text-xs text-muted-foreground">
                              USD basis {formatUsdCurrency(invoice.calculated_fee ?? 0)}
                            </div>
                          )}
                        </td>
                        <td className="py-4 px-2 text-sm text-muted-foreground">
                          {getInvoicePaymentMethodLabel(invoice)}
                        </td>
                        <td className="py-4 px-2 text-center">
                          <span
                            className={cn(
                              "inline-flex rounded-full border px-2 py-1 text-xs font-medium capitalize",
                              status === "paid" && "border-emerald-200 bg-emerald-50 text-emerald-700",
                              status === "pending" && "border-amber-200 bg-amber-50 text-amber-700",
                              status === "failed" && "border-red-200 bg-red-50 text-red-700",
                              status === "skipped" && "border-border bg-muted/30 text-muted-foreground"
                            )}
                          >
                            {status}
                          </span>
                        </td>
                        <td className="py-4 px-2 text-right">
                          <div className="flex items-center justify-end gap-2">
                            {canPayNow && (
                              <>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  disabled={isActionLoading}
                                  onClick={() => onCheckoutInvoice(invoice, "card")}
                                >
                                  {isActionLoading && invoiceActionType === "pay" ? "Opening..." : "Pay card"}
                                </Button>
                                {promptpayEnabled && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    disabled={
                                      isActionLoading ||
                                      Boolean(
                                        promptpayMinimumUsdAmount &&
                                        (invoice.calculated_fee ?? 0) < promptpayMinimumUsdAmount
                                      )
                                    }
                                    onClick={() => onCheckoutInvoice(invoice, "promptpay")}
                                  >
                                    {isActionLoading && invoiceActionType === "promptpay" ? "Opening..." : "Pay PromptPay"}
                                  </Button>
                                )}
                              </>
                            )}
                            {canDownload && (
                              <Button
                                variant="ghost"
                                size="sm"
                                disabled={isActionLoading}
                                onClick={() => onDownloadInvoice(invoice.id)}
                              >
                                {isActionLoading && invoiceActionType === "download" ? "Downloading..." : "Download PDF"}
                              </Button>
                            )}
                            {!hasVisibleAction && (
                              <span className="text-xs text-muted-foreground">No action</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === "payment" && (
        <div className="space-y-4">
          {paymentMethods.length === 0 ? (
            <div className={`${sectionClass} p-8 text-center`}>
              <p className="text-sm text-muted-foreground mb-2">No saved card added yet.</p>
              <p className="text-xs text-muted-foreground mb-4">
                You can still pay due invoices in Stripe Checkout. Paying by card there will save a card for future automatic billing.
                {promptpayEnabled && promptpayExchangeRate
                  ? ` PromptPay is also available in ${promptpayCurrency} at about ${promptpayExchangeRate.toFixed(2)} ${promptpayCurrency}/USD.`
                  : ""}
              </p>
              <Button variant="outline" onClick={() => setAddCardOpen(true)}>
                <CreditCard className="w-4 h-4" />
                Add Payment Method
              </Button>
            </div>
          ) : (
            <>
              {paymentMethods.map((method) => {
                const { brand, last4, expiry } = formatPaymentMethodLabel(method);
                const isActionLoading = methodActionId === method.id;

                return (
                  <div
                    key={method.id}
                    className={cn(
                      "flex items-center justify-between rounded-lg border border-border bg-white p-4",
                      method.is_default && "border-foreground"
                    )}
                  >
                    <div className="flex items-center gap-4">
                      <div className="flex h-10 w-12 items-center justify-center rounded border border-border bg-muted/20">
                        <CreditCard className="w-5 h-5 text-muted-foreground" />
                      </div>
                      <div>
                        <p className="font-medium">
                          {brand} •••• {last4}
                        </p>
                        <p className="text-sm text-muted-foreground">Expires {expiry}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {method.is_default && (
                        <span className="rounded-full border border-border bg-muted/20 px-2 py-1 text-xs font-medium text-foreground">
                          Default
                        </span>
                      )}
                      {!method.is_default && (
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={isActionLoading}
                          onClick={() => onSetDefault(method.id)}
                        >
                          Set Default
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={isActionLoading}
                        onClick={() => onRemoveMethod(method.id)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                );
              })}

              <Button variant="outline" className="w-full" onClick={() => setAddCardOpen(true)}>
                <CreditCard className="w-4 h-4" />
                Add Payment Method
              </Button>
            </>
          )}

          <div className="flex items-center gap-2 justify-center text-sm text-muted-foreground mt-4">
            <Shield className="w-4 h-4" />
            <span>
              Payments secured by Stripe. Card checkout saves cards for automatic billing, while PromptPay checkout pays in THB.
            </span>
          </div>
        </div>
      )}

      <Dialog open={addCardOpen} onOpenChange={setAddCardOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Add Payment Method</DialogTitle>
            <DialogDescription>
              Your card details are collected securely by Stripe.
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-lg border border-border bg-muted/20 p-4">
            <div className="flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border bg-white">
                <CreditCard className="h-4 w-4 text-muted-foreground" />
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground">Secure card setup</p>
                <p className="text-xs text-muted-foreground">
                  Tokenized by Stripe. We never store your full card number or CVC. Paying an invoice in Checkout can also save the card automatically for future billing.
                </p>
              </div>
            </div>
          </div>

          {!STRIPE_READY && (
            <div className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
              Missing `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` in client env.
            </div>
          )}

          {STRIPE_READY && setupIntentLoading && (
            <div className="space-y-3 py-2">
              <div className="h-24 animate-pulse rounded-xl border border-border bg-secondary/40" />
              <div className="h-12 animate-pulse rounded-lg border border-border bg-secondary/30" />
              <div className="flex items-center justify-center pt-1 text-xs text-muted-foreground">
                Preparing secure payment form...
              </div>
            </div>
          )}

          {STRIPE_READY && !setupIntentLoading && setupIntentSecret && (
            <StripeAddCardForm
              publishableKey={STRIPE_PUBLISHABLE_KEY}
              clientSecret={setupIntentSecret}
              submitting={submittingCard}
              setAsDefault={setAsDefault}
              onSetAsDefaultChange={setSetAsDefault}
              onCancel={() => setAddCardOpen(false)}
              onSubmit={onAttachPaymentMethod}
            />
          )}

          {STRIPE_READY && !setupIntentLoading && !setupIntentSecret && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">Unable to initialize Stripe setup intent.</p>
              <Button
                variant="outline"
                onClick={() => {
                  setupIntentRequestedRef.current = false;
                  initializeStripeSetup();
                }}
              >
                Retry
              </Button>
            </div>
          )}
        </DialogContent>
      </Dialog>

    </div>
  );
}
