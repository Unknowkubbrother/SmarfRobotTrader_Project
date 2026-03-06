import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle,
  Clock,
  CreditCard,
  Download,
  Lock,
  Receipt,
  Shield,
  Sparkles,
  Star,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

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
import { useSubscription } from "@/hooks/useSubscription";
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

function formatCurrency(amount: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(amount);
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
      <div className="rounded-xl border border-primary/20 bg-gradient-to-br from-primary/10 via-primary/5 to-background p-4">
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-foreground">Card details</p>
            <p className="text-xs text-muted-foreground">Enter your card once for weekly automatic billing.</p>
          </div>
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-background/80 px-2 py-1 text-[11px] font-medium text-muted-foreground">
            <Lock className="h-3 w-3" />
            PCI DSS
          </span>
        </div>
        <div className="rounded-lg border border-border bg-background/80 p-3 shadow-inner">
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

      <div className="flex items-start gap-3 rounded-lg border border-border bg-secondary/30 p-3">
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
  const [activeTab, setActiveTab] = useState<SubscriptionTab>("overview");
  const [addCardOpen, setAddCardOpen] = useState(false);
  const [setupIntentSecret, setSetupIntentSecret] = useState<string | null>(null);
  const [setupIntentLoading, setSetupIntentLoading] = useState(false);
  const [setAsDefault, setSetAsDefault] = useState(false);
  const [submittingCard, setSubmittingCard] = useState(false);
  const [methodActionId, setMethodActionId] = useState<string | null>(null);
  const [invoiceActionId, setInvoiceActionId] = useState<string | null>(null);
  const [invoiceActionType, setInvoiceActionType] = useState<"download" | "pay" | null>(null);
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
    payInvoice,
  } = useSubscription();

  const weeklyProfit = weeklyPreview?.net_profit ?? 0;
  const estimatedFee = weeklyPreview?.estimated_fee ?? 0;

  const feeLabel = useMemo(() => {
    if (!subscription) return "20%";
    if (subscription.fee_type === "fixed") {
      return formatCurrency(subscription.fee_value);
    }
    return `${subscription.fee_value}%`;
  }, [subscription]);

  const nextBillingLabel = formatDate(subscription?.next_billing_date);
  const subscriptionStatus = subscription?.status ?? "active";
  const statusIsActive = subscriptionStatus === "active";

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

  const onPayInvoice = async (invoiceId: string) => {
    setInvoiceActionId(invoiceId);
    setInvoiceActionType("pay");
    await payInvoice(invoiceId);
    setInvoiceActionId(null);
    setInvoiceActionType(null);
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

      <div className="glass-card p-6 animate-slide-up">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "w-12 h-12 rounded-xl flex items-center justify-center",
                statusIsActive ? "bg-success/10" : "bg-warning/10"
              )}
            >
              <CheckCircle className={cn("w-6 h-6", statusIsActive ? "text-success" : "text-warning")} />
            </div>
            <div>
              <h3 className="text-lg font-semibold capitalize">{subscriptionStatus.replace("_", " ")} Subscription</h3>
              <p className="text-sm text-muted-foreground">Weekly profit-based billing</p>
            </div>
          </div>
          <span
            className={cn(
              "px-3 py-1 rounded-full text-sm font-medium capitalize",
              statusIsActive ? "bg-success/10 text-success" : "bg-warning/10 text-warning"
            )}
          >
            {subscriptionStatus.replace("_", " ")}
          </span>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="p-4 rounded-lg bg-secondary/50">
            <p className="text-sm text-muted-foreground mb-1">This Week&apos;s Profit</p>
            <p className={cn("text-2xl font-bold font-mono", weeklyProfit >= 0 ? "profit-text" : "loss-text")}>
              {weeklyProfit >= 0 ? "+" : ""}
              {formatCurrency(weeklyProfit)}
            </p>
          </div>
          <div className="p-4 rounded-lg bg-secondary/50">
            <p className="text-sm text-muted-foreground mb-1">Estimated Fee ({feeLabel})</p>
            <p className="text-2xl font-bold font-mono">{formatCurrency(estimatedFee)}</p>
          </div>
          <div className="p-4 rounded-lg bg-secondary/50">
            <p className="text-sm text-muted-foreground mb-1">Next Billing</p>
            <p className="text-2xl font-bold font-mono">{nextBillingLabel}</p>
          </div>
        </div>
      </div>

      <div className="flex gap-2 border-b border-border">
        {[
          { id: "overview", label: "Overview", icon: Receipt },
          { id: "history", label: "Billing History", icon: Clock },
          { id: "payment", label: "Payment Methods", icon: CreditCard },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as SubscriptionTab)}
            className={cn(
              "flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 -mb-px",
              activeTab === tab.id
                ? "text-primary border-primary"
                : "text-muted-foreground border-transparent hover:text-foreground"
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "overview" && (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="glass-card p-6 animate-slide-up">
            <h3 className="text-lg font-semibold mb-4">How It Works</h3>
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <span className="text-primary font-bold">1</span>
                </div>
                <div>
                  <p className="font-medium">Weekly Profit Calculation</p>
                  <p className="text-sm text-muted-foreground">
                    Net profit is calculated from your connected accounts during the current week.
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <span className="text-primary font-bold">2</span>
                </div>
                <div>
                  <p className="font-medium">Profit-Based Fee</p>
                  <p className="text-sm text-muted-foreground">
                    Fee is only charged when net profit is above your subscription threshold.
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                  <span className="text-primary font-bold">3</span>
                </div>
                <div>
                  <p className="font-medium">Automatic Billing</p>
                  <p className="text-sm text-muted-foreground">
                    Charges use your default payment method and invoices are listed in billing history.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
            <h3 className="text-lg font-semibold mb-4">This Week Preview</h3>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Gross Profit</span>
                <span className="font-mono profit-text">+{formatCurrency(weeklyPreview?.gross_profit ?? 0)}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Gross Loss</span>
                <span className="font-mono loss-text">-{formatCurrency(weeklyPreview?.gross_loss ?? 0)}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Net Profit</span>
                <span className={cn("font-mono font-bold", weeklyProfit >= 0 ? "profit-text" : "loss-text")}>
                  {weeklyProfit >= 0 ? "+" : ""}
                  {formatCurrency(weeklyProfit)}
                </span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-muted-foreground">Estimated Fee</span>
                <span className="font-mono font-bold">{formatCurrency(estimatedFee)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "history" && (
        <div className="glass-card p-6 animate-slide-up">
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
                    <th className="text-left text-xs font-medium text-muted-foreground py-3 px-2">Period</th>
                    <th className="text-right text-xs font-medium text-muted-foreground py-3 px-2">Net Profit</th>
                    <th className="text-right text-xs font-medium text-muted-foreground py-3 px-2">Fee</th>
                    <th className="text-center text-xs font-medium text-muted-foreground py-3 px-2">Status</th>
                    <th className="text-right text-xs font-medium text-muted-foreground py-3 px-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {invoices.map((invoice) => {
                    const profit = invoice.total_period_profit ?? 0;
                    const status = (invoice.status ?? "pending").toLowerCase();
                    const isActionLoading = invoiceActionId === invoice.id;
                    const canPayNow = status === "pending" || status === "failed";

                    return (
                      <tr key={invoice.id} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
                        <td className="py-4 px-2 font-medium">
                          {formatInvoicePeriod(invoice.billing_start_date, invoice.billing_end_date, invoice.created_at)}
                        </td>
                        <td className="py-4 px-2 text-right">
                          <span className={cn("font-mono", profit >= 0 ? "profit-text" : "loss-text")}>
                            {profit >= 0 ? "+" : ""}
                            {formatCurrency(profit)}
                          </span>
                        </td>
                        <td className="py-4 px-2 text-right font-mono">{formatCurrency(invoice.calculated_fee ?? 0)}</td>
                        <td className="py-4 px-2 text-center">
                          <span
                            className={cn(
                              "px-2 py-1 rounded-full text-xs font-medium capitalize",
                              status === "paid" && "bg-success/10 text-success",
                              status === "pending" && "bg-warning/10 text-warning",
                              status === "failed" && "bg-destructive/10 text-destructive",
                              status === "skipped" && "bg-muted text-muted-foreground"
                            )}
                          >
                            {status}
                          </span>
                        </td>
                        <td className="py-4 px-2 text-right">
                          <div className="flex items-center justify-end gap-2">
                            {canPayNow && (
                              <Button
                                variant="outline"
                                size="sm"
                                disabled={isActionLoading}
                                onClick={() => onPayInvoice(invoice.id)}
                              >
                                {isActionLoading && invoiceActionType === "pay" ? "Paying..." : "Pay now"}
                              </Button>
                            )}
                            <Button
                              variant="ghost"
                              size="sm"
                              disabled={isActionLoading}
                              onClick={() => onDownloadInvoice(invoice.id)}
                            >
                              {isActionLoading && invoiceActionType === "download" ? (
                                "Downloading..."
                              ) : (
                                <Download className="w-4 h-4" />
                              )}
                            </Button>
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
            <div className="glass-card p-8 text-center">
              <p className="text-sm text-muted-foreground mb-4">No payment method added yet.</p>
              <Button variant="outline" onClick={() => setAddCardOpen(true)}>
                <CreditCard className="w-4 h-4" />
                Add Payment Method
              </Button>
            </div>
          ) : (
            <>
              {paymentMethods.map((method, index) => {
                const brand = method.card_brand?.toUpperCase() || "CARD";
                const last4 = method.card_last4 || "----";
                const hasExpiry = method.expiry_month && method.expiry_year;
                const expiry = hasExpiry
                  ? `${String(method.expiry_month).padStart(2, "0")}/${String(method.expiry_year).slice(-2)}`
                  : "N/A";
                const isActionLoading = methodActionId === method.id;

                return (
                  <div
                    key={method.id}
                    className={cn(
                      "glass-card p-4 flex items-center justify-between animate-slide-up",
                      method.is_default && "ring-1 ring-primary"
                    )}
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-8 rounded bg-secondary flex items-center justify-center">
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
                        <span className="px-2 py-1 rounded bg-primary/10 text-primary text-xs font-medium flex items-center gap-1">
                          <Star className="w-3 h-3" />
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
            <span>Payments secured by Stripe</span>
          </div>
        </div>
      )}

      <Dialog open={addCardOpen} onOpenChange={setAddCardOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              Add Payment Method
            </DialogTitle>
            <DialogDescription>
              Your card details are collected securely by Stripe.
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-xl border border-primary/20 bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-4">
            <div className="flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/15">
                <CreditCard className="h-4 w-4 text-primary" />
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground">Secure card setup</p>
                <p className="text-xs text-muted-foreground">
                  Tokenized by Stripe. We never store your full card number or CVC.
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
