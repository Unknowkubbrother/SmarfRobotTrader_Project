import { useEffect, useState } from "react";
import { DollarSign, Percent, Settings, Save } from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface AdminBillingConfig {
  config_id: number | null;
  default_fee_type: "percentage" | "fixed";
  default_fee_value: number;
  default_min_threshold: number;
  default_next_billing_date: string | null;
  updated_at: string | null;
}

interface AdminSubscriptionItem {
  id: string;
  user_id: string;
  user_email: string | null;
  status: "active" | "past_due" | "canceled";
  fee_type: "percentage" | "fixed";
  fee_value: number;
  min_profit_threshold: number;
  next_billing_date: string | null;
  created_at: string | null;
}

interface AdminSubscriptionManagementResponse {
  billing_config: AdminBillingConfig;
  subscriptions: AdminSubscriptionItem[];
}

export function AdminSubscriptions() {
  const [subscriptions, setSubscriptions] = useState<AdminSubscriptionItem[]>([]);
  const [billingConfig, setBillingConfig] = useState<AdminBillingConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingConfig, setSavingConfig] = useState(false);
  const [updatingSubId, setUpdatingSubId] = useState<string | null>(null);
  const [savingNextBillingSubId, setSavingNextBillingSubId] = useState<string | null>(null);
  const [nextBillingDrafts, setNextBillingDrafts] = useState<Record<string, string>>({});
  const [configForm, setConfigForm] = useState({
    fee_type: "percentage" as "percentage" | "fixed",
    fee_value: 20,
    min_threshold: 0,
    next_billing_date: "",
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const { data } = await api.get<AdminSubscriptionManagementResponse>("/subscription/admin/management");
      const rows = data.subscriptions || [];
      setSubscriptions(rows);
      setNextBillingDrafts(
        rows.reduce<Record<string, string>>((acc, row) => {
          acc[row.id] = row.next_billing_date ?? "";
          return acc;
        }, {})
      );
      setBillingConfig(data.billing_config || null);
      setConfigForm({
        fee_type: (data.billing_config?.default_fee_type ?? "percentage") as "percentage" | "fixed",
        fee_value: data.billing_config?.default_fee_value ?? 20,
        min_threshold: data.billing_config?.default_min_threshold ?? 0,
        next_billing_date: data.billing_config?.default_next_billing_date ?? "",
      });
    } catch (error: any) {
      console.error("Error fetching admin subscription data:", error);
      toast.error(error?.message || "Failed to load subscription data");
    } finally {
      setLoading(false);
    }
  };

  const saveBillingConfig = async () => {
    setSavingConfig(true);
    try {
      const { data } = await api.put<AdminBillingConfig>("/subscription/admin/config", {
        default_fee_type: configForm.fee_type,
        default_fee_value: configForm.fee_value,
        default_min_threshold: configForm.min_threshold,
        default_next_billing_date: configForm.next_billing_date || null,
      });
      setBillingConfig(data);
      toast.success("Billing config saved");
      await fetchData();
    } catch (error: any) {
      console.error("Error saving billing config:", error);
      toast.error(error?.message || "Failed to save billing config");
    } finally {
      setSavingConfig(false);
    }
  };

  const updateSubscriptionStatus = async (
    subId: string,
    status: "active" | "past_due" | "canceled"
  ) => {
    setUpdatingSubId(subId);
    try {
      await api.patch(`/subscription/admin/subscriptions/${subId}/status`, { status });
      toast.success(`Subscription updated to ${status}`);
      await fetchData();
    } catch (error: any) {
      console.error("Error updating subscription:", error);
      toast.error(error?.message || "Failed to update subscription");
    } finally {
      setUpdatingSubId(null);
    }
  };

  const updateSubscriptionNextBilling = async (sub: AdminSubscriptionItem) => {
    setSavingNextBillingSubId(sub.id);
    try {
      await api.patch(`/admin/users/${sub.user_id}/subscriptions/${sub.id}/billing`, {
        fee_type: sub.fee_type,
        fee_value: sub.fee_value,
        min_profit_threshold: sub.min_profit_threshold,
        next_billing_date: nextBillingDrafts[sub.id] || null,
      });
      toast.success("Next billing updated");
      await fetchData();
    } catch (error: any) {
      console.error("Error updating next billing date:", error);
      toast.error(error?.message || "Failed to update next billing date");
    } finally {
      setSavingNextBillingSubId(null);
    }
  };

  const getStatusBadge = (status: AdminSubscriptionItem["status"]) => {
    switch (status) {
      case "active":
        return <Badge className="bg-success/10 text-success">Active</Badge>;
      case "past_due":
        return <Badge className="bg-warning/10 text-warning">Past Due</Badge>;
      case "canceled":
        return <Badge className="bg-destructive/10 text-destructive">Canceled</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
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
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">Default Billing Configuration</h3>
        </div>

        <div className="grid md:grid-cols-4 gap-6">
          <div>
            <Label className="mb-3 block">Fee Type</Label>
            <RadioGroup
              value={configForm.fee_type}
              onValueChange={(value) => setConfigForm({ ...configForm, fee_type: value as "percentage" | "fixed" })}
              className="flex gap-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="percentage" id="percentage" />
                <Label htmlFor="percentage" className="flex items-center gap-1">
                  <Percent className="w-4 h-4" />
                  Percentage
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="fixed" id="fixed" />
                <Label htmlFor="fixed" className="flex items-center gap-1">
                  <DollarSign className="w-4 h-4" />
                  Fixed
                </Label>
              </div>
            </RadioGroup>
          </div>

          <div>
            <Label htmlFor="fee_value">
              Fee Value ({configForm.fee_type === "percentage" ? "%" : "$"})
            </Label>
            <Input
              id="fee_value"
              type="number"
              min={0}
              value={configForm.fee_value}
              onChange={(event) =>
                setConfigForm({ ...configForm, fee_value: parseFloat(event.target.value) || 0 })
              }
              className="mt-2"
            />
          </div>

          <div>
            <Label htmlFor="min_threshold">Minimum Profit Threshold ($)</Label>
            <Input
              id="min_threshold"
              type="number"
              min={0}
              value={configForm.min_threshold}
              onChange={(event) =>
                setConfigForm({ ...configForm, min_threshold: parseFloat(event.target.value) || 0 })
              }
              className="mt-2"
            />
          </div>

          <div>
            <Label htmlFor="default_next_billing_date">Default Next Billing Date</Label>
            <Input
              id="default_next_billing_date"
              type="date"
              value={configForm.next_billing_date}
              onChange={(event) =>
                setConfigForm({ ...configForm, next_billing_date: event.target.value })
              }
              className="mt-2"
            />
          </div>
        </div>

        <div className="mt-4 text-xs text-muted-foreground">
          {billingConfig?.updated_at
            ? `Last updated: ${new Date(billingConfig.updated_at).toLocaleString()}`
            : "Using system defaults"}
        </div>

        <Button onClick={saveBillingConfig} disabled={savingConfig} className="mt-4">
          <Save className="w-4 h-4 mr-2" />
          {savingConfig ? "Saving..." : "Save Configuration"}
        </Button>
      </div>

      <div className="glass-card overflow-hidden">
        <div className="p-4 border-b border-border">
          <h3 className="font-semibold">User Subscriptions</h3>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Fee</TableHead>
              <TableHead>Threshold</TableHead>
              <TableHead>Next Billing</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {subscriptions.map((sub) => (
              <TableRow key={sub.id}>
                <TableCell>{sub.user_email || "Unknown"}</TableCell>
                <TableCell>{getStatusBadge(sub.status)}</TableCell>
                <TableCell>
                  {sub.fee_type === "percentage" ? `${sub.fee_value}%` : `$${sub.fee_value}`}
                </TableCell>
                <TableCell>${sub.min_profit_threshold}</TableCell>
                <TableCell>
                  <Input
                    type="date"
                    value={nextBillingDrafts[sub.id] ?? ""}
                    onChange={(event) =>
                      setNextBillingDrafts((prev) => ({ ...prev, [sub.id]: event.target.value }))
                    }
                    className="h-8 w-[155px]"
                  />
                </TableCell>
                <TableCell>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      disabled={savingNextBillingSubId === sub.id}
                      onClick={() => updateSubscriptionNextBilling(sub)}
                    >
                      {savingNextBillingSubId === sub.id ? "Saving..." : "Save Billing Date"}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={updatingSubId === sub.id || sub.status === "active"}
                      onClick={() => updateSubscriptionStatus(sub.id, "active")}
                    >
                      Activate
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={updatingSubId === sub.id || sub.status === "past_due"}
                      onClick={() => updateSubscriptionStatus(sub.id, "past_due")}
                    >
                      Mark Past Due
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={updatingSubId === sub.id || sub.status === "canceled"}
                      onClick={() => updateSubscriptionStatus(sub.id, "canceled")}
                    >
                      Cancel
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
