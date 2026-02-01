import { useState, useEffect } from "react";
import { DollarSign, Percent, Settings, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { supabase } from "@/lib/integrations/supabase/client";
import { toast } from "sonner";

interface Subscription {
  id: string;
  user_id: string;
  status: string | null;
  fee_type: string | null;
  fee_value: number | null;
  next_billing_date: string | null;
  profiles: {
    email: string;
  } | null;
}

interface BillingConfig {
  id: string;
  default_fee_type: string | null;
  default_fee_value: number | null;
  default_min_threshold: number | null;
}

export function AdminSubscriptions() {
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [billingConfig, setBillingConfig] = useState<BillingConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingConfig, setSavingConfig] = useState(false);
  const [configForm, setConfigForm] = useState({
    fee_type: "percentage",
    fee_value: 20,
    min_threshold: 0,
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      // Fetch subscriptions
      const { data: subs, error: subsError } = await supabase
        .from("subscriptions")
        .select("*")
        .order("created_at", { ascending: false });

      if (subsError) throw subsError;

      // Fetch profiles for each subscription
      const subsWithProfiles = await Promise.all(
        (subs || []).map(async (sub) => {
          const { data: profile } = await supabase
            .from("profiles")
            .select("email")
            .eq("id", sub.user_id)
            .maybeSingle();
          return { ...sub, profiles: profile };
        })
      );
      setSubscriptions(subsWithProfiles);

      // Fetch billing config
      const { data: config, error: configError } = await supabase
        .from("system_billing_config")
        .select("*")
        .limit(1)
        .maybeSingle();

      if (configError && !configError.message.includes("No rows")) throw configError;

      if (config) {
        setBillingConfig(config);
        setConfigForm({
          fee_type: config.default_fee_type ?? "percentage",
          fee_value: config.default_fee_value ?? 20,
          min_threshold: config.default_min_threshold ?? 0,
        });
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      toast.error("Failed to load subscription data");
    } finally {
      setLoading(false);
    }
  };

  const saveBillingConfig = async () => {
    setSavingConfig(true);
    try {
      if (billingConfig) {
        const { error } = await supabase
          .from("system_billing_config")
          .update({
            default_fee_type: configForm.fee_type as "percentage" | "fixed",
            default_fee_value: configForm.fee_value,
            default_min_threshold: configForm.min_threshold,
          })
          .eq("id", billingConfig.id);

        if (error) throw error;
      } else {
        const { error } = await supabase.from("system_billing_config").insert({
          default_fee_type: configForm.fee_type as "percentage" | "fixed",
          default_fee_value: configForm.fee_value,
          default_min_threshold: configForm.min_threshold,
        });

        if (error) throw error;
      }
      toast.success("Billing config saved");
      fetchData();
    } catch (error) {
      console.error("Error saving config:", error);
      toast.error("Failed to save billing config");
    } finally {
      setSavingConfig(false);
    }
  };

  const updateSubscriptionStatus = async (subId: string, status: string) => {
    try {
      const { error } = await supabase
        .from("subscriptions")
        .update({ status: status as "active" | "trial" | "suspended" | "expired" })
        .eq("id", subId);

      if (error) throw error;
      toast.success(`Subscription ${status}`);
      fetchData();
    } catch (error) {
      console.error("Error updating subscription:", error);
      toast.error("Failed to update subscription");
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active":
        return <Badge className="bg-success/10 text-success">Active</Badge>;
      case "trial":
        return <Badge className="bg-primary/10 text-primary">Trial</Badge>;
      case "suspended":
        return <Badge className="bg-destructive/10 text-destructive">Suspended</Badge>;
      case "expired":
        return <Badge variant="secondary">Expired</Badge>;
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
      {/* Billing Configuration */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">Default Billing Configuration</h3>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <Label className="mb-3 block">Fee Type</Label>
            <RadioGroup
              value={configForm.fee_type}
              onValueChange={(value) => setConfigForm({ ...configForm, fee_type: value })}
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
              value={configForm.fee_value}
              onChange={(e) => setConfigForm({ ...configForm, fee_value: parseFloat(e.target.value) || 0 })}
              className="mt-2"
            />
          </div>

          <div>
            <Label htmlFor="min_threshold">Minimum Profit Threshold ($)</Label>
            <Input
              id="min_threshold"
              type="number"
              value={configForm.min_threshold}
              onChange={(e) => setConfigForm({ ...configForm, min_threshold: parseFloat(e.target.value) || 0 })}
              className="mt-2"
            />
          </div>
        </div>

        <Button onClick={saveBillingConfig} disabled={savingConfig} className="mt-4">
          <Save className="w-4 h-4 mr-2" />
          {savingConfig ? "Saving..." : "Save Configuration"}
        </Button>
      </div>

      {/* Subscriptions List */}
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
              <TableHead>Next Billing</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {subscriptions.map((sub) => (
              <TableRow key={sub.id}>
                <TableCell>{sub.profiles?.email || "Unknown"}</TableCell>
                <TableCell>{getStatusBadge(sub.status ?? 'unknown')}</TableCell>
                <TableCell>
                  {sub.fee_type === "percentage" ? `${sub.fee_value ?? 0}%` : `$${sub.fee_value ?? 0}`}
                </TableCell>
                <TableCell>
                  {sub.next_billing_date
                    ? new Date(sub.next_billing_date).toLocaleDateString()
                    : "-"}
                </TableCell>
                <TableCell>
                  <div className="flex gap-2">
                    {sub.status === "suspended" ? (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => updateSubscriptionStatus(sub.id, "active")}
                      >
                        Activate
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => updateSubscriptionStatus(sub.id, "suspended")}
                      >
                        Suspend
                      </Button>
                    )}
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
