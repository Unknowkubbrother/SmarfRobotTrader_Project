import { useEffect, useMemo, useState } from "react";
import { Lock, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AccountWithBots, Update_Trading_Account } from "@/hooks/useTradingAccounts";
import { toast } from "sonner";
import { useMt5ServerCatalog } from "@/hooks/useMt5ServerCatalog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface EditAccountDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  account: AccountWithBots | null;
  onSave: (accountId: string, data: Update_Trading_Account) => Promise<{ success: boolean; error?: string }>;
  onAccountUpdated?: () => void;
}

export function EditAccountDialog({ open, onOpenChange, account, onSave, onAccountUpdated }: EditAccountDialogProps) {
  const [loading, setLoading] = useState(false);
  const { catalog, loading: catalogLoading } = useMt5ServerCatalog();
  const [formData, setFormData] = useState({
    broker_name: "",
    server_name: "",
    mt5_login_id: "",
    mt5_password: "",
  });
  useEffect(() => {
    if (!open || !account) return;
    setFormData({
      broker_name: account.broker_name || "",
      server_name: account.server_name || "",
      mt5_login_id: account.mt5_login_id || "",
      mt5_password: "",
    });
  }, [open, account]);

  const brokerOptions = useMemo(() => {
    const names = catalog.brokers.map((entry) => entry.broker_name);
    if (formData.broker_name && !names.includes(formData.broker_name)) {
      return [formData.broker_name, ...names];
    }
    return names;
  }, [catalog.brokers, formData.broker_name]);

  const serverOptions = useMemo(() => {
    const matched = catalog.brokers.find((entry) => entry.broker_name === formData.broker_name);
    const names = matched?.server_names ? [...matched.server_names] : [];
    if (formData.server_name && !names.includes(formData.server_name)) {
      return [formData.server_name, ...names];
    }
    return names;
  }, [catalog.brokers, formData.broker_name, formData.server_name]);

  const handleSubmit = async () => {
    if (!account) return;
    if (!formData.broker_name || !formData.server_name || !formData.mt5_login_id) {
      toast.error("Please fill in broker, server and login fields");
      return;
    }

    setLoading(true);
    try {
      const result = await onSave(account.id, {
        brokerName: formData.broker_name,
        serverName: formData.server_name,
        mt5LoginId: formData.mt5_login_id,
        ...(formData.mt5_password.trim()
          ? { mt5Password: formData.mt5_password.trim() }
          : {}),
      });
      if (!result.success) throw new Error(result.error);
      onOpenChange(false);
      onAccountUpdated?.();
    } catch (error) {
      console.error("Error updating account:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Trading Account</DialogTitle>
          <DialogDescription>
            Updating account will stop all linked bots for safety.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div>
            <Label htmlFor="edit-broker">Broker Name</Label>
            <Select
              value={formData.broker_name}
              onValueChange={(nextBroker) =>
                setFormData((prev) => ({
                  ...prev,
                  broker_name: nextBroker,
                  server_name: nextBroker === prev.broker_name ? prev.server_name : "",
                }))
              }
              disabled={catalogLoading || brokerOptions.length === 0}
            >
              <SelectTrigger id="edit-broker" className="mt-1">
                <SelectValue placeholder={catalogLoading ? "Loading broker list..." : "Select broker"} />
              </SelectTrigger>
              <SelectContent>
                {brokerOptions.map((brokerName) => (
                  <SelectItem key={brokerName} value={brokerName}>
                    {brokerName}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="edit-server">Server Name</Label>
            <Select
              value={formData.server_name}
              onValueChange={(nextServer) =>
                setFormData((prev) => ({
                  ...prev,
                  server_name: nextServer,
                }))
              }
              disabled={!formData.broker_name || serverOptions.length === 0}
            >
              <SelectTrigger id="edit-server" className="mt-1">
                <SelectValue placeholder={!formData.broker_name ? "Select broker first" : "Select server"} />
              </SelectTrigger>
              <SelectContent>
                {serverOptions.map((serverName) => (
                  <SelectItem key={serverName} value={serverName}>
                    {serverName}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="edit-login">MT5 Login ID</Label>
            <div className="relative mt-1">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                id="edit-login"
                className="pl-10"
                value={formData.mt5_login_id}
                onChange={(e) => setFormData((prev) => ({ ...prev, mt5_login_id: e.target.value }))}
              />
            </div>
          </div>
          <div>
            <Label htmlFor="edit-password">MT5 Password (optional)</Label>
            <div className="relative mt-1">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                id="edit-password"
                type="password"
                className="pl-10"
                placeholder="Leave blank to keep current password"
                value={formData.mt5_password}
                onChange={(e) => setFormData((prev) => ({ ...prev, mt5_password: e.target.value }))}
              />
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? "Saving..." : "Save Changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
