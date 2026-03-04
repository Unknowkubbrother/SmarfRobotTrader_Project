import { useState } from "react";
import { Lock, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { useTradingAccounts } from "@/hooks/useTradingAccounts";
import { useMt5ServerCatalog } from "@/hooks/useMt5ServerCatalog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface AddAccountDialogProps {
  onAccountAdded?: () => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  trigger?: React.ReactNode;
}

export function AddAccountDialog({ onAccountAdded, open, onOpenChange, trigger }: AddAccountDialogProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isControlled = open !== undefined;
  const isOpen = isControlled ? open : internalOpen;
  const setIsOpen = isControlled ? (onOpenChange || (() => { })) : setInternalOpen;

  const [loading, setLoading] = useState(false);
  const { user } = useAuth();
  const { createAccount } = useTradingAccounts();
  const { catalog, loading: catalogLoading } = useMt5ServerCatalog();
  const [formData, setFormData] = useState({
    broker_name: "",
    server_name: "",
    mt5_login_id: "",
    mt5_password: "",
  });

  const handleSubmit = async () => {
    if (!user) {
      toast.error("Please sign in to add an account");
      return;
    }

    if (!formData.broker_name || !formData.server_name || !formData.mt5_login_id || !formData.mt5_password) {
      toast.error("Please fill in all fields");
      return;
    }

    setLoading(true);
    try {
      const result = await createAccount({
        brokerName: formData.broker_name,
        serverName: formData.server_name,
        mt5LoginId: formData.mt5_login_id,
        mt5Password: formData.mt5_password,
      });

      if (!result.success) throw new Error(result.error);

      setIsOpen(false);
      setFormData({ broker_name: "", server_name: "", mt5_login_id: "", mt5_password: "" });
      onAccountAdded?.();
    } catch (error) {
      console.error("Error adding account:", error);
    } finally {
      setLoading(false);
    }
  };

  const dialogContent = (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Connect MT5 Account</DialogTitle>
        <DialogDescription>
          Enter your MetaTrader 5 account details to connect
        </DialogDescription>
      </DialogHeader>
      <div className="space-y-4 py-4">
        <div>
          <Label htmlFor="broker">Broker Name</Label>
          <Select
            value={formData.broker_name}
            onValueChange={(nextBroker) =>
              setFormData((prev) => ({
                ...prev,
                broker_name: nextBroker,
                server_name: "",
              }))
            }
            disabled={catalogLoading || catalog.brokers.length === 0}
          >
            <SelectTrigger id="broker" className="mt-1">
              <SelectValue
                placeholder={
                  catalogLoading
                    ? "Loading broker list..."
                    : "Select broker"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {catalog.brokers.map((entry) => (
                <SelectItem key={entry.broker_name} value={entry.broker_name}>
                  {entry.broker_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="server">Server Name</Label>
          <Select
            value={formData.server_name}
            onValueChange={(nextServer) =>
              setFormData((prev) => ({
                ...prev,
                server_name: nextServer,
              }))
            }
            disabled={!formData.broker_name}
          >
            <SelectTrigger id="server" className="mt-1">
              <SelectValue
                placeholder={
                  !formData.broker_name
                    ? "Select broker first"
                    : "Select server"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {(catalog.brokers.find((entry) => entry.broker_name === formData.broker_name)?.server_names || []).map((serverName) => (
                <SelectItem key={serverName} value={serverName}>
                  {serverName}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="login">MT5 Login ID</Label>
          <div className="relative mt-1">
            <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              id="login"
              placeholder="Your account number"
              className="pl-10"
              value={formData.mt5_login_id}
              onChange={(e) => setFormData({ ...formData, mt5_login_id: e.target.value })}
            />
          </div>
        </div>
        <div>
          <Label htmlFor="password">MT5 Password</Label>
          <div className="relative mt-1">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              id="password"
              type="password"
              placeholder="Your trading password"
              className="pl-10"
              value={formData.mt5_password}
              onChange={(e) => setFormData({ ...formData, mt5_password: e.target.value })}
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Your credentials are encrypted with AES-256 and transmitted securely via TLS 1.3
        </p>
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={() => setIsOpen(false)}>
          Cancel
        </Button>
        <Button onClick={handleSubmit} disabled={loading}>
          {loading ? "Connecting..." : "Connect Account"}
        </Button>
      </DialogFooter>
    </DialogContent>
  );

  if (trigger) {
    return (
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogTrigger asChild>
          {trigger}
        </DialogTrigger>
        {dialogContent}
      </Dialog>
    );
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      {dialogContent}
    </Dialog>
  );
}
