import { useState } from "react";
import { ChevronDown, Server, Plus, Check, Pencil, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { AddAccountDialog } from "@/components/dialogs/AddAccountDialog";
import { EditAccountDialog } from "@/components/dialogs/EditAccountDialog";
import { useAuth } from "@/contexts/AuthContext";
import { AccountWithBots, Update_Trading_Account } from "@/hooks/useTradingAccounts";
import { Skeleton } from "@/components/ui/skeleton";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface AccountSelectorProps {
  selectedAccount: AccountWithBots | null;
  onAccountChange: (account: AccountWithBots) => void;
  accounts: AccountWithBots[];
  isLoading: boolean;
  onRefresh: () => void;
  onUpdateAccount: (accountId: string, data: Update_Trading_Account) => Promise<{ success: boolean; error?: string }>;
  onDeleteAccount: (accountId: string) => Promise<{ success: boolean; error?: string }>;
  onAccountDeleted?: (accountId: string) => void;
}

export function AccountSelector({
  selectedAccount,
  onAccountChange,
  accounts,
  isLoading,
  onRefresh,
  onUpdateAccount,
  onDeleteAccount,
  onAccountDeleted,
}: AccountSelectorProps) {
  const { user } = useAuth();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const addAccountDisabled = Boolean(user?.subscription_blocked);
  const addAccountDisabledReason =
    user?.subscription_block_message || "Add a payment method before connecting trading accounts.";

  const getRunningBotsCount = (account: AccountWithBots) => {
    return account.bot_configurations.filter(
      (b) => b.container_status === "running" || b.container_status === "starting"
    ).length;
  };

  const getStatusColor = (account: AccountWithBots) => {
    const runningCount = getRunningBotsCount(account);
    if (runningCount > 0) return "bg-success";
    if (account.bot_configurations.length > 0) return "bg-warning";
    return "bg-muted-foreground";
  };

  const getBotsInfo = (account: AccountWithBots) => {
    const total = account.bot_configurations.length;
    const running = getRunningBotsCount(account);
    if (total === 0) return "No bots";
    if (running > 0) return `${running}/${total} Running`;
    return `${total} Bots`;
  };

  const getAccountPnl = (account: AccountWithBots) =>
    Number(account.total_net_pnl ?? account.total_today_pnl ?? 0);

  const handleDeleteSelectedAccount = async () => {
    if (!selectedAccount) return;
    setDeleteLoading(true);
    try {
      const accountId = selectedAccount.id;
      const result = await onDeleteAccount(accountId);
      if (result.success) {
        onAccountDeleted?.(accountId);
        setDeleteConfirmOpen(false);
      }
    } finally {
      setDeleteLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-card border border-border w-full md:w-auto min-w-[320px]">
        <Skeleton className="w-10 h-10 rounded-lg" />
        <div className="flex-1">
          <Skeleton className="h-4 w-24 mb-1" />
          <Skeleton className="h-3 w-32" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="flex items-center gap-3 px-4 py-3 rounded-lg bg-card border border-border hover:border-primary/50 transition-all w-full md:w-auto min-w-[320px]">
            <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center">
              <Server className="w-5 h-5 text-muted-foreground" />
            </div>
            <div className="flex-1 text-left">
              {selectedAccount ? (
                <>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium">{selectedAccount.broker_name}</p>
                    <span className={cn("w-2 h-2 rounded-full", getStatusColor(selectedAccount))} />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {selectedAccount.server_name} • {getBotsInfo(selectedAccount)}
                  </p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {accounts.length > 0 ? "Select Account" : "No accounts yet"}
                </p>
              )}
            </div>
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-[320px]">
          {accounts.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No trading accounts found. Add one to get started.
            </div>
          ) : (
            accounts.map((account) => (
              <DropdownMenuItem
                key={account.id}
                onClick={() => onAccountChange(account)}
                className="flex items-center gap-3 p-3 cursor-pointer"
              >
                <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
                  <Server className="w-4 h-4 text-muted-foreground" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium">{account.broker_name}</p>
                    <span className={cn("w-1.5 h-1.5 rounded-full", getStatusColor(account))} />
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>{account.server_name}</span>
                    <span>•</span>
                    <span>{getBotsInfo(account)}</span>
                    <span>•</span>
                    <span className={cn(getAccountPnl(account) >= 0 ? "text-success" : "text-destructive")}>
                      {getAccountPnl(account) >= 0 ? "+" : ""}${getAccountPnl(account).toFixed(2)}
                    </span>
                  </div>
                </div>
                {selectedAccount?.id === account.id && (
                  <Check className="w-4 h-4 text-primary" />
                )}
              </DropdownMenuItem>
            ))
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => {
              if (!addAccountDisabled) {
                setDialogOpen(true);
              }
            }}
            className="flex items-center gap-3 p-3 cursor-pointer text-primary"
            disabled={addAccountDisabled}
            title={addAccountDisabled ? addAccountDisabledReason : undefined}
          >
            <Plus className="w-4 h-4" />
            <span className="text-sm font-medium">Add New Account</span>
          </DropdownMenuItem>
          {selectedAccount && (
            <>
              <DropdownMenuItem
                onClick={() => setEditDialogOpen(true)}
                className="flex items-center gap-3 p-3 cursor-pointer"
              >
                <Pencil className="w-4 h-4" />
                <span className="text-sm font-medium">Edit Selected Account</span>
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => setDeleteConfirmOpen(true)}
                className="flex items-center gap-3 p-3 cursor-pointer text-destructive focus:text-destructive"
              >
                <Trash2 className="w-4 h-4" />
                <span className="text-sm font-medium">Delete Selected Account</span>
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <AddAccountDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onAccountAdded={() => {
          onRefresh();
          setDialogOpen(false);
        }}
      />
      <EditAccountDialog
        open={editDialogOpen}
        onOpenChange={setEditDialogOpen}
        account={selectedAccount}
        onSave={onUpdateAccount}
        onAccountUpdated={() => {
          onRefresh();
          setEditDialogOpen(false);
        }}
      />
      <AlertDialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete trading account?</AlertDialogTitle>
            <AlertDialogDescription>
              This will delete this account and all linked bots permanently.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleteLoading}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                void handleDeleteSelectedAccount();
              }}
              disabled={deleteLoading}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleteLoading ? "Deleting..." : "Delete Account"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export type { AccountWithBots };
