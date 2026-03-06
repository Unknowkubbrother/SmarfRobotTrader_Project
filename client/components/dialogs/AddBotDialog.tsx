import { useState, useEffect } from "react";
import { Activity, AlertTriangle, Shield, Sparkles } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import type { BotConfigWithVersion } from "@/hooks/useTradingAccounts";

interface BotVersion {
  id: string;
  model_id: string;
  label: string | null;
  version_tag: string | null;
  symbol: string | null;
  timeframe: string | null;
  release_notes: string[];
  release_date: string | null;
}

interface AddBotDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  accountId: string;
  existingBots?: BotConfigWithVersion[];
  onBotAdded: () => void;
  blockedReason?: string | null;
}

const riskLevels = [
  { id: "low", label: "Low", description: "Conservative trading", color: "text-success" },
  { id: "medium", label: "Medium", description: "Balanced approach", color: "text-warning" },
  { id: "high", label: "High", description: "Aggressive trading", color: "text-destructive" },
];

const normalizePairValue = (value: string | null | undefined): string =>
  String(value || "").trim().toUpperCase();

const getPairKey = (symbol: string | null | undefined, timeframe: string | null | undefined): string | null => {
  const normalizedSymbol = normalizePairValue(symbol);
  const normalizedTimeframe = normalizePairValue(timeframe);
  if (!normalizedSymbol || !normalizedTimeframe) {
    return null;
  }
  return `${normalizedSymbol}__${normalizedTimeframe}`;
};

export function AddBotDialog({
  open,
  onOpenChange,
  accountId,
  existingBots = [],
  onBotAdded,
  blockedReason = null,
}: AddBotDialogProps) {
  const { user } = useAuth();
  const [botVersions, setBotVersions] = useState<BotVersion[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectedRisk, setSelectedRisk] = useState("medium");
  const [loading, setLoading] = useState(false);
  const subscriptionBlocked = Boolean(blockedReason ?? user?.subscription_blocked);
  const subscriptionBlockedReason =
    blockedReason ??
    user?.subscription_block_message ??
    "Complete billing setup before creating new bots.";

  const existingPairKeys = new Set(
    existingBots
      .map((bot) => getPairKey(bot.bot_version?.symbol, bot.bot_version?.timeframe))
      .filter((value): value is string => Boolean(value))
  );

  const isModelAlreadyAdded = (model: BotVersion) => {
    const pairKey = getPairKey(model.symbol, model.timeframe);
    return Boolean(pairKey && existingPairKeys.has(pairKey));
  };

  useEffect(() => {
    if (open) {
      fetchBotVersions();
    }
  }, [open]);

  useEffect(() => {
    if (!open) {
      setSelectedModel(null);
      setSelectedRisk("medium");
      return;
    }

    setSelectedModel((current) => {
      if (!current) return current;
      const selectedVersion = botVersions.find((model) => model.id === current);
      if (!selectedVersion) return null;
      const pairKey = getPairKey(selectedVersion.symbol, selectedVersion.timeframe);
      if (!pairKey) return current;
      const pairAlreadyExists = existingBots.some((bot) => {
        const existingPairKey = getPairKey(bot.bot_version?.symbol, bot.bot_version?.timeframe);
        return existingPairKey === pairKey;
      });
      return pairAlreadyExists ? null : current;
    });
  }, [open, botVersions, existingBots]);

  const fetchBotVersions = async () => {
    try {
      const { data } = await api.get("/bot/versions");
      setBotVersions(data.data || []);
    } catch (error) {
      console.error("Error fetching bot versions:", error);
    }
  };

  const handleCreateBot = async () => {
    if (subscriptionBlocked) {
      toast.error(subscriptionBlockedReason);
      return;
    }

    if (!selectedModel) {
      toast.error("Please select a model");
      return;
    }

    const selectedVersion = botVersions.find((model) => model.id === selectedModel);
    if (selectedVersion && isModelAlreadyAdded(selectedVersion)) {
      toast.error("This symbol and timeframe is already added in this account");
      return;
    }

    setLoading(true);
    try {
      await api.post("/bot/create_bot_configuration", {
        accountId,
        modelId: selectedModel,
        riskLevel: selectedRisk,
      });

      toast.success("Bot added successfully");
      onOpenChange(false);
      onBotAdded();
      setSelectedModel(null);
      setSelectedRisk("medium");
    } catch (error: any) {
      console.error("Error creating bot:", error);
      toast.error(error.message || "Failed to add bot");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-primary" />
            Add New Bot
          </DialogTitle>
          <DialogDescription>
            Select a model and configure your new trading bot
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {subscriptionBlocked && (
            <Alert className="border-warning/30 bg-warning/5 text-foreground [&>svg]:text-warning">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Bot creation is blocked</AlertTitle>
              <AlertDescription>{subscriptionBlockedReason}</AlertDescription>
            </Alert>
          )}

          {/* Model Selection */}
          <div>
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Select Model
            </h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {botVersions.map((model) => {
                const alreadyAdded = isModelAlreadyAdded(model);
                return (
                  <div
                    key={model.id}
                    onClick={() => {
                      if (alreadyAdded || subscriptionBlocked) return;
                      setSelectedModel(model.id);
                    }}
                    className={cn(
                      "rounded-xl border overflow-hidden cursor-pointer transition-all",
                      (alreadyAdded || subscriptionBlocked) && "cursor-not-allowed border-dashed opacity-60",
                      selectedModel === model.id
                        ? "border-primary ring-2 ring-primary/20"
                        : "border-border hover:border-primary/50"
                    )}
                  >
                    <div className="bg-gradient-to-r from-[#0f3460] to-[#16537e] p-3 text-white">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Activity className="w-4 h-4" />
                          <span className="font-medium text-sm">{model.label}</span>
                        </div>
                        <span className="px-2 py-0.5 rounded-full bg-white/20 text-xs">
                          {model.version_tag}
                        </span>
                      </div>
                    </div>
                    <div className="p-3 bg-card space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Lastest update</span>
                        <span className="font-mono">{model.release_date}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Symbol</span>
                        <span className="font-mono">{model.symbol}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">Timeframe</span>
                        <span>{model.timeframe}</span>
                      </div>
                      <div className="flex flex-col gap-1.5 text-xs pt-2 border-t border-border/50">
                        <span className="text-muted-foreground font-medium">Release Notes</span>
                        <div className="max-h-20 overflow-y-auto flex flex-col gap-1 pr-1 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
                          {model.release_notes.map((note, index) => (
                            <div key={index} className="flex items-start gap-1.5">
                              <span className="text-primary mt-1 shrink-0">•</span>
                              <span className="text-muted-foreground/80">{note}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      {alreadyAdded ? (
                        <div className="pt-2">
                          <span className="text-xs text-muted-foreground font-medium">Already added to this account</span>
                        </div>
                      ) : subscriptionBlocked ? (
                        <div className="pt-2">
                          <span className="text-xs text-warning font-medium">Resolve billing before adding new bots</span>
                        </div>
                      ) : selectedModel === model.id ? (
                        <div className="pt-2">
                          <span className="text-xs text-primary font-medium">✓ Selected</span>
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Risk Level Selection */}
          <div>
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Shield className="w-4 h-4" />
              Risk Level
            </h3>
            <div className="grid grid-cols-3 gap-3">
              {riskLevels.map((level) => (
                <button
                  key={level.id}
                  onClick={() => setSelectedRisk(level.id)}
                  disabled={subscriptionBlocked}
                  className={cn(
                    "p-3 rounded-xl border text-left transition-all",
                    selectedRisk === level.id
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  )}
                >
                  <span className={cn("font-medium text-sm", level.color)}>
                    {level.label}
                  </span>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {level.description}
                  </p>
                </button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateBot} disabled={loading || !selectedModel || subscriptionBlocked}>
              {loading ? "Adding..." : "Add Bot"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
