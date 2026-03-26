export type RiskLevelId = "low" | "medium" | "high";
export type BotRiskMode = "level" | "custom_lot";

export interface RiskLevelOption {
  id: RiskLevelId;
  label: string;
  description: string;
  color: string;
}

export const DEFAULT_RISK_LEVEL: RiskLevelId = "medium";
export const DEFAULT_RISK_MODE: BotRiskMode = "level";
export const DEFAULT_RISK_PROFILE_MAP: Record<RiskLevelId, number> = {
  low: 0.5,
  medium: 1.0,
  high: 1.5,
};
export const DEFAULT_RISK_PIPS = 50;
export const DEFAULT_PIP_VALUE_PER_LOT = 10;
export const MIN_LOT_SIZE = 0.01;
export const LOT_STEP = 0.01;

export const RISK_LEVEL_OPTIONS: RiskLevelOption[] = [
  { id: "low", label: "Low", description: "Conservative trading", color: "text-success" },
  { id: "medium", label: "Medium", description: "Balanced approach", color: "text-warning" },
  { id: "high", label: "High", description: "Aggressive trading", color: "text-destructive" },
];

const floorToLotStep = (value: number): number => {
  const steps = Math.floor((value + 1e-9) / LOT_STEP);
  return Math.max(MIN_LOT_SIZE, Number((steps * LOT_STEP).toFixed(2)));
};

export function normalizeRiskLevel(value: string | null | undefined): RiskLevelId {
  const text = String(value || "").trim().toLowerCase();
  if (text === "low" || text === "high") {
    return text;
  }
  return DEFAULT_RISK_LEVEL;
}

export function normalizeRiskMode(value: string | null | undefined): BotRiskMode {
  return String(value || "").trim().toLowerCase() === "custom_lot"
    ? "custom_lot"
    : DEFAULT_RISK_MODE;
}

export function normalizeCustomLot(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const lot = Number(value);
  if (!Number.isFinite(lot) || lot < MIN_LOT_SIZE) {
    return null;
  }
  return floorToLotStep(lot);
}

export function estimateBotLotSize({
  balance,
  riskLevel,
  riskMode,
  customLot,
  riskProfileMap,
}: {
  balance: number;
  riskLevel?: string | null;
  riskMode?: string | null;
  customLot?: number | string | null;
  riskProfileMap?: Partial<Record<string, number>> | null;
}): number {
  const resolvedMode = normalizeRiskMode(riskMode);
  const resolvedCustomLot = normalizeCustomLot(customLot);
  if (resolvedMode === "custom_lot" && resolvedCustomLot !== null) {
    return resolvedCustomLot;
  }

  const profile = {
    ...DEFAULT_RISK_PROFILE_MAP,
    ...(riskProfileMap || {}),
  } as Record<string, number>;
  const resolvedLevel = normalizeRiskLevel(riskLevel);
  const pct = Number(profile[resolvedLevel] ?? DEFAULT_RISK_PROFILE_MAP[DEFAULT_RISK_LEVEL]);
  const safeBalance = Number.isFinite(Number(balance)) ? Number(balance) : 0;
  const riskAmount = safeBalance * pct / 100;
  const rawLot = riskAmount / (DEFAULT_RISK_PIPS * DEFAULT_PIP_VALUE_PER_LOT);
  return floorToLotStep(rawLot);
}

export function formatLotSize(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return "—";
  }
  return `${value.toFixed(2)} lot`;
}

export function getRiskColorClass(riskLevel: string | null | undefined, riskMode?: string | null): string {
  if (normalizeRiskMode(riskMode) === "custom_lot") {
    return "text-primary";
  }
  const level = normalizeRiskLevel(riskLevel);
  if (level === "low") return "text-success";
  if (level === "high") return "text-destructive";
  return "text-warning";
}

export function getRiskLabel(riskLevel: string | null | undefined): string {
  return RISK_LEVEL_OPTIONS.find((option) => option.id === normalizeRiskLevel(riskLevel))?.label || "Medium";
}

export function getRiskSummary({
  riskLevel,
  riskMode,
  customLot,
  lotSize,
}: {
  riskLevel?: string | null;
  riskMode?: string | null;
  customLot?: number | string | null;
  lotSize?: number | null;
}): string {
  const resolvedMode = normalizeRiskMode(riskMode);
  const resolvedCustomLot = normalizeCustomLot(customLot);
  if (resolvedMode === "custom_lot" && resolvedCustomLot !== null) {
    return `Custom • ${formatLotSize(resolvedCustomLot)}`;
  }
  return `${getRiskLabel(riskLevel)} • ${formatLotSize(lotSize)}`;
}
