export type PersistedBotAction = {
  title: string;
  detail: string;
  botId?: string;
  kind?:
    | "starting"
    | "stopping"
    | "deleting"
    | "emergency"
    | "change_model"
    | "apply_update"
    | "update_risk"
    | "update_schedule";
  expectedStatus?: "running" | "stopped" | "deleted";
  startedAt: number;
};

export type PersistedBotLog = {
  id: string;
  level: "info" | "success" | "error";
  message: string;
  at: string;
  botId?: string;
  action?: string;
  phase?: string;
  ts?: number;
};

export type PersistedBotUiState = {
  activeAction: PersistedBotAction | null;
  logs: PersistedBotLog[];
};

export const BOT_UI_STORAGE_KEY = "smarf_bot_ui_ops_v1";

const EMPTY_STATE: PersistedBotUiState = {
  activeAction: null,
  logs: [],
};

export const readBotUiState = (): PersistedBotUiState => {
  if (typeof window === "undefined") return EMPTY_STATE;
  try {
    const raw = window.localStorage.getItem(BOT_UI_STORAGE_KEY);
    if (!raw) return EMPTY_STATE;
    const parsed = JSON.parse(raw);
    return {
      activeAction: parsed?.activeAction || null,
      logs: Array.isArray(parsed?.logs) ? parsed.logs : [],
    };
  } catch {
    return EMPTY_STATE;
  }
};

export const saveBotUiState = (state: PersistedBotUiState) => {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(BOT_UI_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // ignore storage write failures
  }
};

export const saveActiveBotAction = (activeAction: PersistedBotAction | null) => {
  const current = readBotUiState();
  saveBotUiState({
    ...current,
    activeAction,
  });
};
