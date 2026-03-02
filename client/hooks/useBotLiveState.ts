"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export interface MT5Position {
    ticket: number;
    symbol: string;
    type: "BUY" | "SELL";
    volume: number;
    price_open: number;
    price_current: number;
    profit: number;
    swap: number;
    sl: number;
    tp: number;
    time: string;
    comment: string;
}

export interface BotLiveLogEntry {
    timestamp: string;
    type: "info" | "analysis" | "action" | "warning" | "success";
    message: string;
    phase?: string;
    event?: string;
    severity?: "info" | "warning" | "success" | "error";
    meta?: Record<string, unknown>;
}

export interface BotLiveState {
    bot_config_id: string;
    symbol: string;
    timeframe: string;
    // Bot model state
    position: number;
    entry_price: number;
    total_pnl: number;
    trades: number;
    wins: number;
    loss_streak: number;
    last_action: string;
    last_bar_time: string;
    lot_size: number;
    unrealized_pnl: number;
    risk_level?: string;
    risk_percent?: number;
    risk_profile_map?: Record<string, number>;
    trading_schedule?: Record<string, boolean>;
    connected: boolean;
    ws_connected?: boolean;
    // MT5 Account
    balance: number;
    equity: number;
    margin: number;
    free_margin: number;
    margin_level: number;
    leverage: number;
    profit: number;
    currency: string;
    server: string;
    login: number;
    // MT5 Positions
    positions: MT5Position[];
    // Logs
    llm_text?: string;
    recent_logs?: BotLiveLogEntry[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = API_BASE_URL.replace(/^http/i, "ws") + "/bot/ws/dashboard";

const normalizeBotConfigId = (value: unknown): string => String(value ?? "").trim();

/**
 * React hook: connects to the BotHub dashboard WebSocket and
 * maintains a live map of all connected bot states.
 */
export function useBotLiveState() {
    const [botStates, setBotStates] = useState<Map<string, BotLiveState>>(new Map());
    const [isConnected, setIsConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimer = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        try {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                setIsConnected(true);
                console.log("[BotHub] Dashboard WS connected");
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);

                    if (msg.type === "snapshot") {
                        // Initial snapshot: list of all connected bots
                        const map = new Map<string, BotLiveState>();
                        for (const bot of msg.bots || []) {
                            const botId = normalizeBotConfigId(bot?.bot_config_id);
                            if (!botId) continue;
                            map.set(botId, {
                                ...bot,
                                bot_config_id: botId,
                            });
                        }
                        setBotStates(map);
                    } else if (msg.type === "bot_state") {
                        const botId = normalizeBotConfigId(msg.bot_config_id);
                        if (!botId) return;
                        // Incremental update for one bot
                        setBotStates((prev) => {
                            const next = new Map(prev);
                            next.set(botId, {
                                ...prev.get(botId),
                                ...msg,
                                bot_config_id: botId,
                                connected: true,
                            });
                            return next;
                        });
                    }
                } catch {
                    // ignore parse errors
                }
            };

            ws.onclose = () => {
                setIsConnected(false);
                setBotStates((prev) => {
                    const next = new Map<string, BotLiveState>();
                    for (const [botId, state] of prev.entries()) {
                        next.set(botId, {
                            ...state,
                            connected: false,
                        });
                    }
                    return next;
                });
                console.log("[BotHub] Dashboard WS disconnected, reconnecting in 3s");
                reconnectTimer.current = setTimeout(connect, 3000);
            };

            ws.onerror = () => {
                ws.close();
            };
        } catch {
            reconnectTimer.current = setTimeout(connect, 3000);
        }
    }, []);

    useEffect(() => {
        connect();
        return () => {
            reconnectTimer.current && clearTimeout(reconnectTimer.current);
            wsRef.current?.close();
        };
    }, [connect]);

    const getBotState = useCallback(
        (botConfigId: string | number): BotLiveState | undefined => {
            const botId = normalizeBotConfigId(botConfigId);
            if (!botId) return undefined;
            return botStates.get(botId);
        },
        [botStates]
    );

    return {
        botStates,
        isConnected,
        getBotState,
        allBotStates: Array.from(botStates.values()),
    };
}
