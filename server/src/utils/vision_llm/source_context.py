from dataclasses import dataclass

from ..ws_manager import bot_hub


@dataclass(frozen=True)
class VisionSourceContext:
    cache_scope: str
    display_label: str
    bot_config_id: str
    server_name: str
    login_id: str


def _safe_token(raw: object, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return fallback
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_.-")
    return token or fallback


def resolve_vision_source_context(bot_config_id: str | None = None) -> VisionSourceContext:
    bot_id = str(bot_config_id or "").strip()
    if not bot_id:
        return VisionSourceContext(
            cache_scope="shared",
            display_label="shared server MT5 source",
            bot_config_id="",
            server_name="",
            login_id="",
        )

    conn = bot_hub.get_bot(bot_id)
    last_state = dict(getattr(conn, "last_state", {}) or {}) if conn is not None else {}
    server_name = str(last_state.get("server", "") or "").strip()
    login_id = str(last_state.get("login", "") or "").strip()

    scope_parts = [
        "bot",
        _safe_token(bot_id, "unknown"),
        _safe_token(server_name, "noserver"),
        _safe_token(login_id, "nologin"),
    ]
    label_parts = []
    if server_name:
        label_parts.append(server_name)
    if login_id:
        label_parts.append(f"login {login_id}")
    label_parts.append(f"bot {bot_id[:8]}")

    return VisionSourceContext(
        cache_scope=":".join(scope_parts),
        display_label=" | ".join(label_parts),
        bot_config_id=bot_id,
        server_name=server_name,
        login_id=login_id,
    )


def build_vision_cache_key(symbol: str, timeframe: str, dt_str: str, source_context: VisionSourceContext) -> str:
    clean_symbol = str(symbol or "").strip().upper() or "UNKNOWN"
    clean_timeframe = str(timeframe or "H1").strip().upper() or "H1"
    clean_dt = str(dt_str or "").strip()
    scope = str(getattr(source_context, "cache_scope", "") or "shared").strip() or "shared"
    return f"vision_llm:{clean_symbol}:{clean_timeframe}:{clean_dt}:source:{scope}"
