import logging

from prisma import Json

from ..database.client import db
from .ws_manager import bot_hub

logger = logging.getLogger(__name__)
_BOT_LOG_MODEL_MISSING_WARNED = False


def _to_text(value) -> str:
    return str(value or "").strip()


def _normalize_lower(value) -> str:
    return _to_text(value).lower()


def _resolve_level(phase: str) -> str:
    phase_name = _normalize_lower(phase)
    if phase_name == "failed":
        return "error"
    if phase_name == "succeeded":
        return "success"
    return "info"


async def emit_and_store_bot_operation_event(
    *,
    bot_config_id: str,
    action: str,
    phase: str,
    detail: str | None = None,
    status: str | None = None,
    source: str = "system",
    metadata: dict | None = None,
    owner_user_id: str | None = None,
) -> None:
    bot_id = _to_text(bot_config_id)
    action_name = _normalize_lower(action)
    phase_name = _normalize_lower(phase)
    source_name = _normalize_lower(source) or "system"
    status_name = _normalize_lower(status) if status is not None else None
    message = _to_text(detail) or None
    if not bot_id or not action_name or not phase_name:
        return

    level = _resolve_level(phase_name)
    meta_payload = dict(metadata or {})
    if "level" not in meta_payload:
        meta_payload["level"] = level

    create_data: dict = {
        "botConfiguration": {"connect": {"id": bot_id}},
        "source": source_name,
        "action": action_name,
        "phase": phase_name,
        "level": level,
        "message": message,
        "status": status_name,
    }
    owner_id = _to_text(owner_user_id)
    if owner_id:
        create_data["user"] = {"connect": {"id": owner_id}}
    if meta_payload:
        create_data["meta"] = Json(meta_payload)

    bot_operation_model = getattr(db, "botoperationlog", None)
    if bot_operation_model is None:
        global _BOT_LOG_MODEL_MISSING_WARNED
        if not _BOT_LOG_MODEL_MISSING_WARNED:
            _BOT_LOG_MODEL_MISSING_WARNED = True
            logger.warning(
                "Prisma client missing botoperationlog model. Run `prisma generate --schema=src/database/schema.prisma`."
            )
    else:
        try:
            await bot_operation_model.create(data=create_data)
        except Exception as exc:
            logger.warning(
                "failed to persist bot operation event bot=%s action=%s phase=%s: %s",
                bot_id,
                action_name,
                phase_name,
                exc,
            )

    try:
        await bot_hub.broadcast_lifecycle_event(
            bot_config_id=bot_id,
            action=action_name,
            phase=phase_name,
            detail=message,
            status=status_name,
            source=source_name,
            metadata=meta_payload,
        )
    except Exception as exc:
        logger.warning(
            "failed to broadcast bot operation event bot=%s action=%s phase=%s: %s",
            bot_id,
            action_name,
            phase_name,
            exc,
        )
