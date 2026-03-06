import json
from datetime import datetime
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..database.client import db
from ..models.support_model import (
    AdminReplySupportTicketRequest,
    CreateSupportTicketRequest,
    SupportTicketItemResponse,
    SupportTicketMessageItem,
    UserReplySupportTicketRequest,
)
from .authentication import get_current_active_user
from ..utils.notification_delivery import dispatch_notification_to_user

support_router = APIRouter(tags=["Support"])

ALLOWED_TICKET_STATUSES = {"open", "in_progress", "resolved", "closed"}


def _utc_iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _enum_value(value):
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _is_admin(current_user: Any) -> bool:
    role = _enum_value(getattr(current_user, "role", None))
    return role == "admin"


def _normalize_status(raw_status: Optional[str], default: str = "open") -> str:
    normalized = str(raw_status or default).strip().lower()
    if normalized not in ALLOWED_TICKET_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid ticket status")
    return normalized


def _normalize_message_role(raw_role: Any) -> Optional[str]:
    role = str(raw_role or "").strip().lower()
    if role in {"user", "admin"}:
        return role
    return None


def _normalize_message_entry(raw_entry: Any) -> Optional[dict]:
    if not isinstance(raw_entry, dict):
        return None
    role = _normalize_message_role(raw_entry.get("role"))
    text = str(raw_entry.get("text") or "").strip()
    if not role or not text:
        return None
    created_at = str(raw_entry.get("created_at") or "").strip() or _utc_iso_now()
    sender_name = str(raw_entry.get("sender_name") or "").strip() or None
    sender_email = str(raw_entry.get("sender_email") or "").strip() or None
    return {
        "role": role,
        "text": text,
        "created_at": created_at,
        "sender_name": sender_name,
        "sender_email": sender_email,
    }


def _extract_first_user_message(messages: list[dict]) -> str:
    for item in messages:
        if item.get("role") == "user":
            return str(item.get("text") or "").strip()
    if messages:
        return str(messages[0].get("text") or "").strip()
    return ""


def _extract_latest_admin_message(messages: list[dict]) -> Optional[str]:
    for item in reversed(messages):
        if item.get("role") == "admin":
            text = str(item.get("text") or "").strip()
            return text or None
    return None


def _extract_latest_admin_meta(messages: list[dict]) -> tuple[Optional[str], Optional[str]]:
    for item in reversed(messages):
        if item.get("role") != "admin":
            continue
        return (
            str(item.get("created_at") or "").strip() or None,
            str(item.get("sender_name") or "").strip() or None,
        )
    return (None, None)


def _parse_ticket_message(raw_message: Optional[str]) -> dict:
    text = str(raw_message or "").strip()
    if not text:
        return {"category": "Other", "messages": []}

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            category = str(payload.get("category") or "Other").strip() or "Other"
            messages: list[dict] = []

            raw_messages = payload.get("messages")
            if isinstance(raw_messages, list):
                for raw_message_item in raw_messages:
                    normalized = _normalize_message_entry(raw_message_item)
                    if normalized:
                        messages.append(normalized)

            if not messages:
                legacy_user_message = str(payload.get("user_message") or "").strip()
                legacy_admin_reply = str(payload.get("admin_reply") or "").strip()
                legacy_admin_time = str(payload.get("admin_replied_at") or "").strip() or _utc_iso_now()
                legacy_admin_name = str(payload.get("admin_replied_by") or "").strip() or None
                if legacy_user_message:
                    messages.append(
                        {
                            "role": "user",
                            "text": legacy_user_message,
                            "created_at": _utc_iso_now(),
                            "sender_name": None,
                            "sender_email": None,
                        }
                    )
                if legacy_admin_reply:
                    messages.append(
                        {
                            "role": "admin",
                            "text": legacy_admin_reply,
                            "created_at": legacy_admin_time,
                            "sender_name": legacy_admin_name,
                            "sender_email": None,
                        }
                    )

            return {"category": category, "messages": messages}
    except Exception:
        pass

    return {
        "category": "Other",
        "messages": [
            {
                "role": "user",
                "text": text,
                "created_at": _utc_iso_now(),
                "sender_name": None,
                "sender_email": None,
            }
        ],
    }


def _build_ticket_message_payload(*, category: Optional[str], messages: list[dict]) -> str:
    normalized_messages: list[dict] = []
    for item in messages:
        normalized = _normalize_message_entry(item)
        if normalized:
            normalized_messages.append(normalized)

    admin_replied_at, admin_replied_by = _extract_latest_admin_meta(normalized_messages)
    payload = {
        "category": str(category or "Other").strip() or "Other",
        "messages": normalized_messages,
        "user_message": _extract_first_user_message(normalized_messages),
        "admin_reply": _extract_latest_admin_message(normalized_messages),
        "admin_replied_at": admin_replied_at,
        "admin_replied_by": admin_replied_by,
    }
    return json.dumps(payload, ensure_ascii=False)


def _append_message(
    *,
    parsed_payload: dict,
    role: str,
    text: str,
    sender_name: Optional[str],
    sender_email: Optional[str],
) -> list[dict]:
    messages = list(parsed_payload.get("messages") or [])
    messages.append(
        {
            "role": role,
            "text": str(text or "").strip(),
            "created_at": _utc_iso_now(),
            "sender_name": str(sender_name or "").strip() or None,
            "sender_email": str(sender_email or "").strip() or None,
        }
    )
    return messages


def _build_admin_new_ticket_email_html(
    *,
    reporter_name: str,
    reporter_email: str,
    subject_text: str,
    category: str,
    message_text: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>New support ticket</title>
</head>
<body style="font-family: Arial, sans-serif; background:#f5f7fb; padding:24px;">
  <div style="max-width:620px; margin:0 auto; background:#ffffff; border-radius:12px; padding:24px; border:1px solid #e5e7eb;">
    <h2 style="margin:0 0 12px 0; color:#0f172a;">New support ticket</h2>
    <p style="margin:0 0 8px 0; color:#334155;"><strong>From:</strong> {reporter_name} ({reporter_email})</p>
    <p style="margin:0 0 8px 0; color:#334155;"><strong>Subject:</strong> {subject_text}</p>
    <p style="margin:0 0 14px 0; color:#334155;"><strong>Category:</strong> {category}</p>
    <div style="padding:12px; border:1px solid #e5e7eb; border-radius:8px; background:#f8fafc; color:#0f172a;">
      {message_text}
    </div>
    <p style="margin:14px 0 0 0; color:#334155;">Open Admin Panel to reply.</p>
  </div>
</body>
</html>"""


def _build_admin_ticket_update_email_html(
    *,
    reporter_name: str,
    reporter_email: str,
    subject_text: str,
    message_text: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Support ticket updated</title>
</head>
<body style="font-family: Arial, sans-serif; background:#f5f7fb; padding:24px;">
  <div style="max-width:620px; margin:0 auto; background:#ffffff; border-radius:12px; padding:24px; border:1px solid #e5e7eb;">
    <h2 style="margin:0 0 12px 0; color:#0f172a;">User replied to support ticket</h2>
    <p style="margin:0 0 8px 0; color:#334155;"><strong>From:</strong> {reporter_name} ({reporter_email})</p>
    <p style="margin:0 0 8px 0; color:#334155;"><strong>Subject:</strong> {subject_text}</p>
    <div style="padding:12px; border:1px solid #e5e7eb; border-radius:8px; background:#f8fafc; color:#0f172a;">
      {message_text}
    </div>
    <p style="margin:14px 0 0 0; color:#334155;">Open Admin Panel to continue the thread.</p>
  </div>
</body>
</html>"""


def _build_user_ticket_reply_email_html(
    *,
    user_name: str,
    subject_text: str,
    reply_text: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Support ticket reply</title>
</head>
<body style="font-family: Arial, sans-serif; background:#f5f7fb; padding:24px;">
  <div style="max-width:620px; margin:0 auto; background:#ffffff; border-radius:12px; padding:24px; border:1px solid #e5e7eb;">
    <h2 style="margin:0 0 12px 0; color:#0f172a;">Support replied to your ticket</h2>
    <p style="margin:0 0 8px 0; color:#334155;">Hi {user_name},</p>
    <p style="margin:0 0 8px 0; color:#334155;"><strong>Subject:</strong> {subject_text}</p>
    <div style="padding:12px; border:1px solid #e5e7eb; border-radius:8px; background:#f8fafc; color:#0f172a;">
      {reply_text}
    </div>
    <p style="margin:14px 0 0 0; color:#334155;">Open Support page to continue the conversation.</p>
  </div>
</body>
</html>"""


def _map_ticket(ticket) -> SupportTicketItemResponse:
    parsed = _parse_ticket_message(getattr(ticket, "message", ""))
    user = getattr(ticket, "user", None)
    messages = list(parsed.get("messages") or [])
    return SupportTicketItemResponse(
        id=str(ticket.id),
        user_id=str(ticket.userId),
        user_email=str(getattr(user, "email", "") or "") or None,
        user_name=str(getattr(user, "username", "") or "") or None,
        subject=str(getattr(ticket, "subject", "") or "").strip() or "(No subject)",
        category=str(parsed.get("category") or "Other"),
        user_message=_extract_first_user_message(messages),
        admin_reply=_extract_latest_admin_message(messages),
        messages=[SupportTicketMessageItem(**item) for item in messages],
        status=str(_enum_value(getattr(ticket, "status", None)) or "open"),
        created_at=getattr(ticket, "createdAt", None),
        updated_at=getattr(ticket, "updatedAt", None),
    )


async def _notify_admins_new_ticket(ticket, current_user: Any, category: str, user_message: str):
    admins = await db.user.find_many(
        where={"role": "admin"},
        include={"notificationConfig": True},
    )
    if not admins:
        return

    subject_text = str(getattr(ticket, "subject", "") or "").strip() or "(No subject)"
    reporter_name = str(getattr(current_user, "username", "") or "User")
    reporter_email = str(getattr(current_user, "email", "") or "-")
    notification_title = "New support ticket"
    notification_message = f"{subject_text} | from {reporter_name} ({reporter_email})"

    for admin in admins:
        try:
            await dispatch_notification_to_user(
                admin,
                title=notification_title,
                message=notification_message,
                related_link="/admin",
                email_subject=f"[Support] {subject_text}",
                email_html=_build_admin_new_ticket_email_html(
                    reporter_name=reporter_name,
                    reporter_email=reporter_email,
                    subject_text=subject_text,
                    category=category,
                    message_text=user_message,
                ),
                action_label="Open admin panel",
            )
        except Exception as exc:
            print(f"[support] failed to notify admin {getattr(admin, 'email', None)}: {exc}")


async def _notify_admins_user_reply(ticket, current_user: Any, reply_text: str):
    admins = await db.user.find_many(
        where={"role": "admin"},
        include={"notificationConfig": True},
    )
    if not admins:
        return

    subject_text = str(getattr(ticket, "subject", "") or "").strip() or "(No subject)"
    reporter_name = str(getattr(current_user, "username", "") or "User")
    reporter_email = str(getattr(current_user, "email", "") or "-")
    notification_title = "Support ticket updated by user"
    notification_message = f"{subject_text} | from {reporter_name}"

    for admin in admins:
        try:
            await dispatch_notification_to_user(
                admin,
                title=notification_title,
                message=notification_message,
                related_link="/admin",
                email_subject=f"[Support Update] {subject_text}",
                email_html=_build_admin_ticket_update_email_html(
                    reporter_name=reporter_name,
                    reporter_email=reporter_email,
                    subject_text=subject_text,
                    message_text=reply_text,
                ),
                action_label="Open admin panel",
            )
        except Exception as exc:
            print(f"[support] failed to notify admin update {getattr(admin, 'email', None)}: {exc}")


async def _notify_user_ticket_reply(ticket, reply_text: str):
    user = await db.user.find_unique(
        where={"id": str(ticket.userId)},
        include={"notificationConfig": True},
    )
    if not user:
        return

    subject_text = str(getattr(ticket, "subject", "") or "").strip() or "(No subject)"
    try:
        await dispatch_notification_to_user(
            user,
            title="Support replied to your ticket",
            message=subject_text,
            related_link="/support",
            email_subject=f"[Support Reply] {subject_text}",
            email_html=_build_user_ticket_reply_email_html(
                user_name=str(getattr(user, "username", "") or "Trader"),
                subject_text=subject_text,
                reply_text=reply_text,
            ),
            action_label="Open support",
        )
    except Exception as exc:
        print(f"[support] failed to notify ticket reply for {getattr(user, 'email', None)}: {exc}")


@support_router.get("/tickets", response_model=list[SupportTicketItemResponse])
async def get_my_tickets(
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    user_id = str(current_user.id)
    tickets = await db.supportticket.find_many(
        where={"userId": user_id},
        include={"user": True},
        order={"createdAt": "desc"},
    )
    return [_map_ticket(ticket) for ticket in tickets]


@support_router.post(
    "/tickets",
    response_model=SupportTicketItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_ticket(
    data: CreateSupportTicketRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    user_id = str(current_user.id)

    subject_text = str(data.subject or "").strip()
    message_text = str(data.message or "").strip()
    category_text = str(data.category or "Other").strip() or "Other"

    if not subject_text:
        raise HTTPException(status_code=400, detail="subject is required")
    if not message_text:
        raise HTTPException(status_code=400, detail="message is required")

    payload = _build_ticket_message_payload(
        category=category_text,
        messages=[
            {
                "role": "user",
                "text": message_text,
                "created_at": _utc_iso_now(),
                "sender_name": str(getattr(current_user, "username", "") or "User"),
                "sender_email": str(getattr(current_user, "email", "") or ""),
            }
        ],
    )

    ticket = await db.supportticket.create(
        data={
            "user": {"connect": {"id": user_id}},
            "subject": subject_text[:100],
            "message": payload,
            "status": "open",
        },
        include={"user": True},
    )

    await _notify_admins_new_ticket(
        ticket=ticket,
        current_user=current_user,
        category=category_text,
        user_message=message_text,
    )

    return _map_ticket(ticket)


@support_router.post(
    "/tickets/{ticket_id}/reply",
    response_model=SupportTicketItemResponse,
)
async def user_reply_ticket(
    ticket_id: str,
    data: UserReplySupportTicketRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    user_id = str(current_user.id)
    ticket = await db.supportticket.find_first(
        where={"id": ticket_id, "userId": user_id},
        include={"user": True},
    )
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    reply_text = str(data.message or "").strip()
    if not reply_text:
        raise HTTPException(status_code=400, detail="message is required")

    current_status = str(_enum_value(getattr(ticket, "status", None)) or "open")
    if current_status == "closed":
        raise HTTPException(status_code=409, detail="Ticket is closed")

    parsed = _parse_ticket_message(getattr(ticket, "message", ""))
    messages = _append_message(
        parsed_payload=parsed,
        role="user",
        text=reply_text,
        sender_name=str(getattr(current_user, "username", "") or "User"),
        sender_email=str(getattr(current_user, "email", "") or ""),
    )
    next_status = "in_progress" if current_status == "resolved" else _normalize_status(current_status, default="open")
    next_payload = _build_ticket_message_payload(
        category=str(parsed.get("category") or "Other"),
        messages=messages,
    )

    await db.supportticket.update(
        where={"id": ticket_id},
        data={
            "message": next_payload,
            "status": next_status,
        },
    )

    refreshed = await db.supportticket.find_unique(
        where={"id": ticket_id},
        include={"user": True},
    )
    if not refreshed:
        raise HTTPException(status_code=404, detail="Ticket not found after update")

    await _notify_admins_user_reply(
        ticket=refreshed,
        current_user=current_user,
        reply_text=reply_text,
    )
    return _map_ticket(refreshed)


@support_router.get("/admin/tickets", response_model=list[SupportTicketItemResponse])
async def get_admin_tickets(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    current_user: Annotated[Any, Depends(get_current_active_user)] = None,
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")

    where = {}
    if status_filter and status_filter.strip().lower() != "all":
        where["status"] = _normalize_status(status_filter, default="open")

    tickets = await db.supportticket.find_many(
        where=where,
        include={"user": True},
        order={"createdAt": "desc"},
    )
    return [_map_ticket(ticket) for ticket in tickets]


@support_router.patch(
    "/admin/tickets/{ticket_id}/reply",
    response_model=SupportTicketItemResponse,
)
async def admin_reply_ticket(
    ticket_id: str,
    data: AdminReplySupportTicketRequest,
    current_user: Annotated[Any, Depends(get_current_active_user)],
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")

    ticket = await db.supportticket.find_unique(
        where={"id": ticket_id},
        include={"user": True},
    )
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    reply_text = str(data.reply or "").strip()
    if not reply_text:
        raise HTTPException(status_code=400, detail="reply is required")

    next_status = _normalize_status(data.status, default="resolved")

    parsed = _parse_ticket_message(getattr(ticket, "message", ""))
    messages = _append_message(
        parsed_payload=parsed,
        role="admin",
        text=reply_text,
        sender_name=str(getattr(current_user, "username", "") or getattr(current_user, "email", "") or "admin"),
        sender_email=str(getattr(current_user, "email", "") or ""),
    )
    next_payload = _build_ticket_message_payload(
        category=str(parsed.get("category") or "Other"),
        messages=messages,
    )

    await db.supportticket.update(
        where={"id": ticket_id},
        data={
            "message": next_payload,
            "status": next_status,
        },
    )

    refreshed = await db.supportticket.find_unique(
        where={"id": ticket_id},
        include={"user": True},
    )
    if not refreshed:
        raise HTTPException(status_code=404, detail="Ticket not found after update")

    await _notify_user_ticket_reply(ticket=refreshed, reply_text=reply_text)
    return _map_ticket(refreshed)
