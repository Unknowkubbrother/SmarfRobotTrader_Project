from __future__ import annotations

import html
import os
from typing import Any

import httpx

from ..database.client import db, r_cache
from lib.untils import send_email

_DEFAULT_FRONTEND_URL = "http://localhost:3000"


def is_email_notification_enabled(notification_config: Any) -> bool:
    if not notification_config:
        return True
    value = getattr(notification_config, "emailNotificationEnable", None)
    if value is None:
        return True
    return bool(value)


def _trim_text(value: Any, *, max_length: int | None = None) -> str:
    text = str(value or "").strip()
    if max_length is not None and len(text) > max_length:
        return text[:max_length]
    return text


def _resolve_frontend_url() -> str:
    raw_url = (
        os.getenv("FRONTEND_URL")
        or os.getenv("APP_URL")
        or os.getenv("NEXT_PUBLIC_APP_URL")
        or _DEFAULT_FRONTEND_URL
    )
    return str(raw_url).rstrip("/")


def build_absolute_related_link(related_link: str | None) -> str | None:
    path = _trim_text(related_link)
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{_resolve_frontend_url()}{path}"


def build_generic_notification_email_html(
    *,
    title: str,
    greeting: str | None = None,
    message: str,
    related_link: str | None = None,
    action_label: str = "Open dashboard",
) -> str:
    safe_title = html.escape(_trim_text(title) or "Notification")
    safe_greeting = html.escape(_trim_text(greeting) or "Hello,")
    safe_message = html.escape(_trim_text(message) or "You have a new notification.")
    action_url = build_absolute_related_link(related_link)
    action_block = ""
    if action_url:
        safe_url = html.escape(action_url, quote=True)
        safe_label = html.escape(_trim_text(action_label) or "Open dashboard")
        action_block = (
            f'<p style="margin:24px 0 0 0;">'
            f'<a href="{safe_url}" '
            f'style="display:inline-block; background:#2563eb; color:#ffffff; text-decoration:none; '
            f'padding:12px 18px; border-radius:10px; font-weight:600;">{safe_label}</a>'
            f"</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
</head>
<body style="margin:0; padding:24px; background:#f5f7fb; font-family:Arial, sans-serif;">
  <div style="max-width:620px; margin:0 auto; background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:24px;">
    <h2 style="margin:0 0 12px 0; color:#0f172a;">{safe_title}</h2>
    <p style="margin:0 0 12px 0; color:#334155;">{safe_greeting}</p>
    <div style="padding:14px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:10px; color:#0f172a; line-height:1.6;">
      {safe_message}
    </div>
    {action_block}
  </div>
</body>
</html>"""


def claim_notification_dedupe(*, key: str, ttl_seconds: int) -> bool:
    dedupe_key = _trim_text(key)
    if not dedupe_key or ttl_seconds <= 0:
        return True
    try:
        return bool(r_cache.set(dedupe_key, "1", ex=int(ttl_seconds), nx=True))
    except Exception:
        return True


async def _create_in_app_notification(
    *,
    user_id: str,
    title: str,
    message: str,
    related_link: str | None = None,
) -> None:
    await db.notification.create(
        data={
            "userId": user_id,
            "title": _trim_text(title, max_length=100),
            "message": _trim_text(message),
            "relatedLink": _trim_text(related_link, max_length=255) or None,
        }
    )


async def _send_discord_webhook(
    *,
    webhook_url: str,
    title: str,
    message: str,
    related_link: str | None = None,
) -> None:
    url = _trim_text(webhook_url)
    if not url:
        return

    absolute_link = build_absolute_related_link(related_link)
    description = _trim_text(message)
    if absolute_link:
        description = f"{description}\n\nOpen: {absolute_link}"

    payload = {
        "embeds": [
            {
                "title": _trim_text(title, max_length=256) or "Notification",
                "description": description[:4096],
                "color": 3447003,
            }
        ]
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()


async def get_user_with_notification_config(user_id: str):
    normalized_user_id = _trim_text(user_id)
    if not normalized_user_id:
        return None
    return await db.user.find_unique(
        where={"id": normalized_user_id},
        include={"notificationConfig": True},
    )


async def dispatch_notification_to_user(
    user: Any,
    *,
    title: str,
    message: str,
    related_link: str | None = None,
    email_subject: str | None = None,
    email_html: str | None = None,
    email_greeting: str | None = None,
    action_label: str = "Open dashboard",
    send_in_app: bool = True,
    send_email_channel: bool = True,
    send_discord_channel: bool = True,
) -> dict[str, bool]:
    if not user:
        return {"in_app": False, "email": False, "discord": False}

    user_id = _trim_text(getattr(user, "id", None))
    if not user_id:
        return {"in_app": False, "email": False, "discord": False}

    notification_config = getattr(user, "notificationConfig", None)
    normalized_title = _trim_text(title, max_length=100) or "Notification"
    normalized_message = _trim_text(message) or "You have a new notification."
    normalized_related_link = _trim_text(related_link, max_length=255) or None
    status = {"in_app": False, "email": False, "discord": False}

    if send_in_app:
        try:
            await _create_in_app_notification(
                user_id=user_id,
                title=normalized_title,
                message=normalized_message,
                related_link=normalized_related_link,
            )
            status["in_app"] = True
        except Exception as exc:
            print(f"[notify] failed to create in-app notification for {user_id}: {exc}")

    if send_discord_channel:
        webhook_url = _trim_text(getattr(notification_config, "discordWebhookUrl", None))
        if webhook_url:
            try:
                await _send_discord_webhook(
                    webhook_url=webhook_url,
                    title=normalized_title,
                    message=normalized_message,
                    related_link=normalized_related_link,
                )
                status["discord"] = True
            except Exception as exc:
                print(f"[notify] failed to send Discord webhook for {user_id}: {exc}")

    if send_email_channel and getattr(user, "email", None) and is_email_notification_enabled(notification_config):
        try:
            html_content = email_html or build_generic_notification_email_html(
                title=normalized_title,
                greeting=email_greeting or f"Hi {_trim_text(getattr(user, 'username', None)) or 'Trader'},",
                message=normalized_message,
                related_link=normalized_related_link,
                action_label=action_label,
            )
            send_email(
                to_email=str(user.email),
                subject=_trim_text(email_subject or normalized_title) or "Notification",
                html_content=html_content,
            )
            status["email"] = True
        except Exception as exc:
            print(f"[notify] failed to send email for {user_id}: {exc}")

    return status


async def dispatch_notification_to_user_id(
    user_id: str,
    *,
    title: str,
    message: str,
    related_link: str | None = None,
    email_subject: str | None = None,
    email_html: str | None = None,
    email_greeting: str | None = None,
    action_label: str = "Open dashboard",
    send_in_app: bool = True,
    send_email_channel: bool = True,
    send_discord_channel: bool = True,
) -> dict[str, bool]:
    user = await get_user_with_notification_config(user_id)
    if not user:
        return {"in_app": False, "email": False, "discord": False}
    return await dispatch_notification_to_user(
        user,
        title=title,
        message=message,
        related_link=related_link,
        email_subject=email_subject,
        email_html=email_html,
        email_greeting=email_greeting,
        action_label=action_label,
        send_in_app=send_in_app,
        send_email_channel=send_email_channel,
        send_discord_channel=send_discord_channel,
    )
