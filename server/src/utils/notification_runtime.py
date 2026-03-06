from __future__ import annotations

import asyncio
import html
import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from ..database.client import db
from .notification_delivery import (
    claim_notification_dedupe,
    dispatch_notification_to_user,
)

logger = logging.getLogger(__name__)

_THRESHOLD_POLL_SECONDS = max(60, int(os.getenv("NOTIFICATION_THRESHOLD_POLL_SECONDS", "300")))
_SUMMARY_POLL_SECONDS = max(300, int(os.getenv("NOTIFICATION_SUMMARY_POLL_SECONDS", "1800")))
_SUMMARY_HOUR = min(23, max(0, int(os.getenv("NOTIFICATION_SUMMARY_HOUR", "9"))))
_DEFAULT_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Bangkok")
_MARGIN_ALERT_TTL_SECONDS = 60 * 60 * 6
_SUMMARY_WEEKLY_TTL_SECONDS = 60 * 60 * 24 * 45
_SUMMARY_MONTHLY_TTL_SECONDS = 60 * 60 * 24 * 90
_DAILY_ALERT_TTL_SECONDS = 60 * 60 * 36


def _resolve_timezone():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(_DEFAULT_TIMEZONE)
    except Exception:
        return timezone.utc


APP_TIMEZONE = _resolve_timezone()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _trim_text(value: Any) -> str:
    return str(value or "").strip()


def _enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def _order_net_profit(order: Any) -> float:
    profit = _to_float(getattr(order, "profit", 0.0), 0.0)
    commission = _to_float(getattr(order, "commission", 0.0), 0.0)
    swap = _to_float(getattr(order, "swap", 0.0), 0.0)
    return profit + commission + swap


def _local_period_bounds(start_day: date, end_day_exclusive: date) -> tuple[datetime, datetime]:
    local_start = datetime.combine(start_day, time.min, tzinfo=APP_TIMEZONE)
    local_end = datetime.combine(end_day_exclusive, time.min, tzinfo=APP_TIMEZONE)
    return (local_start.astimezone(timezone.utc), local_end.astimezone(timezone.utc))


async def _get_active_accounts(user_id: str) -> list[Any]:
    return await db.tradingaccount.find_many(
        where={
            "userId": user_id,
            "recordStatus": "active",
        }
    )


async def _get_orders_for_accounts(
    *,
    account_ids: list[str],
    start_dt_utc: datetime,
    end_dt_utc: datetime,
) -> list[Any]:
    if not account_ids:
        return []
    return await db.orderhistory.find_many(
        where={
            "accountId": {"in": account_ids},
            "closeTime": {
                "gte": start_dt_utc,
                "lt": end_dt_utc,
            },
        },
    )


def _build_summary_email_html(
    *,
    title: str,
    user_name: str,
    period_label: str,
    total_net_profit: float,
    total_trades: int,
    profitable_days: int,
    win_rate: float,
) -> str:
    safe_title = html.escape(title)
    safe_user_name = html.escape(user_name or "Trader")
    safe_period = html.escape(period_label)
    profit_color = "#16a34a" if total_net_profit >= 0 else "#dc2626"
    profit_prefix = "+" if total_net_profit >= 0 else ""
    win_rate_text = f"{win_rate:.1f}%"
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
    <p style="margin:0 0 18px 0; color:#334155;">Hi {safe_user_name}, here is your performance summary for {safe_period}.</p>
    <div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px;">
      <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc;">
        <div style="font-size:12px; color:#64748b; text-transform:uppercase;">Net profit</div>
        <div style="font-size:24px; font-weight:700; color:{profit_color};">{profit_prefix}${total_net_profit:,.2f}</div>
      </div>
      <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc;">
        <div style="font-size:12px; color:#64748b; text-transform:uppercase;">Trades</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a;">{total_trades}</div>
      </div>
      <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc;">
        <div style="font-size:12px; color:#64748b; text-transform:uppercase;">Profitable days</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a;">{profitable_days}</div>
      </div>
      <div style="padding:14px; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc;">
        <div style="font-size:12px; color:#64748b; text-transform:uppercase;">Win rate</div>
        <div style="font-size:24px; font-weight:700; color:#0f172a;">{win_rate_text}</div>
      </div>
    </div>
    <p style="margin:18px 0 0 0; color:#475569;">Open the calendar page to inspect the full trading breakdown.</p>
  </div>
</body>
</html>"""


def _build_summary_stats(orders: list[Any]) -> dict[str, float | int]:
    total_net_profit = 0.0
    total_trades = 0
    winning_trades = 0
    profitable_days: dict[date, float] = {}

    for order in orders:
        total_trades += 1
        net_profit = _order_net_profit(order)
        total_net_profit += net_profit
        if net_profit > 0:
            winning_trades += 1

        close_time = getattr(order, "closeTime", None)
        if close_time is None:
            continue
        if getattr(close_time, "tzinfo", None) is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        local_day = close_time.astimezone(APP_TIMEZONE).date()
        profitable_days[local_day] = float(profitable_days.get(local_day, 0.0)) + net_profit

    positive_days = sum(1 for value in profitable_days.values() if float(value) > 0.0)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    return {
        "total_net_profit": round(total_net_profit, 2),
        "total_trades": int(total_trades),
        "profitable_days": int(positive_days),
        "win_rate": round(win_rate, 1),
    }


async def process_threshold_notifications() -> None:
    configs = await db.notificationconfig.find_many(include={"user": True})
    if not configs:
        return

    now_local = datetime.now(APP_TIMEZONE)
    today_local = now_local.date()
    tomorrow_local = today_local + timedelta(days=1)
    start_dt_utc, end_dt_utc = _local_period_bounds(today_local, tomorrow_local)

    for config in configs:
        profit_target = getattr(config, "alertProfitTarget", None)
        loss_limit = getattr(config, "alertLossLimit", None)
        margin_threshold = getattr(config, "alertMarginLevelThreshold", None)
        if profit_target is None and loss_limit is None and margin_threshold is None:
            continue

        user = getattr(config, "user", None)
        if not user:
            continue
        if _trim_text(_enum_value(getattr(user, "status", None))).lower() == "banned":
            continue

        user_id = _trim_text(getattr(user, "id", None))
        if not user_id:
            continue

        accounts = await _get_active_accounts(user_id)
        account_ids = [str(getattr(account, "id", "") or "").strip() for account in accounts if getattr(account, "id", None)]
        orders = await _get_orders_for_accounts(
            account_ids=[aid for aid in account_ids if aid],
            start_dt_utc=start_dt_utc,
            end_dt_utc=end_dt_utc,
        )
        total_daily_net = round(sum(_order_net_profit(order) for order in orders), 2)

        if profit_target is not None:
            threshold_value = abs(_to_float(profit_target, 0.0))
            if threshold_value > 0 and total_daily_net >= threshold_value:
                dedupe_key = f"notify:profit:{user_id}:{today_local.isoformat()}"
                if claim_notification_dedupe(key=dedupe_key, ttl_seconds=_DAILY_ALERT_TTL_SECONDS):
                    title = "Daily profit target reached"
                    message = f"Today's net profit is +${total_daily_net:,.2f}, above your ${threshold_value:,.2f} target."
                    await dispatch_notification_to_user(
                        user,
                        title=title,
                        message=message,
                        related_link="/calendar",
                        email_subject=f"[Alert] {title}",
                        action_label="Open calendar",
                    )

        if loss_limit is not None:
            threshold_value = -abs(_to_float(loss_limit, 0.0))
            if threshold_value < 0 and total_daily_net <= threshold_value:
                dedupe_key = f"notify:loss:{user_id}:{today_local.isoformat()}"
                if claim_notification_dedupe(key=dedupe_key, ttl_seconds=_DAILY_ALERT_TTL_SECONDS):
                    title = "Daily loss limit reached"
                    message = (
                        f"Today's net profit is ${total_daily_net:,.2f}, below your ${threshold_value:,.2f} loss limit."
                    )
                    await dispatch_notification_to_user(
                        user,
                        title=title,
                        message=message,
                        related_link="/calendar",
                        email_subject=f"[Alert] {title}",
                        action_label="Open calendar",
                    )

        if margin_threshold is not None:
            threshold_value = abs(_to_float(margin_threshold, 0.0))
            if threshold_value <= 0:
                continue

            for account in accounts:
                margin_level = _to_float(getattr(account, "marginLevel", None), 0.0)
                if margin_level <= 0 or margin_level > threshold_value:
                    continue

                account_id = _trim_text(getattr(account, "id", None))
                dedupe_key = f"notify:margin:{user_id}:{account_id}:{int(threshold_value)}"
                if not claim_notification_dedupe(key=dedupe_key, ttl_seconds=_MARGIN_ALERT_TTL_SECONDS):
                    continue

                broker_name = _trim_text(getattr(account, "brokerName", None))
                server_name = _trim_text(getattr(account, "serverName", None))
                login_id = _trim_text(getattr(account, "mt5LoginId", None))
                account_label = " / ".join(part for part in [broker_name, server_name] if part)
                if login_id:
                    account_label = f"{account_label} ({login_id})" if account_label else f"MT5 {login_id}"

                title = "Margin level is below threshold"
                message = f"Current margin level is {margin_level:,.2f}%"
                if account_label:
                    message = f"{message} on {account_label}"
                message = f"{message}, below your {threshold_value:,.2f}% threshold."
                await dispatch_notification_to_user(
                    user,
                    title=title,
                    message=message,
                    related_link="/bot-control",
                    email_subject=f"[Alert] {title}",
                    action_label="Open bot control",
                )


async def process_scheduled_summaries() -> None:
    now_local = datetime.now(APP_TIMEZONE)
    if now_local.hour < _SUMMARY_HOUR:
        return

    configs = await db.notificationconfig.find_many(include={"user": True})
    if not configs:
        return

    today_local = now_local.date()
    current_week_start = today_local - timedelta(days=today_local.weekday())
    weekly_start = current_week_start - timedelta(days=7)
    weekly_end = current_week_start

    current_month_start = today_local.replace(day=1)
    monthly_end_inclusive = current_month_start - timedelta(days=1)
    monthly_start = monthly_end_inclusive.replace(day=1)
    monthly_end = current_month_start

    for config in configs:
        user = getattr(config, "user", None)
        if not user:
            continue
        if _trim_text(_enum_value(getattr(user, "status", None))).lower() == "banned":
            continue
        user_id = _trim_text(getattr(user, "id", None))
        if not user_id:
            continue

        accounts = await _get_active_accounts(user_id)
        account_ids = [str(getattr(account, "id", "") or "").strip() for account in accounts if getattr(account, "id", None)]
        user_name = _trim_text(getattr(user, "username", None)) or "Trader"

        if bool(getattr(config, "enableWeeklySummary", False)):
            dedupe_key = f"notify:summary:weekly:{user_id}:{weekly_start.isoformat()}"
            if claim_notification_dedupe(key=dedupe_key, ttl_seconds=_SUMMARY_WEEKLY_TTL_SECONDS):
                weekly_start_dt_utc, weekly_end_dt_utc = _local_period_bounds(weekly_start, weekly_end)
                weekly_orders = await _get_orders_for_accounts(
                    account_ids=[aid for aid in account_ids if aid],
                    start_dt_utc=weekly_start_dt_utc,
                    end_dt_utc=weekly_end_dt_utc,
                )
                stats = _build_summary_stats(weekly_orders)
                period_label = f"{weekly_start.strftime('%b %d')} - {(weekly_end - timedelta(days=1)).strftime('%b %d, %Y')}"
                title = "Weekly performance summary"
                message = (
                    f"{period_label}: {stats['total_trades']} trades, "
                    f"net {'+' if float(stats['total_net_profit']) >= 0 else ''}${float(stats['total_net_profit']):,.2f}, "
                    f"win rate {float(stats['win_rate']):.1f}%."
                )
                email_html = _build_summary_email_html(
                    title=title,
                    user_name=user_name,
                    period_label=period_label,
                    total_net_profit=float(stats["total_net_profit"]),
                    total_trades=int(stats["total_trades"]),
                    profitable_days=int(stats["profitable_days"]),
                    win_rate=float(stats["win_rate"]),
                )
                await dispatch_notification_to_user(
                    user,
                    title=title,
                    message=message,
                    related_link="/calendar",
                    email_subject=f"[Summary] {title} - {period_label}",
                    email_html=email_html,
                    action_label="Open calendar",
                )

        if bool(getattr(config, "enableMonthlySummary", False)):
            dedupe_key = f"notify:summary:monthly:{user_id}:{monthly_start.isoformat()}"
            if claim_notification_dedupe(key=dedupe_key, ttl_seconds=_SUMMARY_MONTHLY_TTL_SECONDS):
                monthly_start_dt_utc, monthly_end_dt_utc = _local_period_bounds(monthly_start, monthly_end)
                monthly_orders = await _get_orders_for_accounts(
                    account_ids=[aid for aid in account_ids if aid],
                    start_dt_utc=monthly_start_dt_utc,
                    end_dt_utc=monthly_end_dt_utc,
                )
                stats = _build_summary_stats(monthly_orders)
                period_label = monthly_start.strftime("%B %Y")
                title = "Monthly performance summary"
                message = (
                    f"{period_label}: {stats['total_trades']} trades, "
                    f"net {'+' if float(stats['total_net_profit']) >= 0 else ''}${float(stats['total_net_profit']):,.2f}, "
                    f"win rate {float(stats['win_rate']):.1f}%."
                )
                email_html = _build_summary_email_html(
                    title=title,
                    user_name=user_name,
                    period_label=period_label,
                    total_net_profit=float(stats["total_net_profit"]),
                    total_trades=int(stats["total_trades"]),
                    profitable_days=int(stats["profitable_days"]),
                    win_rate=float(stats["win_rate"]),
                )
                await dispatch_notification_to_user(
                    user,
                    title=title,
                    message=message,
                    related_link="/calendar",
                    email_subject=f"[Summary] {title} - {period_label}",
                    email_html=email_html,
                    action_label="Open calendar",
                )


async def _run_worker(name: str, interval_seconds: int, processor) -> None:
    logger.info("%s worker started (interval=%ss)", name, interval_seconds)
    while True:
        try:
            await processor()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("%s worker failed: %s", name, exc)
        await asyncio.sleep(interval_seconds)


async def run_threshold_notification_worker() -> None:
    await _run_worker("notification-threshold", _THRESHOLD_POLL_SECONDS, process_threshold_notifications)


async def run_summary_notification_worker() -> None:
    await _run_worker("notification-summary", _SUMMARY_POLL_SECONDS, process_scheduled_summaries)
