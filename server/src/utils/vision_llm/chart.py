"""
Chart generation module — fetches OHLC data from bot runtime containers
or uses provided chart payloads, then renders candlestick charts as
base64-encoded PNG images.
"""

import base64
import json
import subprocess
from dataclasses import dataclass
from io import BytesIO
from datetime import datetime, timedelta, timezone

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf                # noqa: E402
import matplotlib.ticker as ticker      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402

from ..mt5_bot_runner import BotRunnerError, resolve_bot_runtime_container

# ── Chart style (TradingView dark) ───────────────────────────────────
_MARKET_COLORS = mpf.make_marketcolors(
    up="#089981", down="#F23645",
    edge="inherit", wick="inherit", ohlc="i",
)
_CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=_MARKET_COLORS,
    gridstyle="",
    facecolor="#131722",
    y_on_right=True,
)

_MAX_CHART_FETCH_BARS = 2000


class NoMarketDataError(ValueError):
    """Raised when MT5 returns no candles for requested interval."""


class MT5ConnectionError(RuntimeError):
    """Raised when MT5 bridge cannot be reached or initialized."""


@dataclass(frozen=True)
class ChartImageResult:
    image_base64: str
    source_mode: str
    source_label: str
    resolved_bar_time: str


@dataclass(frozen=True)
class ChartRatesResult:
    rates: list[dict]
    source_mode: str
    source_label: str
    resolved_from_time: str
    resolved_to_time: str

_DOCKER_FETCH_RATES_SCRIPT = r"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

try:
    import rpyc
    rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = 120.0
except Exception:
    pass

from mt5linux import MetaTrader5

FORMATS = ("%Y.%m.%d %H.%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")
TF_SECONDS = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
}

payload = {"rates": [], "resolved_bar_time": "", "server": "", "login": "", "error": ""}

def parse_dt(raw):
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        raise ValueError("date_time is required")
    for fmt in FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"invalid date_time: {text}")

try:
    req = json.loads(os.environ.get("VISION_LLM_CHART_REQ", "{}"))
    date_time = parse_dt(req.get("date_time"))
    symbol = str(req.get("symbol", "EURUSD") or "EURUSD").strip().upper()
    timeframe = str(req.get("timeframe", "H1") or "H1").strip().upper()
    timeout_ms = max(30000, int(req.get("timeout_ms", 120000) or 120000))
    tf_secs = int(TF_SECONDS.get(timeframe, 3600))
    epoch = int(date_time.replace(tzinfo=timezone.utc).timestamp()) if date_time.tzinfo is None else int(
        date_time.astimezone(timezone.utc).timestamp()
    )
    aligned_epoch = epoch - (epoch % tf_secs)
    start_utc = datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)
    end_utc = start_utc + timedelta(seconds=tf_secs)

    mt5 = MetaTrader5(host="localhost", port=8001)
    if not mt5.initialize(timeout=timeout_ms):
        payload["error"] = f"initialize_failed:{mt5.last_error()!r}"
    else:
        account = mt5.account_info()
        if account is not None:
            payload["server"] = str(getattr(account, "server", "") or "")
            try:
                payload["login"] = str(int(getattr(account, "login", 0) or 0))
            except Exception:
                payload["login"] = str(getattr(account, "login", "") or "")
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_utc, end_utc)
        if rates is None or len(rates) == 0:
            payload["error"] = f"no_m1_data:{symbol}:{start_utc.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            payload["resolved_bar_time"] = start_utc.strftime("%Y-%m-%d %H:%M:%S")
            payload["rates"] = [
                {
                    "time": int(row["time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "tick_volume": float(row["tick_volume"]),
                }
                for row in rates
            ]
    try:
        mt5.shutdown()
    except Exception:
        pass
except Exception as exc:
    payload["error"] = f"exception:{type(exc).__name__}:{exc}"

print(json.dumps(payload, ensure_ascii=False))
""".strip()


_DOCKER_FETCH_TIMEFRAME_RATES_SCRIPT = r"""
import json
import os
from datetime import datetime, timedelta, timezone

try:
    import rpyc
    rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = 120.0
except Exception:
    pass

from mt5linux import MetaTrader5

FORMATS = ("%Y.%m.%d %H.%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")
TF_SECONDS = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
}

payload = {
    "rates": [],
    "resolved_from_time": "",
    "resolved_to_time": "",
    "server": "",
    "login": "",
    "error": "",
}


def parse_dt(raw):
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        raise ValueError("date_time is required")
    for fmt in FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"invalid date_time: {text}")


def serialize_rates(rows):
    out = []
    if rows is None:
        return out
    for row in rows:
        try:
            volume = float(row["tick_volume"])
        except Exception:
            try:
                volume = float(row["real_volume"])
            except Exception:
                volume = 0.0
        out.append(
            {
                "time": int(row["time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "tick_volume": volume,
            }
        )
    return out


def aggregate_rates(rows, tf_secs, start_epoch, end_epoch):
    normalized = sorted(
        [
            row for row in serialize_rates(rows)
            if start_epoch <= int(row["time"]) < end_epoch
        ],
        key=lambda item: int(item["time"]),
    )
    if tf_secs <= 60:
        return normalized

    out = []
    for row in normalized:
        bucket = (int(row["time"]) // tf_secs) * tf_secs
        if out and int(out[-1]["time"]) == bucket:
            out[-1]["high"] = max(float(out[-1]["high"]), float(row["high"]))
            out[-1]["low"] = min(float(out[-1]["low"]), float(row["low"]))
            out[-1]["close"] = float(row["close"])
            out[-1]["tick_volume"] = float(out[-1]["tick_volume"]) + float(row["tick_volume"])
            continue
        out.append(
            {
                "time": int(bucket),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "tick_volume": float(row["tick_volume"]),
            }
        )
    return out


def fetch_m1_rates_from_pos(mt5, symbol, count):
    try:
        rows = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, int(count))
    except Exception:
        rows = None
    return serialize_rates(rows)


def quote_snapshot(tick):
    if tick is None:
        return None
    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    tick_time = int(getattr(tick, "time", 0) or 0)
    price = bid if bid > 0.0 else ask
    if price <= 0.0 or tick_time <= 0:
        return None
    return {
        "price": float(price),
        "time": int(tick_time),
    }


def server_offset_seconds(quote, fallback_now_epoch):
    if not isinstance(quote, dict):
        return 0
    quote_time = int(quote.get("time", 0) or 0)
    if quote_time <= 0:
        return 0
    raw_offset = int(quote_time - int(fallback_now_epoch))
    return int(round(raw_offset / 60.0) * 60)


def overlay_quote(rows, tf_secs, start_epoch, end_epoch, quote):
    if not isinstance(quote, dict):
        return rows

    quote_time = int(quote.get("time", 0) or 0)
    quote_price = float(quote.get("price", 0.0) or 0.0)
    if quote_time <= 0 or quote_price <= 0.0:
        return rows

    bucket = (quote_time // tf_secs) * tf_secs
    if bucket < start_epoch or bucket >= end_epoch:
        return rows

    if rows and int(rows[-1]["time"]) == bucket:
        rows[-1]["high"] = max(float(rows[-1]["high"]), quote_price)
        rows[-1]["low"] = min(float(rows[-1]["low"]), quote_price)
        rows[-1]["close"] = quote_price
        rows[-1]["tick_volume"] = max(1.0, float(rows[-1]["tick_volume"]))
        return rows

    open_px = float(rows[-1]["close"]) if rows else quote_price
    rows.append(
        {
            "time": int(bucket),
            "open": open_px,
            "high": max(open_px, quote_price),
            "low": min(open_px, quote_price),
            "close": quote_price,
            "tick_volume": 1.0,
        }
    )
    return rows


try:
    req = json.loads(os.environ.get("VISION_LLM_CHART_REQ", "{}"))
    date_time = parse_dt(req.get("date_time"))
    symbol = str(req.get("symbol", "EURUSD") or "EURUSD").strip().upper()
    timeframe = str(req.get("timeframe", "H1") or "H1").strip().upper()
    bars = max(20, min(int(req.get("bars", 240) or 240), 2000))
    timeout_ms = max(30000, int(req.get("timeout_ms", 120000) or 120000))
    tf_secs = int(TF_SECONDS.get(timeframe, 3600))
    request_epoch = int(date_time.replace(tzinfo=timezone.utc).timestamp()) if date_time.tzinfo is None else int(
        date_time.astimezone(timezone.utc).timestamp()
    )

    mt5 = MetaTrader5(host="localhost", port=8001)
    if not mt5.initialize(timeout=timeout_ms):
        payload["error"] = f"initialize_failed:{mt5.last_error()!r}"
    else:
        account = mt5.account_info()
        if account is not None:
            payload["server"] = str(getattr(account, "server", "") or "")
            try:
                payload["login"] = str(int(getattr(account, "login", 0) or 0))
            except Exception:
                payload["login"] = str(getattr(account, "login", "") or "")
        quote = quote_snapshot(mt5.symbol_info_tick(symbol))
        time_offset_sec = server_offset_seconds(quote, request_epoch)
        broker_epoch = int(request_epoch + time_offset_sec)
        aligned_epoch = broker_epoch - (broker_epoch % tf_secs)
        end_utc = datetime.fromtimestamp(aligned_epoch, tz=timezone.utc) + timedelta(seconds=tf_secs)
        start_utc = end_utc - timedelta(seconds=tf_secs * bars)
        timeframe_value = getattr(mt5, f"TIMEFRAME_{timeframe}", None)
        if timeframe_value is None:
            payload["error"] = f"unsupported_timeframe:{timeframe}"
        else:
            start_epoch = int(start_utc.timestamp())
            end_epoch = int(end_utc.timestamp())
            rates = mt5.copy_rates_range(symbol, timeframe_value, start_utc, end_utc)
            rows = serialize_rates(rates)

            quote_bucket = (
                (int(quote.get("time", 0)) // tf_secs) * tf_secs
                if isinstance(quote, dict)
                else 0
            )
            last_row_time = int(rows[-1]["time"]) if rows else 0
            need_fresh_tail = bool(
                not rows
                or (quote_bucket > 0 and last_row_time < quote_bucket - max(60, tf_secs))
            )

            full_m1_count = int(((end_epoch - start_epoch) // 60) + 8)
            if timeframe in {"M1", "M5", "M15"} and full_m1_count <= 20000:
                m1_rows = fetch_m1_rates_from_pos(mt5, symbol, max(120, full_m1_count))
                agg_rows = aggregate_rates(m1_rows, tf_secs, start_epoch, end_epoch)
                if agg_rows:
                    rows = agg_rows
                    last_row_time = int(rows[-1]["time"])
                    need_fresh_tail = bool(
                        quote_bucket > 0 and last_row_time < quote_bucket - max(60, tf_secs)
                    )

            if need_fresh_tail:
                tail_start_epoch = max(
                    start_epoch,
                    last_row_time if last_row_time > 0 else end_epoch - min(end_epoch - start_epoch, 6 * 3600),
                )
                needed_m1_count = min(
                    20000,
                    max(120, int(((end_epoch - tail_start_epoch) // 60) + 16)),
                )
                m1_rows = fetch_m1_rates_from_pos(mt5, symbol, needed_m1_count)
                tail_rows = aggregate_rates(m1_rows, tf_secs, tail_start_epoch, end_epoch)
                if tail_rows:
                    tail_first_time = int(tail_rows[0]["time"])
                    rows = [row for row in rows if int(row["time"]) < tail_first_time] + tail_rows

            rows = overlay_quote(rows, tf_secs, start_epoch, end_epoch, quote)
            payload["rates"] = rows[-bars:]
            if payload["rates"]:
                payload["resolved_from_time"] = datetime.fromtimestamp(
                    int(payload["rates"][0]["time"]), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                payload["resolved_to_time"] = datetime.fromtimestamp(
                    int(payload["rates"][-1]["time"]), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
            else:
                payload["error"] = f"no_rates_data:{symbol}:{timeframe}:{start_utc.strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        mt5.shutdown()
    except Exception:
        pass
except Exception as exc:
    payload["error"] = f"exception:{type(exc).__name__}:{exc}"

print(json.dumps(payload, ensure_ascii=False))
""".strip()
def _coerce_utc_datetime(raw: datetime) -> datetime:
    if not isinstance(raw, datetime):
        raise ValueError("date_time must be datetime")
    if raw.tzinfo is None:
        return raw.replace(tzinfo=timezone.utc)
    return raw.astimezone(timezone.utc)


def _rates_to_dataframe(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        raise NoMarketDataError("No chart rates returned")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    df.set_index("time", inplace=True)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        },
        inplace=True,
    )
    return df


def _render_chart(df: pd.DataFrame) -> str:
    def _fmt_time(x, _pos=None):
        if x < 0 or x >= len(df):
            return ""
        return df.index[int(x)].strftime("%H:%M")

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=_CHART_STYLE,
        volume=False,
        show_nontrading=False,
        tight_layout=True,
        figratio=(16, 9),
        returnfig=True,
    )

    ax = axes[0]
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_time))
    if len(df) > 1:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    else:
        ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_chart_result_from_payload(
    *,
    chart_rates: list[dict] | None,
    resolved_bar_time: str | None,
    bot_config_id: str | None,
    source_server: str | None,
    source_login: str | None,
) -> ChartImageResult | None:
    if not chart_rates:
        return None

    df = _rates_to_dataframe(chart_rates)
    label_parts = ["bot payload"]
    server_name = str(source_server or "").strip()
    login_id = str(source_login or "").strip()
    bot_id = str(bot_config_id or "").strip()
    if server_name:
        label_parts.append(server_name)
    if login_id:
        label_parts.append(f"login {login_id}")
    if bot_id:
        label_parts.append(f"bot {bot_id[:8]}")
    resolved_bar = str(resolved_bar_time or "").strip()
    if not resolved_bar and len(df.index) > 0:
        resolved_bar = df.index[0].strftime("%Y-%m-%d %H:%M:%S")
    return ChartImageResult(
        image_base64=_render_chart(df),
        source_mode="bot_runtime_payload",
        source_label=" | ".join(label_parts),
        resolved_bar_time=resolved_bar,
    )


def _fetch_rates_from_bot_container(
    date_time: datetime,
    *,
    symbol: str,
    timeframe: str,
    bot_config_id: str,
    timeout_sec: int = 150,
) -> tuple[object, str, str, str]:
    try:
        ref = resolve_bot_runtime_container(instance_name=bot_config_id)
    except BotRunnerError as exc:
        raise MT5ConnectionError(str(exc)) from exc
    if not ref.container_id:
        raise MT5ConnectionError(f"bot runtime container not found for {bot_config_id}")

    request_payload = json.dumps(
        {
            "date_time": _coerce_utc_datetime(date_time).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": str(symbol or "EURUSD").strip().upper(),
            "timeframe": str(timeframe or "H1").strip().upper(),
            "timeout_ms": 120000,
        },
        ensure_ascii=False,
    )
    try:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-e",
                f"VISION_LLM_CHART_REQ={request_payload}",
                ref.container_id,
                "/shared-pydeps/venv-py311/bin/python",
                "-c",
                _DOCKER_FETCH_RATES_SCRIPT,
            ],
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        raise MT5ConnectionError(
            f"bot runtime chart fetch timed out for {bot_config_id} after {timeout_sec}s"
        ) from exc
    except OSError as exc:
        raise MT5ConnectionError(f"bot runtime chart fetch failed to start: {exc}") from exc

    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"exit_{proc.returncode}"
        raise MT5ConnectionError(f"bot runtime chart fetch failed: {detail}")
    if not stdout:
        raise MT5ConnectionError("bot runtime chart fetch returned empty stdout")

    raw_json = stdout.splitlines()[-1]
    try:
        payload = json.loads(raw_json)
    except Exception as exc:
        detail = stderr or stdout[-400:]
        raise MT5ConnectionError(f"bot runtime chart fetch returned invalid JSON: {detail}") from exc

    error = str(payload.get("error", "") or "").strip()
    if error:
        if error.startswith("no_m1_data:"):
            raise NoMarketDataError(error)
        raise MT5ConnectionError(error)

    rates = payload.get("rates")
    if not rates:
        raise NoMarketDataError("bot runtime chart fetch returned no rates")

    server_name = str(payload.get("server", "") or "").strip()
    login_id = str(payload.get("login", "") or "").strip()
    resolved_bar_time = str(payload.get("resolved_bar_time", "") or "").strip()
    label_parts = [f"bot runtime {bot_config_id[:8]}"]
    if server_name:
        label_parts.append(server_name)
    if login_id:
        label_parts.append(f"login {login_id}")
    return rates, "bot_runtime", " | ".join(label_parts) or "bot runtime", resolved_bar_time


def _fetch_timeframe_rates_from_bot_container(
    date_time: datetime,
    *,
    symbol: str,
    timeframe: str,
    bot_config_id: str,
    bars: int = 240,
    timeout_sec: int = 150,
) -> ChartRatesResult:
    try:
        ref = resolve_bot_runtime_container(instance_name=bot_config_id)
    except BotRunnerError as exc:
        raise MT5ConnectionError(str(exc)) from exc
    if not ref.container_id:
        raise MT5ConnectionError(f"bot runtime container not found for {bot_config_id}")

    safe_bars = max(20, min(int(bars or 240), _MAX_CHART_FETCH_BARS))
    request_payload = json.dumps(
        {
            "date_time": _coerce_utc_datetime(date_time).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": str(symbol or "EURUSD").strip().upper(),
            "timeframe": str(timeframe or "H1").strip().upper(),
            "bars": safe_bars,
            "timeout_ms": 120000,
        },
        ensure_ascii=False,
    )
    try:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-e",
                f"VISION_LLM_CHART_REQ={request_payload}",
                ref.container_id,
                "/shared-pydeps/venv-py311/bin/python",
                "-c",
                _DOCKER_FETCH_TIMEFRAME_RATES_SCRIPT,
            ],
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired as exc:
        raise MT5ConnectionError(
            f"bot runtime chart fetch timed out for {bot_config_id} after {timeout_sec}s"
        ) from exc
    except OSError as exc:
        raise MT5ConnectionError(f"bot runtime chart fetch failed to start: {exc}") from exc

    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"exit_{proc.returncode}"
        raise MT5ConnectionError(f"bot runtime chart fetch failed: {detail}")
    if not stdout:
        raise MT5ConnectionError("bot runtime chart fetch returned empty stdout")

    raw_json = stdout.splitlines()[-1]
    try:
        payload = json.loads(raw_json)
    except Exception as exc:
        detail = stderr or stdout[-400:]
        raise MT5ConnectionError(f"bot runtime chart fetch returned invalid JSON: {detail}") from exc

    error = str(payload.get("error", "") or "").strip()
    if error:
        if error.startswith("no_rates_data:"):
            raise NoMarketDataError(error)
        raise MT5ConnectionError(error)

    rates = payload.get("rates")
    if not rates:
        raise NoMarketDataError("bot runtime chart fetch returned no rates")

    server_name = str(payload.get("server", "") or "").strip()
    login_id = str(payload.get("login", "") or "").strip()
    label_parts = [f"bot runtime {bot_config_id[:8]}"]
    if server_name:
        label_parts.append(server_name)
    if login_id:
        label_parts.append(f"login {login_id}")
    return ChartRatesResult(
        rates=list(rates),
        source_mode="bot_runtime",
        source_label=" | ".join(label_parts) or "bot runtime",
        resolved_from_time=str(payload.get("resolved_from_time", "") or "").strip(),
        resolved_to_time=str(payload.get("resolved_to_time", "") or "").strip(),
    )


def fetch_chart_rates_result(
    date_time: datetime,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    bot_config_id: str | None = None,
    bars: int = 240,
) -> ChartRatesResult:
    symbol_text = str(symbol or "EURUSD").strip().upper()
    timeframe_text = str(timeframe or "H1").strip().upper()
    bot_id = str(bot_config_id or "").strip()
    safe_bars = max(20, min(int(bars or 240), _MAX_CHART_FETCH_BARS))
    if not bot_id:
        raise ValueError("bot_config_id is required to load chart candles from bot runtime")
    return _fetch_timeframe_rates_from_bot_container(
        date_time,
        symbol=symbol_text,
        timeframe=timeframe_text,
        bot_config_id=bot_id,
        bars=safe_bars,
    )


def generate_image_result(
    date_time: datetime,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    bot_config_id: str | None = None,
    chart_rates: list[dict] | None = None,
    resolved_bar_time: str | None = None,
    source_server: str | None = None,
    source_login: str | None = None,
) -> ChartImageResult:
    symbol_text = str(symbol or "EURUSD").strip().upper()
    timeframe_text = str(timeframe or "H1").strip().upper()
    bot_id = str(bot_config_id or "").strip()

    payload_result = _build_chart_result_from_payload(
        chart_rates=chart_rates,
        resolved_bar_time=resolved_bar_time,
        bot_config_id=bot_id,
        source_server=source_server,
        source_login=source_login,
    )
    if payload_result is not None:
        return payload_result

    if not bot_id:
        raise ValueError("bot_config_id or chart_rates is required to generate a chart image")

    rates, source_mode, source_label, resolved_bar_time = _fetch_rates_from_bot_container(
        date_time,
        symbol=symbol_text,
        timeframe=timeframe_text,
        bot_config_id=bot_id,
    )
    df = _rates_to_dataframe(rates)
    return ChartImageResult(
        image_base64=_render_chart(df),
        source_mode=source_mode,
        source_label=source_label,
        resolved_bar_time=resolved_bar_time,
    )


def generate_image(
    date_time: datetime,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    bot_config_id: str | None = None,
) -> str:
    return generate_image_result(
        date_time=date_time,
        symbol=symbol,
        timeframe=timeframe,
        bot_config_id=bot_config_id,
    ).image_base64
