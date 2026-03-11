"""
Chart generation module — fetches OHLC data from MT5 and renders
candlestick charts as base64-encoded PNG images.
"""

import base64
import json
import os
import socket
import subprocess
from dataclasses import dataclass
from io import BytesIO
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf                # noqa: E402
import matplotlib.ticker as ticker      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from mt5linux import MetaTrader5
from dotenv import load_dotenv

from ..mt5_bot_runner import BotRunnerError, resolve_bot_runtime_container

load_dotenv()

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


def _clean_env_text(raw: str | None) -> str:
    text = str(raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def _coerce_port(raw: str | None, default: int) -> int:
    text = _clean_env_text(raw)
    if not text:
        return int(default)
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return int(default)


def _parse_mt5_endpoint() -> tuple[str, int]:
    default_host = "localhost"
    default_port = 8001
    host_text = _clean_env_text(os.getenv("MT5_HOST"))
    port = _coerce_port(os.getenv("MT5_PORT"), default_port)

    if not host_text:
        return default_host, port

    parsed = urlparse(host_text if "://" in host_text else f"//{host_text}")
    if parsed.hostname:
        parsed_host = _clean_env_text(parsed.hostname)
        if parsed.port:
            port = int(parsed.port)
        return parsed_host or default_host, port

    return _clean_env_text(host_text) or default_host, port


def _candidate_hosts(configured_host: str) -> list[str]:
    candidates: list[str] = []
    for host in (configured_host, "localhost", "127.0.0.1"):
        normalized = _clean_env_text(host)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _connect_mt5() -> tuple[MetaTrader5, str, int]:
    host, port = _parse_mt5_endpoint()
    attempts: list[str] = []

    for candidate in _candidate_hosts(host):
        try:
            mt5 = MetaTrader5(host=candidate, port=port)
            if mt5.initialize():
                return mt5, candidate, port

            detail = ""
            try:
                detail = f" last_error={mt5.last_error()!r}"
            except Exception:
                pass
            attempts.append(f"{candidate}:{port} initialize=False{detail}")
            try:
                mt5.shutdown()
            except Exception:
                pass
        except (socket.gaierror, OSError, TimeoutError, ConnectionError) as exc:
            attempts.append(f"{candidate}:{port} {type(exc).__name__}: {exc}")

    tried_hosts = ", ".join(_candidate_hosts(host))
    detail = " | ".join(attempts) if attempts else "no attempts recorded"
    raise MT5ConnectionError(
        f"Unable to connect to MT5 bridge (hosts={tried_hosts}, port={port}). {detail}"
    )


_TF_SECONDS = {
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


def _timeframe_seconds(timeframe: str) -> int:
    tf = str(timeframe or "H1").strip().upper()
    return int(_TF_SECONDS.get(tf, 3600))


def _coerce_utc_datetime(raw: datetime) -> datetime:
    if not isinstance(raw, datetime):
        raise ValueError("date_time must be datetime")
    if raw.tzinfo is None:
        return raw.replace(tzinfo=timezone.utc)
    return raw.astimezone(timezone.utc)


def _aligned_bar_window(date_time: datetime, timeframe: str) -> tuple[datetime, datetime]:
    tf_secs = _timeframe_seconds(timeframe)
    epoch = int(_coerce_utc_datetime(date_time).timestamp())
    aligned_epoch = epoch - (epoch % tf_secs)
    start_utc = datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)
    end_utc = start_utc + timedelta(seconds=tf_secs)
    return start_utc, end_utc


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
    return rates, "bot_runtime_mt5", " | ".join(label_parts) or "bot runtime MT5", resolved_bar_time


def _fetch_rates_from_server_mt5(
    date_time: datetime,
    *,
    symbol: str,
    timeframe: str,
) -> tuple[object, str, str, str]:
    mt5, connected_host, connected_port = _connect_mt5()
    start_utc, end_utc = _aligned_bar_window(date_time, timeframe)

    try:
        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M1,
            start_utc,
            end_utc,
        )
        if rates is None or len(rates) == 0:
            raise NoMarketDataError(f"No M1 data for {symbol} at {date_time}")
        return (
            rates,
            "server_mt5",
            f"server MT5 {connected_host}:{connected_port}",
            start_utc.strftime("%Y-%m-%d %H:%M:%S"),
        )
    finally:
        mt5.shutdown()


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

    if bot_id:
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

    rates, source_mode, source_label, resolved_bar_time = _fetch_rates_from_server_mt5(
        date_time,
        symbol=symbol_text,
        timeframe=timeframe_text,
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
