"""
Chart generation module — fetches OHLC data from MT5 and renders
candlestick charts as base64-encoded PNG images.
"""

import os
import base64
from io import BytesIO
from datetime import datetime, timedelta, timezone

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf                # noqa: E402
import matplotlib.ticker as ticker      # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from mt5linux import MetaTrader5
from dotenv import load_dotenv

load_dotenv()

# ── MT5 connection defaults ──────────────────────────────────────────
_MT5_HOST = os.getenv("MT5_HOST", "localhost")
_MT5_PORT = int(os.getenv("MT5_PORT", 8001))

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


def generate_image(date_time: datetime, symbol: str = "EURUSD") -> str:
    """Return a base64-encoded PNG candlestick chart for one 1-hour bar.

    Parameters
    ----------
    date_time : datetime
        The target hour (minutes/seconds are ignored).
    symbol : str
        MT5 symbol name.

    Returns
    -------
    str
        Base64-encoded PNG image.
    """
    mt5 = MetaTrader5(host=_MT5_HOST, port=_MT5_PORT)
    mt5.initialize()

    try:
        start = date_time.replace(minute=0, second=0, microsecond=0)
        start_utc = (
            start.replace(tzinfo=timezone.utc)
            if start.tzinfo is None
            else start.astimezone(timezone.utc)
        )
        end_utc = start_utc + timedelta(hours=1)

        rates = mt5.copy_rates_range(
            symbol, mt5.TIMEFRAME_M1, start_utc, end_utc,
        )
        if rates is None or len(rates) == 0:
            raise NoMarketDataError(f"No M1 data for {symbol} at {date_time}")

        # Build OHLCV DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close",
                "tick_volume": "Volume",
            },
            inplace=True,
        )

        # Render chart
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

        # Export to base64
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    finally:
        mt5.shutdown()
