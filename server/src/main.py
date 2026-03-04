from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import os
from fastapi.staticfiles import StaticFiles

from .routes import authentication, bot, bot_ws, trading, settings, notification, search, subscription, admin, support, vision_llm
from .database.client import db

SECRET_KEY = os.getenv("JWT_SECRET", "UknownmeInLove")
ALGORITHM = "HS256"

PUBLIC_PATHS = [
    "/",
    "/docs",
    "/openapi.json",
    "/auth/login",
    "/auth/logout",
    "/auth/google",
    "/auth/register/otp",
    "/auth/register/verify_otp",
    "/auth/register/complete",
    "/auth/forgot-password/request",
    "/auth/forgot-password/verify",
    "/auth/forgot-password/reset",
    "/auth/login/verify",
    "/auth/google/register/otp",
    "/auth/google/register/verify",
    "/auth/google/register/complete",
    "/auth/check-user",
    "/auth/login/otp-init",

    # TEST VISION LLM
    "/vision_llm/",

    # Bot WebSocket + Cron
    "/bot/ws/cron",
]


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
            
        if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/static"):
            return await call_next(request)
        
        token = request.cookies.get("access_token")
        if not token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        
        if not token:
            return JSONResponse(
                status_code=401, 
                content={"detail": "Not authenticated"}
            )
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            request.state.user_id = payload.get("sub")
            request.state.email = payload.get("email")
            request.state.role = payload.get("role")
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401, 
                content={"detail": "Token expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401, 
                content={"detail": "Invalid token"}
            )
        
        return await call_next(request)

import asyncio
import logging
from datetime import datetime, timezone

from .routes.bot_ws import _get_cached, _set_cached
from .utils.vision_llm.use_llm import generate_llm_cls_for_bar
from .utils.vision_llm.chart import MT5ConnectionError, NoMarketDataError
from .utils.ws_manager import bot_hub

logger = logging.getLogger(__name__)

# Symbol:Timeframe pairs, comma-separated  (e.g. "EURUSD:H1,EURUSD:M15,USDJPY:H1")
_raw_pairs = os.getenv("CRON_PAIRS", "EURUSD:H1")
CRON_PAIRS: list[tuple[str, str]] = []
for p in _raw_pairs.split(","):
    p = p.strip()
    if ":" in p:
        sym, tf = p.split(":", 1)
        CRON_PAIRS.append((sym.strip().upper(), tf.strip().upper()))
CRON_ENABLED = os.getenv("CRON_ENABLED", "1").strip().lower() in {"1", "true", "yes"}

# Timeframe → seconds
_TF_SECS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


async def _cron_worker(symbol: str, timeframe: str):
    """Background loop: compute + broadcast vision_llm for one symbol/timeframe."""
    import time as _time
    interval = _TF_SECS.get(timeframe, 3600)
    logger.info("cron_worker started  | %s/%s  interval=%ds", symbol, timeframe, interval)

    while True:
        # Sleep until next candle close + 5s buffer
        now = datetime.now(timezone.utc)
        epoch = int(now.timestamp())
        seconds_to_next = interval - (epoch % interval) + 5
        logger.info("cron  💤  %s/%s  sleeping %ds", symbol, timeframe, seconds_to_next)
        await asyncio.sleep(seconds_to_next)

        now = datetime.now(timezone.utc)
        # Align to candle boundary
        epoch = int(now.timestamp())
        aligned = epoch - (epoch % interval)
        candle_dt = datetime.fromtimestamp(aligned, tz=timezone.utc)
        dt_str = candle_dt.strftime("%Y-%m-%d %H:%M:%S")

        try:
            cached = _get_cached(symbol, timeframe, dt_str)
            if cached:
                result_data = cached
                logger.info("cron  ⚡  cache hit %s/%s  %s", symbol, timeframe, dt_str)
            else:
                start = _time.perf_counter()
                result, cls_vec = await asyncio.to_thread(
                    generate_llm_cls_for_bar, candle_dt, symbol,
                )
                elapsed = _time.perf_counter() - start
                result_data = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "date_time": dt_str,
                    "llm_text": result,
                    "cls_vec": cls_vec.tolist(),
                    "elapsed_seconds": round(elapsed, 2),
                }
                _set_cached(symbol, timeframe, dt_str, result_data)
                logger.info("cron  ✔  %s/%s  %s  %.1fs", symbol, timeframe, dt_str, elapsed)

            await bot_hub.broadcast_llm(symbol, timeframe, result_data)
        except NoMarketDataError as exc:
            logger.warning(
                "cron skip %s/%s at %s: %s",
                symbol,
                timeframe,
                dt_str,
                exc,
            )
        except MT5ConnectionError as exc:
            logger.warning(
                "cron mt5 unavailable %s/%s at %s: %s",
                symbol,
                timeframe,
                dt_str,
                exc,
            )
        except Exception as exc:
            logger.exception("cron failed %s/%s: %s", symbol, timeframe, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()

    # Start one cron worker per symbol:timeframe pair
    cron_tasks = []
    if CRON_ENABLED and CRON_PAIRS:
        for symbol, timeframe in CRON_PAIRS:
            task = asyncio.create_task(_cron_worker(symbol, timeframe))
            cron_tasks.append(task)

    yield

    for task in cron_tasks:
        task.cancel()
    await db.disconnect()


app = FastAPI(lifespan=lifespan)

# Mount static files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
app.mount("/static", StaticFiles(directory=UPLOADS_DIR), name="static")


_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(authentication.auth_router, prefix="/auth")
app.include_router(bot.bot_router, prefix="/bot")
app.include_router(trading.trading_router, prefix="/trading")
app.include_router(settings.settings_router, prefix="/settings")
app.include_router(notification.notification_router, prefix="/notifications")
app.include_router(search.search_router, prefix="/search")
app.include_router(subscription.subscription_router, prefix="/subscription")
app.include_router(admin.admin_router, prefix="/admin")
app.include_router(support.support_router, prefix="/support")
app.include_router(vision_llm.vision_llm_router, prefix="/vision_llm")
app.include_router(bot_ws.bot_ws_router, prefix="/bot")

# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
# cloudflared tunnel --url http://localhost:8000
# prisma db push --schema=src/database/schema.prisma
# prisma generate --schema=src/database/schema.prisma

# python -m prisma validate --schema=src/database/schema.prisma
# echo CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; | python -m prisma db execute --stdin --schema=src/database/schema.prisma
# python -m prisma db push --schema=src/database/schema.prisma --skip-generate
# python -m prisma generate --schema=src/database/schema.prisma
