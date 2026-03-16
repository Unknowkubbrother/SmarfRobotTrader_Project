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
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

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

    # Stripe webhooks
    "/subscription/stripe/webhook",
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


def _normalize_origin(value: str | None) -> str | None:
    text = str(value or "").strip().strip("'\"").rstrip("/")
    if not text:
        return None
    return text


def _resolve_cors_origins() -> list[str]:
    origins: list[str] = []

    def add(value: str | None):
        origin = _normalize_origin(value)
        if origin and origin not in origins:
            origins.append(origin)

    for raw_origin in str(os.getenv("CORS_ALLOWED_ORIGINS", "") or "").split(","):
        add(raw_origin)

    for env_name in ("FRONTEND_URL", "APP_URL", "NEXT_PUBLIC_APP_URL"):
        add(os.getenv(env_name))

    for default_origin in _DEFAULT_CORS_ORIGINS:
        add(default_origin)

    return origins

import asyncio
import logging

from .utils.notification_runtime import (
    run_summary_notification_worker,
    run_threshold_notification_worker,
)
from .utils.subscription_billing import run_subscription_billing_worker

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()

    background_tasks = []
    background_tasks.append(asyncio.create_task(run_threshold_notification_worker()))
    background_tasks.append(asyncio.create_task(run_summary_notification_worker()))
    background_tasks.append(asyncio.create_task(run_subscription_billing_worker()))

    yield

    for task in background_tasks:
        task.cancel()
    await db.disconnect()


app = FastAPI(lifespan=lifespan)

# Mount static files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
app.mount("/static", StaticFiles(directory=UPLOADS_DIR), name="static")


_cors_origins = _resolve_cors_origins()

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
