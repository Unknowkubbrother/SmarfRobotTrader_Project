from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import os

from .routes import authentication
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
]


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
            
        if request.url.path in PUBLIC_PATHS:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(authentication.auth_router, prefix="/auth")

# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
# cloudflared tunnel --url http://localhost:8000