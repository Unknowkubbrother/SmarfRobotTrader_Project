from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
    "/redoc",
    "/auth/login",
    "/auth/logout",
    "/auth/register/otp",
    "/auth/register/verify_otp",
    "/auth/register/complete",
]


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
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
app.add_middleware(AuthMiddleware)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/profile")
async def profile(request: Request):
    user_id = request.state.user_id
    email = request.state.email
    role = request.state.role
    return {"user_id": user_id, "email": email, "role": role}

app.include_router(authentication.auth_router, prefix="/auth")

# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload