from fastapi import APIRouter, HTTPException, Depends, status, Response, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional
from pydantic import BaseModel
import jwt
from jwt.exceptions import InvalidTokenError
import bcrypt
import os
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

from ..database.client import r_cache, db
from ..models.authentication_model import Register_OTP, Register_Verify, Register_Complete
from lib.untils import random_with_N_digits

SECRET_KEY = os.getenv("JWT_SECRET", "UknownmeInLove")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
COOKIE_NAME = "access_token"

cred = credentials.Certificate("smarfrobottrade-firebase.json")
firebase_admin.initialize_app(cred)

auth_router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: str | None = None
    email: str | None = None
    role: str | None = None


class GoogleAuth(BaseModel):
    id_token: str


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    request: Request,
    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_to_use = request.cookies.get(COOKIE_NAME) or token
    
    if not token_to_use:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(token_to_use, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(
            user_id=user_id, 
            email=payload.get("email"),
            role=payload.get("role")
        )
    except InvalidTokenError:
        raise credentials_exception
    
    user = await db.user.find_unique(where={"id": token_data.user_id})
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[any, Depends(get_current_user)],
):
    if current_user.status == "banned":
        raise HTTPException(status_code=400, detail="User is banned")
    return current_user


@auth_router.post("/google", response_model=Token, tags=["auth"])
async def google_auth(response: Response, google_auth: GoogleAuth):
    try:
        decoded_token = firebase_auth.verify_id_token(google_auth.id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")
    
    email = decoded_token.get("email")
    name = decoded_token.get("name", email.split("@")[0])
    picture = decoded_token.get("picture")
    google_uid = decoded_token.get("uid")
    
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in Google token")
    
    user = await db.user.find_unique(where={"email": email})
    
    if not user:
        try:
            user = await db.user.create(
                data={
                    "username": name,
                    "email": email,
                    "googleAuthId": google_uid,
                    "avatarUrl": picture,
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
    else:
        if not user.googleAuthId:
            await db.user.update(
                where={"id": str(user.id)},
                data={
                    "googleAuthId": google_uid,
                    "avatarUrl": picture or user.avatarUrl,
                }
            )
    
    if user.status == "banned":
        raise HTTPException(status_code=403, detail="Your account has been banned")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": user.role
        },
        expires_delta=access_token_expires
    )
    
    response.set_cookie(
        key=COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False
    )
    
    return Token(access_token=access_token, token_type="bearer")


@auth_router.post("/register/otp", tags=["auth"])
async def register_otp(register_otp: Register_OTP):
    existing_user = await db.user.find_unique(where={"email": register_otp.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    recovery_email_key = register_otp.recovery_email.split('@')[0]
    
    hashed_password = bcrypt.hashpw(
        register_otp.password.encode('utf-8'), 
        bcrypt.gensalt()
    ).decode('utf-8')
    
    r_cache.hset(f"reg_detail_{recovery_email_key}", mapping={
        "email": register_otp.email,
        "recovery_email": register_otp.recovery_email,
        "password": hashed_password
    })
    r_cache.expire(f"reg_detail_{recovery_email_key}", 60 * 10)
    
    otp = str(random_with_N_digits(6))
    r_cache.set(f"reg_otp_{recovery_email_key}", otp, ex=60 * 5)
    
    print(f"[DEV] OTP for {register_otp.recovery_email}: {otp}")
    
    return {
        "status_code": 200,
        "message": "OTP sent to your recovery email",
        "dev_otp": otp
    }


@auth_router.post("/register/verify_otp", tags=["auth"])
async def register_verify_otp(register_verify: Register_Verify):
    recovery_email_key = register_verify.recovery_email.split('@')[0]
    
    stored_otp = r_cache.get(f"reg_otp_{recovery_email_key}")
    if not stored_otp:
        raise HTTPException(status_code=400, detail="OTP expired or not found")
    
    if stored_otp != register_verify.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    user_detail = r_cache.hgetall(f"reg_detail_{recovery_email_key}")
    if not user_detail:
        raise HTTPException(status_code=400, detail="Registration data expired. Please start over.")
    
    existing_user = await db.user.find_unique(where={"email": user_detail["email"]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    try:
        user = await db.user.create(
            data={
                "username": user_detail["email"].split("@")[0],
                "email": user_detail["email"],
                "recoveryEmail": user_detail["recovery_email"],
                "password": user_detail["password"],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
        
    r_cache.delete(f"reg_otp_{recovery_email_key}")
    r_cache.set(f"reg_user_{recovery_email_key}", str(user.id), ex=60 * 30)
    
    return {
        "status_code": 201,
        "message": "OTP verified, account created. Please set your username.",
        "user": {
            "id": str(user.id),
            "email": user.email
        }
    }


@auth_router.post("/register/complete", tags=["auth"])
async def register_complete(register_complete: Register_Complete):
    recovery_email_key = register_complete.recovery_email.split('@')[0]
    
    user_id = r_cache.get(f"reg_user_{recovery_email_key}")
    if not user_id:
        raise HTTPException(status_code=400, detail="Session expired. Please login and update username.")
    
    existing_username = await db.user.find_first(where={"username": register_complete.username})
    if existing_username and str(existing_username.id) != user_id:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    try:
        user = await db.user.update(
            where={"id": user_id},
            data={"username": register_complete.username}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update username: {str(e)}")
    
    r_cache.delete(f"reg_user_{recovery_email_key}")
    r_cache.delete(f"reg_detail_{recovery_email_key}")
    
    return {
        "status_code": 200,
        "message": "Registration complete!",
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email
        }
    }


@auth_router.post("/login", response_model=Token, tags=["auth"])
async def login(response: Response, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await db.user.find_unique(where={"email": form_data.username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please login with Google",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not bcrypt.checkpw(form_data.password.encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.status == "banned":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been banned",
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": user.role
        },
        expires_delta=access_token_expires
    )
    
    response.set_cookie(
        key=COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False
    )
    
    return Token(access_token=access_token, token_type="bearer")


@auth_router.post("/logout", tags=["auth"])
async def logout(response: Response):
    response.delete_cookie(key=COOKIE_NAME)
    return {"message": "Logged out successfully"}


@auth_router.get("/me", tags=["auth"])
async def get_me(current_user: Annotated[any, Depends(get_current_active_user)]):
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "status": current_user.status
    }

