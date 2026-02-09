from fastapi import APIRouter, HTTPException, Depends, status, Response, Request, Form
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
from ..models.authentication_model import Register_OTP, Register_Verify, Register_Complete, Login_Verify, Google_Register_OTP, Google_Register_Verify, Google_Register_Complete
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


@auth_router.post("/login", tags=["auth"])
async def login(
    response: Response,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()]
):
    user = await db.user.find_unique(where={"email": username})
    
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.status == "banned":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been banned",
        )

    if not user.recoveryEmail:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Account security incomplete: No recovery email set. Contact support or use forgot password to recover/set it."
        )

    otp = str(random_with_N_digits(6))
    
    r_cache.setex(f"login_pending:{user.email}", 300, otp)
    
    print(f"[DEV] Login OTP for {user.email} (sent to {user.recoveryEmail}): {otp}")

    return {
        "require_otp": True,
        "message": "OTP sent to your recovery email",
        "email": user.email,
        "recovery_email_hint": f"{user.recoveryEmail[:4]}****@{user.recoveryEmail.split('@')[1]}",
        "dev_otp": otp
    }


@auth_router.post("/login/verify", response_model=Token, tags=["auth"])
async def login_verify(response: Response, data: Login_Verify):
    email = data.email
    otp = data.otp
    
    stored_otp = r_cache.get(f"login_pending:{email}")
    
    if not stored_otp or stored_otp != otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    user = await db.user.find_unique(where={"email": email})
    if not user:
         raise HTTPException(status_code=404, detail="User not found")

    # Clear OTP
    r_cache.delete(f"login_pending:{email}")
    
    # Issue Token
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


@auth_router.post("/google", tags=["auth"])
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
        return {
            "require_register": True,
            "message": "New account. Please complete registration.",
            "google_info": {
                "email": email,
                "name": name,
                "picture": picture,
                "uid": google_uid
            }
        }
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
        
        if not user.recoveryEmail:
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Account security incomplete: No recovery email set. Contact support."
            )
        otp = str(random_with_N_digits(6))
        r_cache.setex(f"login_pending:{user.email}", 300, otp)
        
        print(f"[DEV] Google Login OTP for {user.email}: {otp}")
        
        return {
            "require_otp": True,
            "message": "OTP sent to your recovery email",
            "email": user.email,
            "recovery_email_hint": f"{user.recoveryEmail[:4]}****@{user.recoveryEmail.split('@')[1]}",
             "dev_otp": otp
        }


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


@auth_router.post("/forgot-password/request", tags=["auth"])
async def forgot_password_request(data: dict):
    email = data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    
    user = await db.user.find_unique(where={"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="Email not found in system")
    
    if not user.recoveryEmail:
        raise HTTPException(status_code=400, detail="No recovery email found for this account")
    
    otp = random_with_N_digits(6)
    
    r_cache.setex(
        f"forgot_password_otp:{email}",
        300,
        otp
    )
    
    return {
        "message": f"OTP sent to recovery email",
        "recovery_email_hint": f"{user.recoveryEmail[:4]}****@{user.recoveryEmail.split('@')[1]}",
        "dev_otp": otp
    }


@auth_router.post("/forgot-password/verify", tags=["auth"])
async def forgot_password_verify(data: dict):
    email = data.get("email")
    otp = data.get("otp")
    
    if not email or not otp:
        raise HTTPException(status_code=400, detail="Email and OTP are required")
    
    stored_otp = r_cache.get(f"forgot_password_otp:{email}")
    if not stored_otp:
        raise HTTPException(status_code=400, detail="OTP expired or not found")
    
    if stored_otp != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    return {"message": "OTP verified", "verified": True}


@auth_router.post("/forgot-password/reset", tags=["auth"])
async def forgot_password_reset(data: dict):
    email = data.get("email")
    otp = data.get("otp")
    new_password = data.get("new_password")
    
    if not email or not otp or not new_password:
        raise HTTPException(status_code=400, detail="Email, OTP, and new password are required")
    
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    stored_otp = r_cache.get(f"forgot_password_otp:{email}")
    if not stored_otp:
        raise HTTPException(status_code=400, detail="OTP expired or not found")
    
    if stored_otp != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    user = await db.user.find_unique(where={"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    hashed_password = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
    
    await db.user.update(
        where={"id": user.id},
        data={"password": hashed_password.decode("utf-8")}
    )
    
    r_cache.delete(f"forgot_password_otp:{email}")
    
    return {"message": "Password reset successfully"}


@auth_router.post("/google/register/otp", tags=["auth"])
async def google_register_otp(data: Google_Register_OTP):
    try:
        decoded_token = firebase_auth.verify_id_token(data.id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")
    
    email = decoded_token.get("email")
    name = decoded_token.get("name", email.split("@")[0])
    picture = decoded_token.get("picture")
    google_uid = decoded_token.get("uid")
    
    if not email:
        raise HTTPException(status_code=400, detail="Email not found in Google token")
    
    # Check if recovery email matches account email
    if data.recovery_email.lower() == email.lower():
         raise HTTPException(status_code=400, detail="Recovery email must be different from your Google email")

    otp = str(random_with_N_digits(6))
    
    # Cache Registration Info
    reg_key = f"google_reg_pending:{email}"
    r_cache.hset(reg_key, mapping={
        "email": email,
        "name": name,
        "picture": picture if picture else "",
        "google_uid": google_uid,
        "recovery_email": data.recovery_email,
        "otp": otp
    })
    r_cache.expire(reg_key, 600) # 10 minutes
    
    print(f"[DEV] Google Register OTP for {email} (sent to {data.recovery_email}): {otp}")
    
    return {
        "message": "OTP sent to recovery email",
        "dev_otp": otp
    }


@auth_router.post("/google/register/verify", tags=["auth"])
async def google_register_verify(data: Google_Register_Verify):
    email = data.email
    otp = data.otp
    
    reg_key = f"google_reg_pending:{email}"
    stored_data = r_cache.hgetall(reg_key)
    
    if not stored_data or stored_data.get("otp") != otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    # Allow proceeding to next step (Completion)
    # We could issue a temporary signed token, but for now relying on cache existence and passing email is ok for slight simplicity, 
    # but strictly we should ensure the next step verifies this verification happened.
    # To secure it, we update the cache to mark as 'verified'
    r_cache.hset(reg_key, "verified", "true")
    
    return {"message": "OTP verified", "verified": True}


@auth_router.post("/google/register/complete", response_model=Token, tags=["auth"])
async def google_register_complete(response: Response, data: Google_Register_Complete):
    email = data.email
    username = data.username
    
    reg_key = f"google_reg_pending:{email}"
    stored_data = r_cache.hgetall(reg_key)
    
    if not stored_data:
        raise HTTPException(status_code=400, detail="Registration session expired. Please try again.")
    
    if stored_data.get("verified") != "true":
         raise HTTPException(status_code=400, detail="Email not verified")

    try:
        user = await db.user.create(
            data={
                "username": username,
                "email": email,
                "recoveryEmail": stored_data.get("recovery_email"),
                "googleAuthId": stored_data.get("google_uid"),
                "avatarUrl": stored_data.get("picture") or None,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
        
    # Clear Cache
    r_cache.delete(reg_key)
    
    # Issue Token
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
