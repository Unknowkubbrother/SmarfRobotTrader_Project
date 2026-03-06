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
from ..models.auth_model import Register_OTP, Register_Verify, Register_Complete, Login_Verify, Google_Register_OTP, Google_Register_Verify, Google_Register_Complete, CheckUser_Request, Login_OTP_Init, SetPassword_Request
from ..utils.subscription_access import get_user_subscription_access_state
from lib.untils import random_with_N_digits, send_otp_email
from ..utils.turnstile import verify_turnstile

SECRET_KEY = os.getenv("JWT_SECRET", "UknownmeInLove")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300
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
    
    if not user or not user.password or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
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
    
    send_otp_email(user.recoveryEmail, otp, purpose="login")

    return {
        "require_otp": True,
        "message": "OTP sent to your recovery email",
        "email": user.email,
        "recovery_email_hint": f"{user.recoveryEmail[:4]}****@{user.recoveryEmail.split('@')[1]}"
    }


@auth_router.post("/login/verify", response_model=Token, tags=["auth"])
async def login_verify(response: Response, request: Request, data: Login_Verify):
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
    
    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        
        await db.activitylog.create(
            data={
                "userId": str(user.id),
                "topic": "Login",
                "detail": "User logged in via Email/OTP",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255] # Truncate to fit DB
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

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
        
        send_otp_email(user.recoveryEmail, otp, purpose="login")
        
        return {
            "require_otp": True,
            "message": "OTP sent to your recovery email",
            "email": user.email,
            "recovery_email_hint": f"{user.recoveryEmail[:4]}****@{user.recoveryEmail.split('@')[1]}"
        }


@auth_router.post("/register/otp", tags=["auth"])
async def register_otp(register_otp: Register_OTP):
    if not await verify_turnstile(register_otp.cf_token):
        raise HTTPException(status_code=400, detail="Invalid security token")

    existing_user = await db.user.find_unique(where={"email": register_otp.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    
    hashed_password = bcrypt.hashpw(
        register_otp.password.encode('utf-8'), 
        bcrypt.gensalt()
    ).decode('utf-8')
    
    # Store registration details
    r_cache.hset(f"register_pending:{register_otp.recovery_email}", mapping={
        "email": register_otp.email,
        "recovery_email": register_otp.recovery_email,
        "password": hashed_password
    })
    r_cache.expire(f"register_pending:{register_otp.recovery_email}", 60 * 10)
    
    otp = str(random_with_N_digits(6))
    r_cache.set(f"register_otp:{register_otp.recovery_email}", otp, ex=60 * 5)
    
    send_otp_email(register_otp.recovery_email, otp, purpose="register")
    
    return {
        "status_code": 200,
        "message": "OTP sent to your recovery email"
    }


@auth_router.post("/register/verify_otp", tags=["auth"])
async def register_verify_otp(register_verify: Register_Verify):
    
    stored_otp = r_cache.get(f"register_otp:{register_verify.recovery_email}")
    if not stored_otp:
        raise HTTPException(status_code=400, detail="OTP expired or not found")
    
    if stored_otp != register_verify.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    user_detail = r_cache.hgetall(f"register_pending:{register_verify.recovery_email}")
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
        
    await db.notificationconfig.create(data={"userId": str(user.id)})
        
    r_cache.delete(f"register_otp:{register_verify.recovery_email}")
    r_cache.set(f"register_verified_user_id:{register_verify.recovery_email}", str(user.id), ex=60 * 30)
    
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
    
    user_id = r_cache.get(f"register_verified_user_id:{register_complete.recovery_email}")
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
    
    r_cache.delete(f"register_verified_user_id:{register_complete.recovery_email}")
    r_cache.delete(f"register_pending:{register_complete.recovery_email}")
    
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
async def login(
    response: Response, 
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    cf_token: Annotated[Optional[str], Form()] = None
):
    if not await verify_turnstile(cf_token):
         raise HTTPException(status_code=400, detail="Invalid security token")

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
    access_state = await get_user_subscription_access_state(str(current_user.id))
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role,
        "status": current_user.status,
        "avatar_url": current_user.avatarUrl,
        "subscription_status": access_state.subscription_status,
        "subscription_blocked": access_state.blocked,
        "subscription_block_message": access_state.block_message,
        "subscription_unpaid_invoice_id": access_state.unpaid_invoice_id,
        "subscription_unpaid_invoice_status": access_state.unpaid_invoice_status,
        "subscription_has_active_payment_method": access_state.has_active_payment_method,
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
    
    otp = str(random_with_N_digits(6))
    
    r_cache.setex(
        f"forgot_password_otp:{email}",
        300,
        otp
    )
    
    send_otp_email(user.recoveryEmail, otp, purpose="forgot_password")
    
    return {
        "message": f"OTP sent to recovery email",
        "recovery_email_hint": f"{user.recoveryEmail[:4]}****@{user.recoveryEmail.split('@')[1]}"
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
    if not await verify_turnstile(data.cf_token):
         raise HTTPException(status_code=400, detail="Invalid security token")

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
    
    send_otp_email(data.recovery_email, otp, purpose="register")
    
    return {
        "message": "OTP sent to recovery email"
    }


@auth_router.post("/google/register/verify", tags=["auth"])
async def google_register_verify(data: Google_Register_Verify):
    email = data.email
    otp = data.otp
    
    reg_key = f"google_reg_pending:{email}"
    stored_data = r_cache.hgetall(reg_key)
    
    if not stored_data or stored_data.get("otp") != otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    r_cache.hset(reg_key, "verified", "true")
    
    return {"message": "OTP verified", "verified": True}


@auth_router.post("/google/register/complete", response_model=Token, tags=["auth"])
async def google_register_complete(response: Response, request: Request, data: Google_Register_Complete):
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
        
    await db.notificationconfig.create(data={"userId": str(user.id)})
        
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

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        
        await db.activitylog.create(
            data={
                "userId": str(user.id),
                "topic": "Login",
                "detail": "User registered and logged in via Google",
                "ipAddress": ip_address, 
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    response.set_cookie(
        key=COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False
    )
    
    return Token(access_token=access_token, token_type="bearer")


@auth_router.post("/check-user", tags=["auth"])
async def check_user(data: CheckUser_Request):
    if not await verify_turnstile(data.cf_token):
         raise HTTPException(status_code=400, detail="Invalid security token")

    user = await db.user.find_unique(where={"email": data.email})
    
    if not user:
        return {"exists": False}

    response_data = {
        "exists": True,
        "has_password": bool(user.password),
        "is_google": bool(user.googleAuthId),
        "recovery_email_hint": f"{user.recoveryEmail[:4]}****@{user.recoveryEmail.split('@')[1]}" if user.recoveryEmail else None,
        "otp_sent": False
    }

    if user.googleAuthId and not user.password:
         if user.recoveryEmail:
             otp = str(random_with_N_digits(6))
             r_cache.setex(f"login_pending:{user.email}", 300, otp)
             send_otp_email(user.recoveryEmail, otp, purpose="login")
             response_data["otp_sent"] = True

    return response_data


@auth_router.post("/login/otp-init", tags=["auth"])
async def login_otp_init(data: Login_OTP_Init):
    if not await verify_turnstile(data.cf_token):
         raise HTTPException(status_code=400, detail="Invalid security token")

    user = await db.user.find_unique(where={"email": data.email})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.googleAuthId and not user.password:
         raise HTTPException(status_code=400, detail="Account setup incomplete. Please contact support.")

    if not user.recoveryEmail:
        raise HTTPException(status_code=400, detail="No recovery email set.")

    otp = str(random_with_N_digits(6))
    r_cache.setex(f"login_pending:{user.email}", 300, otp)
    
    send_otp_email(user.recoveryEmail, otp, purpose="login")
    
    return {
        "message": "OTP sent to recovery email"
    }


@auth_router.post("/set-password", tags=["auth"])
async def set_password(
    data: SetPassword_Request,
    current_user: Annotated[any, Depends(get_current_active_user)]
):
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
        
    hashed_password = bcrypt.hashpw(data.new_password.encode("utf-8"), bcrypt.gensalt())
    
    await db.user.update(
        where={"id": str(current_user.id)},
        data={"password": hashed_password.decode("utf-8")}
    )
    
    return {"message": "Password update successfully"}
