from fastapi import APIRouter, HTTPException
from ..database.client import r_cache, db
from ..models.authentication_model import Register_OTP, Register_Verify, Register_Complete, Login
from lib.untils import random_with_N_digits
import bcrypt

auth_router = APIRouter()


@auth_router.post("/register/otp", tags=["users"])
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
    r_cache.expire(f"reg_detail_{recovery_email_key}", 60 * 10)  # หมดอายุใน 10 นาที
    
    otp = str(random_with_N_digits(6))
    r_cache.set(f"reg_otp_{recovery_email_key}", otp, ex=60 * 5)  # หมดอายุใน 5 นาที
    
    print(f"[DEV] OTP for {register_otp.recovery_email}: {otp}")
    
    return {
        "status_code": 200,
        "message": "OTP sent to your recovery email",
        "dev_otp": otp  # ลบออกใน production
    }


@auth_router.post("/register/verify_otp", tags=["users"])
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
    r_cache.set(f"reg_user_{recovery_email_key}", str(user.id), ex=60 * 30)  # หมดอายุใน 30 นาที
    
    return {
        "status_code": 201,
        "message": "OTP verified, account created. Please set your username.",
        "user": {
            "id": str(user.id),
            "email": user.email
        }
    }


@auth_router.post("/register/complete", tags=["users"])
async def register_complete(register_complete: Register_Complete):
    recovery_email_key = register_complete.recovery_email.split('@')[0]
    
    user_id = r_cache.get(f"reg_user_{recovery_email_key}")
    if not user_id:
        raise HTTPException(status_code=400, detail="Session expired. Please login and update username.")
    
    existing_username = await db.user.find_first(where={"username": register_complete.username})
    if existing_username and str(existing_username.id) != user_id:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # อัพเดท username
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


@auth_router.post("/login", tags=["users"])
async def login(login_data: Login):
    from jose import jwt
    from datetime import datetime, timedelta
    import os
    
    user = await db.user.find_unique(where={"email": login_data.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.password:
        raise HTTPException(status_code=401, detail="Please login with Google")
    
    if not bcrypt.checkpw(login_data.password.encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if user.status == "banned":
        raise HTTPException(status_code=403, detail="Your account has been banned")
    
    SECRET_KEY = os.getenv("JWT_SECRET", "UknownmeInLove")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS = 1
    
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "exp": expire
    }
    access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    
    return {
        "status_code": 200,
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "status": user.status
        }
    }
