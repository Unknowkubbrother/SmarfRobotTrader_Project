from pydantic import BaseModel

class Register_OTP(BaseModel):
    email: str
    recovery_email: str
    password: str

class Register_Verify(BaseModel):
    recovery_email: str
    otp: str

class Register_Complete(BaseModel):
    recovery_email: str
    username: str

class Login(BaseModel):
    email: str
    password: str