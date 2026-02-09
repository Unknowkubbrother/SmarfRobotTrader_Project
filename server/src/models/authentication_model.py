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

class ForgotPassword_Request(BaseModel):
    email: str

class ForgotPassword_Verify(BaseModel):
    email: str
    otp: str

class ForgotPassword_Reset(BaseModel):
    email: str
    otp: str
    new_password: str

class Login_Verify(BaseModel):
    email: str
    otp: str

class Google_Register_OTP(BaseModel):
    id_token: str
    recovery_email: str

class Google_Register_Verify(BaseModel):
    email: str
    otp: str

class Google_Register_Complete(BaseModel):
    email: str
    username: str