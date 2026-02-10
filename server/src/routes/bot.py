from fastapi import APIRouter, HTTPException, Depends, status, Response, Request, Form
from pydantic import BaseModel

bot_router = APIRouter()

@bot_router.get("/", tags=["bot"])
async def bot_all(request: Request):
    return {"message": request.state.user_id, "email": request.state.email, "role": request.state.role}