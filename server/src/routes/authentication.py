from fastapi import APIRouter
from ..database.client import db

auth_router = APIRouter()


@auth_router.get("/users/", tags=["users"])
async def read_users():
    """Get all users from database"""
    users = await db.user.find_many()
    return users


@auth_router.get("/users/{user_id}", tags=["users"])
async def read_user(user_id: str):
    """Get a single user by ID"""
    user = await db.user.find_unique(where={"id": user_id})
    if not user:
        return {"error": "User not found"}
    return user