from fastapi import APIRouter
auth_router = APIRouter()

@auth_router.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]