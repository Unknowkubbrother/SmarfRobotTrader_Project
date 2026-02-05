from contextlib import asynccontextmanager
from fastapi import FastAPI
from .routes import authentication
from .database.client import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database connection lifecycle"""
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


app.include_router(authentication.auth_router)