from contextlib import asynccontextmanager
from fastapi import FastAPI
from .routes import authentication
from .database.client import db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


app.include_router(authentication.auth_router, prefix="/auth")

# uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload