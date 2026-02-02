from fastapi import FastAPI
from .routes import authentication

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(authentication.auth_router)