from prisma import Prisma
import redis
from dotenv import load_dotenv
import os

load_dotenv()

db = Prisma(datasource={'url': os.getenv('DATABASE_URL')})
r_cache = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)
