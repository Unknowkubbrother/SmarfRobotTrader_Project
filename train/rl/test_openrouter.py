from langchain.chat_models import init_chat_model
from os import getenv
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="mistralai/ministral-14b-2512",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

response = model.invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")
print(response.content)
