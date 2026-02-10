from pydantic import BaseModel

class Create_Trading_Account(BaseModel):
    brokerName: str
    serverName: str
    mt5LoginId: str
    mt5Password: str