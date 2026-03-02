from pydantic import BaseModel, Field


class Create_Trading_Account(BaseModel):
    brokerName: str
    serverName: str
    mt5LoginId: str
    mt5Password: str


class Update_Trading_Account(BaseModel):
    accountId: str
    brokerName: str | None = None
    serverName: str | None = None
    mt5LoginId: str | None = None
    mt5Password: str | None = None


class Delete_Trading_Account(BaseModel):
    accountId: str


class UpsertTradingJournalRequest(BaseModel):
    ticketId: int
    tradeRationale: str | None = None
    mistakeLesson: str | None = None
    tags: list[str] = Field(default_factory=list)
    attachmentUrls: list[str] = Field(default_factory=list)
