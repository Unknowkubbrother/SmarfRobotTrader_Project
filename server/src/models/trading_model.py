from pydantic import BaseModel, Field


class Create_Trading_Account(BaseModel):
    brokerName: str
    serverName: str
    mt5LoginId: str
    mt5Password: str


class UpsertTradingJournalRequest(BaseModel):
    ticketId: int
    tradeRationale: str | None = None
    mistakeLesson: str | None = None
    tags: list[str] = Field(default_factory=list)
    attachmentUrls: list[str] = Field(default_factory=list)
