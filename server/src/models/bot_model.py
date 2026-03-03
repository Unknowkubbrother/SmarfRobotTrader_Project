from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum

class Create_Bot_Version(BaseModel):
    label: str
    dockerImageId: str | None = None
    versionTag: str
    symbol: str | None = None
    timeframe: str | None = None
    releaseNotes: List[str]


class RiskLevelEnum(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Create_Bot_Configuration(BaseModel):
    accountId: str
    modelId: str
    riskLevel: RiskLevelEnum

class Update_Bot_Status(BaseModel):
    botConfigId: str
    status: str  # "running" or "stopped"

class Update_Bot_Risk(BaseModel):
    botConfigId: str
    riskLevel: RiskLevelEnum

class Update_Bot_Schedule(BaseModel):
    botConfigId: str
    tradingSchedule: Dict[str, bool]

class Change_Bot_Model(BaseModel):
    botConfigId: str
    newModelId: str

class Delete_Bot(BaseModel):
    botConfigId: str


class Apply_Bot_Update(BaseModel):
    botConfigId: str


class Emergency_Bot_Stop(BaseModel):
    botConfigId: str
