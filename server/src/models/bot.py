from pydantic import BaseModel
from typing import List
from enum import Enum

class Create_Bot_Version(BaseModel):
    label: str
    dockerImageId: str
    versionTag: str
    symbol: str
    timeframe: str
    releaseNotes: List[str]



class RiskLevelEnum(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Create_Bot_Configuration(BaseModel):
    accountId: str
    modelId: str
    riskLevel: RiskLevelEnum
    
    
