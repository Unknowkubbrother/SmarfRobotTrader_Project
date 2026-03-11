from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional


class VisionLLMChartRate(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0


class VisionLLMRequest(BaseModel):
    date_time: datetime
    symbol: str
    timeframe: str = "H1"
    bot_config_id: Optional[str] = None
    chart_rates: Optional[list[VisionLLMChartRate]] = None
    resolved_bar_time: Optional[str] = None
    source_server: Optional[str] = None
    source_login: Optional[str] = None

    @field_validator("date_time", mode="before")
    @classmethod
    def parse_date_time(cls, v):
        if isinstance(v, str):
            for fmt in ("%Y.%m.%d %H.%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            raise ValueError(
                f"Invalid date_time format: '{v}'. "
                "Expected: '2025.12.31 15.00' or '2025-12-31 15:00:00'"
            )
        return v


class VisionLLMTextEmbeddingRequest(BaseModel):
    text: str
