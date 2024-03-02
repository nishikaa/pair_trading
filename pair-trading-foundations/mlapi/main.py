import logging
import os
from contextlib import asynccontextmanager

# Import Modules for pydantic and validation
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel
from redis import asyncio

logger = logging.getLogger(__name__)
# SERVICE_REDIS_URL = "redis://redis-service:6379"
LOCAL_REDIS_URL = "redis://localhost:6379"

# app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI): 
    HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
    logger.debug(HOST_URL)
    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield

app = FastAPI(lifespan=lifespan)

class FinanceModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration_in_days: int = Field(ge=0)
    dollar_amt: int = Field(ge=0)
    
    def to_numpy(self):
        return np.array(
            [
                self.duration_in_days,
                self.dollar_amt,
            ]
        )

    @model_validator(mode="after")
    def check_age(self) -> "Input":
        # Check for valid
        days = self.duration_in_days
        dollars = self.dollar_amt
        if days < 0:
            raise ValueError("Invalid days")
        if dollars < 0:
            raise ValueError("Invalid dollars")
        return self


class FinanceModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions : list[str]


@app.post("/mlapi-predict", response_model=FinanceModelResponse)
# @cache(expire=60)
async def mlapi(finance_model: FinanceModelRequest):
    finance_model_output = []
    finance_model_output.append("None")

    output = FinanceModelResponse(predictions=finance_model_output)
    # Return the pydantic output
    return output

@app.get("/health")
async def health():
    return {"status": "healthy"}
