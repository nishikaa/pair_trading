import logging
import os
from contextlib import asynccontextmanager

# Import Modules for pydantic and validation
from pydantic import BaseModel, ConfigDict, Field, model_validator

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel
from redis import asyncio
import json

logger = logging.getLogger(__name__)
# SERVICE_REDIS_URL = "redis://redis-service:6379"
LOCAL_REDIS_URL = "redis://localhost:6379"

@asynccontextmanager
async def lifespan(app: FastAPI): 
    HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
    logger.debug(HOST_URL)
    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield

# Load the model on startup, this requires container to be instantiated
xgb_model = joblib.load("xgb_st_entry.pkl")

# Load the csv on startup
transformed_data = pd.read_csv('transformed_data.csv')

app = FastAPI(lifespan=lifespan)

class FinanceModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration_in_days: int = Field(ge=0)
    dollar_amt: int = Field(ge=0)
    requested_pairs: int = Field(ge=0)
    
    def to_numpy(self):
        return np.array(
            [
                self.duration_in_days,
                self.dollar_amt,
                self.requested_pairs
            ]
        )

    @model_validator(mode="after")
    def check_age(self) -> "Input":
        # Check for valid
        days = self.duration_in_days
        dollars = self.dollar_amt
        requested_pairs = self.requested_pairs
        if days < 0:
            raise ValueError("Invalid days")
        if dollars < 0:
            raise ValueError("Invalid dollars")
        if requested_pairs < 1:
            raise ValueError("Invalid amount")
        return self


class FinanceModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions : list[list[str]]


@app.post("/mlapi-predict", response_model=FinanceModelResponse)
# @cache(expire=60)
async def mlapi(financemodel_request: FinanceModelRequest):
    finance_model_output = []

    latest = transformed_data[transformed_data.Date == transformed_data.Date.max()]
    X = xgb_model.feature_scaler.transform(latest[xgb_model.features_names])
    # Run inference via matrix
    probability_class_1 = xgb_model.predict_proba(X)
    predictions = np.argmax(probability_class_1, axis=1) # Not used
    probability = probability_class_1[:, 1]

    latest = transformed_data[transformed_data.Date == transformed_data.Date.max()]
    latest['probability'] = [x for x in probability]
    latest = latest.reset_index()
    # Get top K
    K = financemodel_request.requested_pairs
    output = latest.sort_values('probability', ascending=False).head(K)
    probabilities = np.array(output['probability'])
    ticker_left = np.array(output['Ticker_P1'])
    ticker_right = np.array(output['Ticker_P2'])
    for i in range(K):
        pairs_data = []
        pairs_data.append(str(ticker_left[i]))
        pairs_data.append(str(ticker_right[i]))
        pairs_data.append(str(probabilities[i]))
        finance_model_output.append(pairs_data)

    # Generate output
    output = FinanceModelResponse(predictions=finance_model_output)
    # Return the pydantic output
    return output

@app.get("/health")
async def health():
    return {"status": "healthy"}
