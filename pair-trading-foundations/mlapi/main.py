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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
 
model_path = "./distilbert-base-uncased-finetuned-sst2"
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    top_k=None,
)

logger = logging.getLogger(__name__)
LOCAL_REDIS_URL = "redis://redis-service:6379"
# LOCAL_REDIS_URL = "redis://localhost:6379"

@asynccontextmanager
async def lifespan(app: FastAPI):
    HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
    logger.debug(HOST_URL)
    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield

app = FastAPI(lifespan=lifespan)

class SentimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: list[str]
    pass

class Sentiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label : str
    score : float
    pass

class SentimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions : list[list[Sentiment]]
    pass


@app.post("/project-predict", response_model=SentimentResponse)
@cache(expire=60)
async def predict(sentiments: SentimentRequest):
    sentiments_output = []
    for text in sentiments.text:
        sentiments_output.append(classifier(text)[0])

    output = SentimentResponse(predictions=sentiments_output)
    # Return the pydantic output
    return output

@app.get("/health")
async def health():
    return {"status": "healthy"}
