from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class ScoreResponse(BaseModel):
    is_bot_probability: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str


class KeywordBotClassifier:
    keywords = ("automate", "help", "assistant", "bot", "support", "instant", "generate")

    def predict_proba(self, text: str) -> float:
        normalized = text.lower()
        hits = sum(keyword in normalized for keyword in self.keywords)
        probability = 0.08 + hits * 0.15 + min(len(normalized) / 500.0, 0.2)
        return max(0.01, min(probability, 0.99))


SCORES = Counter("classifier_score_requests_total", "Total score requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = KeywordBotClassifier()
    app.state.model_name = os.getenv("MODEL_NAME", "keyword-bot-classifier")
    app.state.model_version = os.getenv("MODEL_VERSION", "service-v1")
    yield
    app.state.model = None


app = FastAPI(
    title="Session 12 Classifier Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
def ready() -> dict[str, Any]:
    return {
        "status": "ready",
        "model_name": app.state.model_name,
        "model_version": app.state.model_version,
    }


@app.post("/score", response_model=ScoreResponse, tags=["inference"])
def score(request: ScoreRequest) -> ScoreResponse:
    SCORES.inc()
    return ScoreResponse(
        is_bot_probability=app.state.model.predict_proba(request.text),
        model_name=app.state.model_name,
        model_version=app.state.model_version,
    )


@app.get("/metrics", tags=["system"])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
