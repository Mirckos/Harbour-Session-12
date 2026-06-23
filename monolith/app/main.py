from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("session12.monolith")


class IncomingMessage(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    dialog_id: UUID
    id: UUID
    participant_index: int = Field(ge=0)


class Prediction(BaseModel):
    id: UUID
    message_id: UUID
    dialog_id: UUID
    participant_index: int
    is_bot_probability: float = Field(ge=0.0, le=1.0)


class KeywordBotClassifier:
    """Tiny deterministic model used only to keep the HTTP demo self-contained."""

    keywords = (
        "automate",
        "help",
        "assistant",
        "bot",
        "support",
        "instant",
        "generate",
    )

    def predict_proba(self, text: str) -> float:
        normalized = text.lower()
        hits = sum(keyword in normalized for keyword in self.keywords)
        length_bonus = min(len(normalized) / 500.0, 0.2)
        probability = 0.08 + hits * 0.15 + length_bonus
        return max(0.01, min(probability, 0.99))


REQUESTS = Counter("predict_requests_total", "Total /predict requests")
LATENCY = Histogram("predict_latency_seconds", "Latency of /predict requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = KeywordBotClassifier()
    app.state.model_name = os.getenv("MODEL_NAME", "keyword-bot-monolith")
    app.state.model_version = os.getenv("MODEL_VERSION", "baked-v1")
    logger.info(
        "model_loaded model_name=%s model_version=%s",
        app.state.model_name,
        app.state.model_version,
    )
    yield
    app.state.model = None


app = FastAPI(
    title="Session 12 Monolith Inference API",
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


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(message: IncomingMessage) -> Prediction:
    started = time.perf_counter()
    request_id = uuid4()
    REQUESTS.inc()

    probability = app.state.model.predict_proba(message.text)
    latency_ms = (time.perf_counter() - started) * 1000
    LATENCY.observe(latency_ms / 1000)

    logger.info(
        "request_id=%s model_name=%s model_version=%s latency_ms=%.2f status_code=200",
        request_id,
        app.state.model_name,
        app.state.model_version,
        latency_ms,
    )

    return Prediction(
        id=request_id,
        message_id=message.id,
        dialog_id=message.dialog_id,
        participant_index=message.participant_index,
        is_bot_probability=probability,
    )


@app.get("/metrics", tags=["system"])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
