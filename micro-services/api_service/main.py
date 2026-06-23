from __future__ import annotations

import logging
import os
import time
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("session12.api")

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "2.0"))


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


REQUESTS = Counter("api_predict_requests_total", "Total public /predict requests")
UPSTREAM_ERRORS = Counter("api_upstream_errors_total", "Total classifier upstream errors")
LATENCY = Histogram("api_predict_latency_seconds", "Latency of public /predict requests")

app = FastAPI(title="Session 12 Public Inference API", version="1.0.0")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
async def ready() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{CLASSIFIER_URL}/ready")
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="classifier readiness timeout") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="classifier is not ready") from exc

    return {"status": "ready", "classifier": response.json()}


@app.post("/predict", response_model=Prediction, tags=["inference"])
async def predict(message: IncomingMessage) -> Prediction:
    started = time.perf_counter()
    request_id = uuid4()
    REQUESTS.inc()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{CLASSIFIER_URL}/score",
                json={"text": message.text},
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        UPSTREAM_ERRORS.inc()
        raise HTTPException(status_code=504, detail="classifier timeout") from exc
    except httpx.HTTPError as exc:
        UPSTREAM_ERRORS.inc()
        raise HTTPException(status_code=502, detail="classifier request failed") from exc

    payload = response.json()
    latency_ms = (time.perf_counter() - started) * 1000
    LATENCY.observe(latency_ms / 1000)

    logger.info(
        "request_id=%s model_name=%s model_version=%s latency_ms=%.2f status_code=200",
        request_id,
        payload.get("model_name"),
        payload.get("model_version"),
        latency_ms,
    )

    return Prediction(
        id=request_id,
        message_id=message.id,
        dialog_id=message.dialog_id,
        participant_index=message.participant_index,
        is_bot_probability=payload["is_bot_probability"],
    )


@app.get("/metrics", tags=["system"])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
