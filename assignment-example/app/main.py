from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("session12.assignment")


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


class ProbabilityModel(Protocol):
    def predict_proba(self, text: str) -> float:
        pass


class KeywordBotClassifier:
    keywords = ("automate", "help", "assistant", "bot", "support", "instant", "generate")

    def predict_proba(self, text: str) -> float:
        normalized = text.lower()
        hits = sum(keyword in normalized for keyword in self.keywords)
        probability = 0.08 + hits * 0.15 + min(len(normalized) / 500.0, 0.2)
        return max(0.01, min(probability, 0.99))


class MlflowPyFuncClassifier:
    def __init__(self, model: Any):
        self.model = model

    def predict_proba(self, text: str) -> float:
        prediction = self.model.predict([text])
        value = prediction[0]
        if isinstance(value, dict):
            value = value.get("is_bot_probability", value.get("probability"))
        return float(value)


@dataclass(frozen=True)
class LoadedModel:
    model: ProbabilityModel
    model_uri: str
    model_name: str
    model_version: str


def load_demo_model(model_uri: str) -> LoadedModel:
    return LoadedModel(
        model=KeywordBotClassifier(),
        model_uri=model_uri,
        model_name="keyword-bot-demo",
        model_version="local-demo",
    )


def resolve_mlflow_alias(model_uri: str) -> tuple[str, str]:
    if not model_uri.startswith("models:/") or "@" not in model_uri:
        return model_uri, "unknown"

    model_name, alias = model_uri.removeprefix("models:/").split("@", 1)

    from mlflow import MlflowClient

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, alias)
    return model_name, str(model_version.version)


def load_from_config() -> LoadedModel:
    model_uri = os.getenv("MODEL_URI", "demo://keyword-bot")

    if model_uri == "demo://keyword-bot":
        return load_demo_model(model_uri)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required when MODEL_URI points to MLflow.")

    import mlflow
    import mlflow.pyfunc

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    model_name, model_version = resolve_mlflow_alias(model_uri)

    return LoadedModel(
        model=MlflowPyFuncClassifier(model),
        model_uri=model_uri,
        model_name=model_name,
        model_version=model_version,
    )


REQUESTS = Counter("assignment_predict_requests_total", "Total /predict requests")
LATENCY = Histogram("assignment_predict_latency_seconds", "Latency of /predict requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_from_config()
    app.state.model = loaded.model
    app.state.model_uri = loaded.model_uri
    app.state.model_name = loaded.model_name
    app.state.model_version = loaded.model_version
    logger.info(
        "model_loaded model_uri=%s model_name=%s model_version=%s",
        loaded.model_uri,
        loaded.model_name,
        loaded.model_version,
    )
    yield
    app.state.model = None


app = FastAPI(
    title="Session 12 Assignment Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
def ready() -> dict[str, Any]:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    return {
        "status": "ready",
        "model_uri": app.state.model_uri,
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
