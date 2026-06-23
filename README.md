# Harbour Session 12: HTTP Web Services & Model Serving

This repository contains the Session 12 deck and small runnable demos for model serving over HTTP.

The demos follow the lecture story:

- package a model behind a stable API contract;
- expose `/health`, `/ready`, `/predict`, and `/metrics`;
- compare a monolith with a small model-as-a-service setup;
- show how production code can load an MLflow `champion` alias.

## Contents

- `Session_12_HTTP.pptx` and `Session_12_HTTP.pdf` - lecture slides.
- `monolith/` - one FastAPI service with the model loaded inside the app.
- `micro-services/` - an API service calling a separate classifier service.
- `assignment-example/` - homework-shaped service that can use either a local demo model or an MLflow Registry model URI.

All examples use a tiny deterministic text classifier. It is intentionally simple: the goal is to explain API shape, service boundaries, readiness, and deployment wiring, not model quality.

## Prerequisites

- Docker Desktop or Docker Engine with Docker Compose v2.
- `curl` for smoke tests.

Check:

```bash
docker compose version
curl --version
```

## Demo 1: Monolith

The model is loaded once at startup inside the same FastAPI process that serves HTTP requests.

```bash
cd monolith
./run.sh
```

Open:

```text
Swagger UI: http://127.0.0.1:8012/docs
Health:     http://127.0.0.1:8012/health
Ready:      http://127.0.0.1:8012/ready
Metrics:    http://127.0.0.1:8012/metrics
```

Talking point:

> This is the simplest deployment shape. The app and model are shipped together, which is easy to reason about, but every model update usually means rebuilding and redeploying the whole service.

Stop:

```bash
docker compose down
```

## Demo 2: Micro-services

The public API service keeps the course `/predict` contract. It calls a separate classifier service over HTTP.

```bash
cd ../micro-services
./run.sh
```

Open:

```text
API Swagger UI:        http://127.0.0.1:8013/docs
Classifier Swagger UI: http://127.0.0.1:8014/docs
```

Talking point:

> The API service owns the external contract. The classifier service owns model loading and scoring. This makes the model independently replaceable, but adds network calls, timeouts, readiness checks, and more operational moving parts.

Stop:

```bash
docker compose down
```

## Demo 3: Assignment Example

This example mirrors the homework from the final slide. By default it uses a local demo model so the service starts without MLflow.

```bash
cd ../assignment-example
./run.sh
```

Open:

```text
Swagger UI: http://127.0.0.1:8015/docs
Ready:      http://127.0.0.1:8015/ready
```

To use a real MLflow Registry model instead, set:

```bash
MODEL_URI=models:/bot_classifier@champion
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
docker compose up --build
```

Talking point:

> The service may load `models:/bot_classifier@champion`, but it must log the resolved concrete model version. Aliases are convenient for rollout; concrete versions are required for reproducibility.

## Request Example

All demos accept the same public prediction payload:

```bash
curl -s -X POST http://127.0.0.1:8012/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello, I can help you automate this task",
    "dialog_id": "11111111-1111-1111-1111-111111111111",
    "id": "22222222-2222-2222-2222-222222222222",
    "participant_index": 0
  }' | python3 -m json.tool
```

Expected shape:

```json
{
  "id": "generated-response-uuid",
  "message_id": "22222222-2222-2222-2222-222222222222",
  "dialog_id": "11111111-1111-1111-1111-111111111111",
  "participant_index": 0,
  "is_bot_probability": 0.46
}
```

## What To Emphasize

- `/health` answers: is the process alive?
- `/ready` answers: is the model loaded and can the service receive traffic?
- `/predict` is the stable business contract.
- `/metrics` is machine-readable telemetry for Prometheus-style scraping.
- Do not load the model inside `/predict`; load once during startup.
- Do not log raw user text by default.
- For MLflow aliases, log the resolved concrete model version.
