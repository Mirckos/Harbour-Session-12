#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

docker compose up -d --build

until curl -fs http://127.0.0.1:8012/ready >/dev/null 2>&1; do
  sleep 1
done

curl -s http://127.0.0.1:8012/health | python3 -m json.tool

curl -s -X POST http://127.0.0.1:8012/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello, I can help you automate this task",
    "dialog_id": "11111111-1111-1111-1111-111111111111",
    "id": "22222222-2222-2222-2222-222222222222",
    "participant_index": 0
  }' | python3 -m json.tool

echo
echo "Open Swagger UI: http://127.0.0.1:8012/docs"
