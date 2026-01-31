#!/usr/bin/env bash
set -euo pipefail

echo "Starting Cyrene AI Backend..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
