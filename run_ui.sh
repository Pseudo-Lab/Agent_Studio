#!/bin/bash
# Run Kiosk Agent Backend API

cd "$(dirname "$0")/backend"

echo "Starting Kiosk Agent API..."
echo "API available at http://localhost:8080"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
