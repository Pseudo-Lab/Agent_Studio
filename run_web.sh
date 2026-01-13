#!/bin/bash
# Run Kiosk Agent Web Frontend

cd "$(dirname "$0")/web"

echo "Starting Kiosk Agent UI..."
echo "Open http://localhost:3000"
echo ""

npm run dev
