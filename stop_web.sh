#!/bin/bash
# Stop backend (port 8080) and Next.js frontend (port 3000)

set -e

echo "🛑 Stopping Agent Kiosk Agent services..."
echo ""

# Function to kill processes on a specific port
kill_port() {
  local port=$1
  local service=$2

  if lsof -ti:${port} > /dev/null 2>&1; then
    echo "🔍 Found ${service} running on port ${port}"
    lsof -ti:${port} | xargs kill -9 2>/dev/null || true
    sleep 1

    # Verify process was killed
    if lsof -ti:${port} > /dev/null 2>&1; then
      echo "⚠️  Warning: ${service} still running on port ${port}"
    else
      echo "✅ ${service} stopped successfully"
    fi
  else
    echo "ℹ️  No ${service} process found on port ${port}"
  fi
}

# Stop backend on port 8080
kill_port 8080 "Backend (FastAPI)"

echo ""

# Stop frontend on port 3000
kill_port 3000 "Frontend (Next.js)"

echo ""
echo "✅ All services stopped!"
