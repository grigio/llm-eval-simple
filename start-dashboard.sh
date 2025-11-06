#!/bin/bash

# Start script for LLM Eval Dashboard

echo "🚀 Starting LLM Eval Dashboard..."

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: api_server.py not found. Please run from project root."
    exit 1
fi

# Kill any existing processes on ports 4000 and 3000
echo "🔄 Cleaning up existing processes..."
pkill -f "python api_server.py" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# Start API server
echo "🔧 Starting API server on port 4000..."
uv run python api_server.py &
API_PID=$!

# Wait for API server to start
sleep 2

# Start frontend
echo "🎨 Starting frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Dashboard started successfully!"
echo ""
echo "📊 Frontend: http://localhost:3000"
echo "🔌 API: http://localhost:4000/api/results"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $API_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for processes
wait