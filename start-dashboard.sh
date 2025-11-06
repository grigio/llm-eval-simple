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

# More aggressive port cleanup
echo "🔧 Ensuring ports are free..."
lsof -ti:4000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Wait for sockets to be released
echo "⏳ Waiting for sockets to be released..."
sleep 3

# Double-check ports are really free
for port in 4000 3000; do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "⚠️  Port $port still in use, forcing cleanup..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
done

# Start API server with retry mechanism
echo "🔧 Starting API server on port 4000..."
for attempt in {1..3}; do
    uv run python api_server.py &
    API_PID=$!
    
    # Verify the process actually started
    sleep 2
    if kill -0 $API_PID 2>/dev/null; then
        echo "✅ API server process started (attempt $attempt)"
        break
    else
        echo "⚠️  API server failed to start (attempt $attempt)"
        if [ $attempt -eq 3 ]; then
            echo "❌ API server failed to start after 3 attempts"
            exit 1
        fi
        echo "🔄 Waiting before retry..."
        sleep 2
    fi
done

# Wait for API server to start and verify it's responding
echo "⏳ Waiting for API server to be ready..."
for i in {1..15}; do
    if curl -s http://localhost:4000/api/results > /dev/null 2>&1; then
        echo "✅ API server is ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ API server failed to start after 15 seconds"
        echo "🔍 Checking what's using port 4000:"
        lsof -i :4000 || echo "Port 4000 appears to be free"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

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