#!/bin/bash
# Start both backend and frontend locally

set -e

echo "🚀 Starting Brown Course Search (local mode)..."
echo ""

# Check if data exists
if [ ! -f "data/faiss.index" ]; then
    echo "⚠️  First time setup: Generating data (5-10 minutes)..."
    echo ""
fi

# Start backend in background
echo "📡 Starting backend API on http://localhost:8000..."
python app/app.py > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✅ Backend ready!"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "❌ Backend failed to start. Check logs/backend.log"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
done

# Start frontend in background
echo "🎨 Starting frontend UI on http://localhost:8501..."
streamlit run frontend/ui.py > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 3

echo ""
echo "✅ Both services running!"
echo ""
echo "  Backend API:  http://localhost:8000/docs"
echo "  Frontend UI:  http://localhost:8501"
echo ""
echo "📋 Logs:"
echo "  Backend:  tail -f logs/backend.log"
echo "  Frontend: tail -f logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop both services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    # Kill any remaining streamlit/uvicorn processes
    pkill -f "streamlit run" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    echo "✅ Services stopped"
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Follow logs from both services
tail -f logs/backend.log logs/frontend.log 2>/dev/null || sleep infinity
