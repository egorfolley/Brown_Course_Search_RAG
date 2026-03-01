#!/bin/bash
# Quick Start Script for Docker

set -e

echo "🚀 Brown Course Search - Docker Quick Start"
echo ""

# Check if data files exist
if [ -f "data/faiss.index" ] && [ -f "data/courses.json" ]; then
    echo "✅ Data files found! Docker will start fast (~30 seconds)"
    echo ""
    docker-compose up --build
else
    echo "⚠️  No data files found. Two options:"
    echo ""
    echo "OPTION 1 (RECOMMENDED - Fast Docker startup):"
    echo "  1. Generate data locally first:"
    echo "     python app/app.py"
    echo "     (Wait 5-10 minutes, then Ctrl+C to stop)"
    echo ""
    echo "  2. Then run Docker (will start in ~30 seconds):"
    echo "     docker-compose up --build"
    echo ""
    echo "OPTION 2 (Run everything in Docker):"
    echo "  docker-compose up --build"
    echo "  (Will take 8-15 minutes on first run)"
    echo ""
    read -p "Choose option (1 or 2): " choice
    
    if [ "$choice" = "1" ]; then
        echo ""
        echo "📦 Generating data locally..."
        echo "This will take 5-10 minutes. Press Ctrl+C when you see 'Uvicorn running'"
        echo ""
        python app/app.py
    else
        echo ""
        echo "🐳 Starting Docker (this will take 8-15 minutes)..."
        docker-compose up --build
    fi
fi
