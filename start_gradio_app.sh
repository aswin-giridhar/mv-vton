#!/bin/bash

# MV-VTON Gradio App Startup Script - Environment Isolation Version
# This script runs Gradio in the BASE environment to avoid package conflicts
# while the API server runs in the mv-vton conda environment

set -e

echo "🌐 MV-VTON Gradio Web Interface Startup (Environment Isolation)"
echo "=============================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda or miniconda."
    exit 1
fi

# Initialize conda for shell script
eval "$(conda shell.bash hook)"

# Use mv-vton environment for both API server and Gradio (unified environment)
echo "🔧 Activating mv-vton conda environment for Gradio..."
conda activate mv-vton

# Check if API server is running
echo "🔍 Checking API server status..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ API server is running on localhost:5000"
else
    echo "⚠️  API server not detected on localhost:5000"
    echo ""
    echo "📝 IMPORTANT: You need to start the API server first!"
    echo "   In another terminal, run: ./start_api_server.sh"
    echo ""
    echo "🤔 Do you want to:"
    echo "   1. Continue starting Gradio (you can start API server later)"
    echo "   2. Exit and start API server first"
    echo ""
    read -p "Enter choice (1 or 2): " choice
    
    case $choice in
        1)
            echo "🚀 Starting Gradio app (remember to start API server)..."
            ;;
        2)
            echo "👋 Please start the API server first: ./start_api_server.sh"
            exit 0
            ;;
        *)
            echo "Invalid choice. Starting Gradio app..."
            ;;
    esac
fi

# Verify required packages in mv-vton environment
echo "🔍 Verifying packages in mv-vton environment..."

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Using Python: $python_version (mv-vton environment)"

# Quick verification of key packages
if python -c "import gradio, requests, PIL, numpy" 2>/dev/null; then
    echo "✅ All required packages are available in mv-vton environment"
else
    echo "❌ Some packages missing. Installing requirements..."
    pip install --upgrade gradio requests pillow numpy
fi

echo ""
echo "🌟 Starting Gradio Web Interface in mv-vton Environment..."
echo "🔗 Local access: http://localhost:7860"
echo "🔗 Network access: http://$(hostname -I | awk '{print $1}'):7860"
echo ""
echo "📋 Unified Environment Setup:"
echo "   ✅ API Server: Running in mv-vton conda environment (localhost:5000)"
echo "   ✅ Gradio UI: Running in mv-vton conda environment (localhost:7860)"
echo "   ✅ Package Conflicts: Resolved with FastAPI and modern dependencies"
echo ""
echo "📋 Usage Notes:"
echo "   • API server should be running on localhost:5000 (same mv-vton environment)"
echo "   • Upload person and clothing images to get started"
echo "   • Use advanced parameters to fine-tune results"
echo "   • Interactive API docs available at localhost:5000/docs"
echo ""
echo "Press Ctrl+C to stop the application"
echo "=============================================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Gradio application..."
    if [ ! -z "$GRADIO_PID" ]; then
        kill $GRADIO_PID 2>/dev/null || true
        wait $GRADIO_PID 2>/dev/null || true
    fi
    # Also kill any remaining gradio processes
    pkill -f "python gradio_app.py" 2>/dev/null || true
    echo "👋 Gradio application stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start the Gradio app in background and get its PID
echo "🚀 Launching full-featured Gradio application with gradio_app.py..."
python gradio_app.py &
GRADIO_PID=$!

echo "🌐 Gradio app started with PID: $GRADIO_PID"
echo ""

# Wait for the Gradio process
wait $GRADIO_PID