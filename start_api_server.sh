#!/bin/bash

# MV-VTON API Server Startup Script
# This script activates the conda environment and starts the API server

set -e  # Exit on any error

echo "🚀 MV-VTON API Server Startup Script"
echo "====================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda or miniconda."
    exit 1
fi

# Initialize conda for shell script
eval "$(conda shell.bash hook)"

# Check if mv-vton environment exists
if ! conda env list | grep -q "mv-vton"; then
    echo "❌ mv-vton conda environment not found."
    echo "Please create it with: conda env create -f environment.yaml"
    exit 1
fi

echo "🔧 Activating mv-vton conda environment..."
conda activate mv-vton

echo "🧪 Checking high-quality checkpoint..."

# Check for high-quality Frontal-View VTON files
if [ ! -f "Frontal-View VTON/checkpoint/vitonhd.ckpt" ]; then
    echo "❌ High-quality checkpoint not found: Frontal-View VTON/checkpoint/vitonhd.ckpt"
    echo "This is required for best quality results."
    exit 1
fi

if [ ! -f "Frontal-View VTON/configs/viton512.yaml" ]; then
    echo "❌ Config file not found: Frontal-View VTON/configs/viton512.yaml"
    exit 1
fi

echo "✅ High-quality checkpoint found ($(ls -lh 'Frontal-View VTON/checkpoint/vitonhd.ckpt' | awk '{print $5}'))"
echo ""
echo "✅ High-quality setup verified! Starting API server..."
echo "🌐 Server will be available at: http://localhost:5000"
echo "📖 Health check: http://localhost:5000/health"
echo "📋 Quality: HIGH (Frontal-View VTON checkpoint)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "====================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down API server..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
        wait $API_PID 2>/dev/null || true
    fi
    # Also kill any remaining mvvton_api_server processes
    pkill -f "python mvvton_api_server.py" 2>/dev/null || true
    echo "👋 API server stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start the high-quality API server in background and get its PID
python mvvton_api_server.py &
API_PID=$!

echo "🤖 High-quality API server started with PID: $API_PID"
echo "📊 Expected quality: 77+ /100 (using Frontal-View VTON checkpoint)"
echo ""

# Wait for the API server process
wait $API_PID