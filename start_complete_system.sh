#!/bin/bash

# Complete MV-VTON System Startup Script
# This script starts both the API server and Gradio interface

set -e

echo "🚀 MV-VTON Complete System Startup"
echo "================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda or miniconda."
    exit 1
fi

# Initialize conda
eval "$(conda shell.bash hook)"

# Check if mv-vton environment exists
if ! conda env list | grep -q "mv-vton"; then
    echo "❌ mv-vton conda environment not found."
    echo "Please create it with: conda env create -f environment.yaml"
    exit 1
fi

echo "🔧 Unified Environment Setup:"
echo "   • API Server: mv-vton conda environment (FastAPI)"  
echo "   • Gradio UI: mv-vton conda environment (same environment)"
echo "   • Package Conflicts: Resolved with modern FastAPI"
echo ""

echo "📋 System Startup Options:"
echo "   1. Start API server only (mv-vton environment)"
echo "   2. Start Gradio app only (mv-vton environment, requires API server running)"
echo "   3. Start both API server and Gradio app (unified mv-vton environment)"
echo "   4. Check system status"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🤖 Starting API server in mv-vton environment..."
        conda activate mv-vton
        ./start_api_server.sh
        ;;
    2)
        echo "🌐 Starting Gradio app in mv-vton environment..."
        conda activate mv-vton
        ./start_gradio_app.sh
        ;;
    3)
        echo "🚀 Starting complete system with unified environment..."
        echo ""
        echo "🧹 Cleaning up any existing processes..."
        
        # Kill any existing processes to avoid conflicts
        pkill -f "python.*api_server" 2>/dev/null || true
        pkill -f "python.*gradio" 2>/dev/null || true
        sleep 2
        
        echo "📍 Step 1: Starting API server in mv-vton environment (background)..."
        
        # Activate conda environment and start API server in background
        conda activate mv-vton
        
        echo "🧪 Verifying high-quality checkpoint..."
        # Show checkpoint status before starting in background
        if [ -f "Frontal-View VTON/checkpoint/vitonhd.ckpt" ]; then
            checkpoint_size=$(ls -lh "Frontal-View VTON/checkpoint/vitonhd.ckpt" | awk '{print $5}')
            echo "✅ High-quality checkpoint found ($checkpoint_size) - Frontal-View VTON"
        else
            echo "❌ High-quality checkpoint not found: Frontal-View VTON/checkpoint/vitonhd.ckpt"
            echo "This will result in poor quality virtual try-on results!"
            exit 1
        fi
        
        echo "🚀 Starting high-quality API server..."
        nohup ./start_api_server.sh > api_server.log 2>&1 &
        API_PID=$!
        
        echo "🔄 Waiting for API server to initialize..."
        sleep 40
        
        # Check if API server started successfully
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            echo "✅ High-quality API server started successfully (PID: $API_PID)"
            echo "📊 Quality Level: HIGH (using Frontal-View VTON checkpoint)"
        else
            echo "❌ API server failed to start. Check api_server.log"
            exit 1
        fi
        
        echo ""
        echo "📍 Step 2: Starting Gradio interface in mv-vton environment..."
        echo "🔧 Using unified mv-vton environment for both services..."
        
        echo "🔗 API Server: http://localhost:5000 (mv-vton environment)"
        echo "🔗 Gradio Interface: http://localhost:7860 (mv-vton environment)"
        echo "🔗 API Documentation: http://localhost:5000/docs"
        echo ""
        echo "📋 Unified Environment Active:"
        echo "   • API: mv-vton conda environment (FastAPI, PyTorch, CUDA, diffusion models)"
        echo "   • UI: mv-vton conda environment (Gradio, all dependencies resolved)"
        echo ""
        echo "Press Ctrl+C to stop both services"
        echo ""
        
        # Function to cleanup on exit
        cleanup() {
            echo ""
            echo "🛑 Shutting down services..."
            
            # Kill API server process
            if [ ! -z "$API_PID" ]; then
                echo "   Stopping API server (PID: $API_PID)..."
                kill $API_PID 2>/dev/null || true
                wait $API_PID 2>/dev/null || true
            fi
            
            # Kill Gradio process if it exists
            if [ ! -z "$GRADIO_PID" ]; then
                echo "   Stopping Gradio app (PID: $GRADIO_PID)..."
                kill $GRADIO_PID 2>/dev/null || true
                wait $GRADIO_PID 2>/dev/null || true
            fi
            
            # Fallback: kill any remaining processes
            echo "   Cleaning up any remaining processes..."
            pkill -f "python mvvton_api_server.py" 2>/dev/null || true
            pkill -f "python gradio_app.py" 2>/dev/null || true
            
            # Clear port 5000 if needed
            lsof -ti:5000 | xargs -r kill -9 2>/dev/null || true
            
            echo "👋 All services stopped cleanly"
            exit 0
        }
        
        # Set trap to cleanup on script exit
        trap cleanup SIGINT SIGTERM EXIT
        
        # Ensure we're still in mv-vton environment for Gradio
        conda activate mv-vton
        
        # Start Gradio app in mv-vton environment in background and track PID
        echo "🚀 Starting full-featured Gradio interface..."
        python gradio_app.py &
        GRADIO_PID=$!
        
        echo "🌐 Gradio app started with PID: $GRADIO_PID"
        echo ""
        echo "🔗 System URLs:"
        echo "   • API Server: http://localhost:5000 (mv-vton environment)"
        echo "   • Gradio Interface: http://localhost:7860 (base environment)"
        echo ""
        
        # Wait for the Gradio process
        wait $GRADIO_PID
        ;;
    4)
        echo "🔍 Checking system status..."
        echo ""
        
        # Check API server
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            echo "✅ API Server: Running on localhost:5000"
            api_status=$(curl -s http://localhost:5000/health | python -c "import json,sys; data=json.load(sys.stdin); print(f\"Model: {'Loaded' if data['model_loaded'] else 'Not Loaded'}, Device: {data['device']}\")" 2>/dev/null || echo "Status available via /health endpoint")
            echo "   $api_status"
        else
            echo "❌ API Server: Not running on localhost:5000"
        fi
        
        # Check Gradio
        if curl -s http://localhost:7860 > /dev/null 2>&1; then
            echo "✅ Gradio Interface: Running on localhost:7860"
        else
            echo "❌ Gradio Interface: Not running on localhost:7860"
        fi
        
        # Check files
        echo ""
        echo "📁 Essential File Status:"
        files=("mvvton_api_server.py" "gradio_app.py" "integrated_preprocessing.py" "configs/viton512.yaml" "checkpoint/mvg.ckpt")
        for file in "${files[@]}"; do
            if [ -f "$file" ]; then
                echo "   ✅ $file"
            else
                echo "   ❌ $file (missing)"
            fi
        done
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
