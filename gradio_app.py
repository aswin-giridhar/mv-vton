#!/usr/bin/env python3
"""
MV-VTON Gradio Web Interface - Environment Isolation Version
A beautiful web interface for the MV-VTON API server

Environment Setup:
- UI: Base Python environment (this interface) 
- API: mv-vton conda environment (ML models)
- Benefit: Zero package conflicts, optimal performance

Author: Claude Code
Version: 2.0.0 (Environment Isolation)
"""

# ================================================================================
# IMPORTS AND ENVIRONMENT ISOLATION SETUP
# ================================================================================

import sys
import os

print("🔧 MV-VTON Gradio Interface - Environment Isolation Version")
print("📍 UI Environment: Base Python (avoiding conda conflicts)")
print("📍 API Environment: mv-vton conda (ML models)")
print("=" * 60)

def ensure_base_environment_packages():
    """
    Ensure required packages are available in base environment
    Environment isolation approach: install modern packages in base environment only
    """
    import subprocess
    
    print("📦 Installing required packages in base environment...")
    packages = ["gradio", "requests", "pillow", "numpy"]
    
    try:
        for package in packages:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", package
            ])
        print("✅ All packages installed successfully in base environment!")
        return True
    except Exception as e:
        print(f"⚠️ Package installation warning: {e}")
        return False

# Ensure packages are installed in base environment
ensure_base_environment_packages()

# Import gradio with environment isolation support
try:
    import gradio as gr
    print(f"✅ Gradio imported successfully in base environment! Version: {gr.__version__}")
except ImportError as e:
    print(f"⚠️ Gradio import failed: {e}")
    print("🔧 Attempting to install Gradio in base environment...")
    
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        import gradio as gr
        print(f"✅ Gradio installed and imported! Version: {gr.__version__}")
    except Exception as install_error:
        print(f"❌ Failed to install/import Gradio: {install_error}")
        print("\n🔄 Fallback: Use simple interface")
        print("Run: python gradio_app.py")
        sys.exit(1)

# Standard library imports
import time
import json
import os
from datetime import datetime

# Third-party imports
import requests
import base64
import numpy as np
from PIL import Image, ImageStat
import io
import logging

# Enhanced logging setup for Gradio app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio_app.log'),
        logging.StreamHandler()
    ]
)
gradio_logger = logging.getLogger('gradio_app')

# ================================================================================
# CONFIGURATION
# ================================================================================

# API Configuration
API_URL = "http://localhost:5000"
TIMEOUT = 120  # 2 minutes timeout for API calls

# Default parameter values
DEFAULT_DDIM_STEPS = 30
DEFAULT_SCALE = 1.0
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 384

# ================================================================================
# API INTERACTION FUNCTIONS
# ================================================================================

def check_api_health():
    """
    Check if API server is running and get detailed status information
    
    Returns:
        str: Formatted status message with server health details
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            status_parts = [
                f"✅ Status: {data['status'].title()}",
                f"Model: {'✅ Loaded' if data['model_loaded'] else '❌ Not Loaded'}",
                f"Device: {data.get('device', 'Unknown')}",
                f"CUDA: {'✅ Available' if data.get('cuda_available') else '❌ Not Available'}"
            ]
            return " | ".join(status_parts)
        else:
            return f"❌ API Error: HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "🔌 API server not reachable. Start with: ./start_api_server.sh"
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

def get_model_info():
    """
    Get detailed model information from the API server
    
    Returns:
        str: Formatted model information string
    """
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            info_parts = [
                f"Model: {data.get('model_type', 'Unknown')}",
                f"Size: {data.get('image_size', 'Unknown')}",
                f"Parameters: {data.get('model_parameters', 0):,}",
                f"Channels: {data.get('channels', 'Unknown')}"
            ]
            return " | ".join(info_parts)
        else:
            return "Model info unavailable"
    except:
        return "Model info unavailable"

def calculate_image_quality_score(image):
    """Calculate a simple quality score for an image"""
    if image is None:
        return 0
    try:
        stat = ImageStat.Stat(image)
        # Simple quality metric based on contrast and detail
        contrast = np.std(np.array(image.convert('L')))
        detail = np.mean(stat.stddev)
        return min(100, (contrast + detail) / 3)
    except:
        return 0

def virtual_try_on(person_image, cloth_image, cloth_back_image=None, 
                   ddim_steps=DEFAULT_DDIM_STEPS, scale=DEFAULT_SCALE, 
                   height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):
    """
    Main function to call the MV-VTON API and get virtual try-on results
    
    Args:
        person_image (PIL.Image): The person image
        cloth_image (PIL.Image): The front clothing image
        cloth_back_image (PIL.Image, optional): The back clothing image
        ddim_steps (int): Number of DDIM sampling steps
        scale (float): Guidance scale for generation
        height (int): Output image height
        width (int): Output image width
        
    Returns:
        tuple: (result_image, status_message)
    """
    gradio_logger.info("=" * 60)
    gradio_logger.info("🎭 NEW GRADIO VIRTUAL TRY-ON REQUEST")
    gradio_logger.info("=" * 60)
    
    # Input validation with logging
    if person_image is None:
        gradio_logger.warning("❌ No person image provided")
        return None, "❌ Please upload a person image"
    
    if cloth_image is None:
        gradio_logger.warning("❌ No cloth image provided")
        return None, "❌ Please upload a clothing image"
    
    # Log input image quality
    person_quality = calculate_image_quality_score(person_image)
    cloth_quality = calculate_image_quality_score(cloth_image)
    gradio_logger.info(f"📊 Input quality scores:")
    gradio_logger.info(f"   Person image: {person_quality:.1f}/100 (size: {person_image.size})")
    gradio_logger.info(f"   Cloth image: {cloth_quality:.1f}/100 (size: {cloth_image.size})")
    gradio_logger.info(f"⚙️ Parameters: steps={ddim_steps}, scale={scale}, size={width}x{height}")
    
    try:
        # Prepare API request data
        files = {}
        data = {
            'ddim_steps': int(ddim_steps),
            'scale': float(scale),
            'height': int(height),
            'width': int(width)
        }
        
        # Convert PIL images to bytes for API transmission
        person_bytes = io.BytesIO()
        person_image.save(person_bytes, format='PNG')
        person_bytes.seek(0)
        files['person_image'] = ('person.png', person_bytes, 'image/png')
        
        cloth_bytes = io.BytesIO()
        cloth_image.save(cloth_bytes, format='PNG')
        cloth_bytes.seek(0)
        files['cloth_image'] = ('cloth.png', cloth_bytes, 'image/png')
        
        # Add back cloth image if provided
        if cloth_back_image is not None:
            cloth_back_bytes = io.BytesIO()
            cloth_back_image.save(cloth_back_bytes, format='PNG')
            cloth_back_bytes.seek(0)
            files['cloth_back_image'] = ('cloth_back.png', cloth_back_bytes, 'image/png')
        
        # Record processing start time
        start_time = time.time()
        
        # Make API request with enhanced logging
        gradio_logger.info(f"🚀 Sending request to API server...")
        response = requests.post(
            f"{API_URL}/try-on", 
            files=files, 
            data=data,
            timeout=TIMEOUT
        )
        
        processing_time = time.time() - start_time
        gradio_logger.info(f"📡 API Response: {response.status_code} (took {processing_time:.2f}s)")
        
        # Handle API response
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                # Decode base64 image from API response
                image_data = base64.b64decode(result['result_image'])
                result_image = Image.open(io.BytesIO(image_data))
                
                # Calculate and log result quality
                result_quality = calculate_image_quality_score(result_image)
                quality_improvement = result_quality - person_quality
                
                gradio_logger.info(f"📊 Output quality: {result_quality:.1f}/100")
                if quality_improvement > 0:
                    gradio_logger.info(f"📈 Quality improved by {quality_improvement:.1f} points")
                    quality_status = "📈 Quality improved!"
                elif quality_improvement < -5:
                    gradio_logger.warning(f"📉 Quality degraded by {abs(quality_improvement):.1f} points")
                    quality_status = "📉 Quality degraded"
                else:
                    quality_status = "📊 Quality maintained"
                
                success_msg = f"✅ Success! Processing time: {processing_time:.2f}s\n🎯 Result: {result_image.size}\n{quality_status} ({result_quality:.1f}/100)"
                gradio_logger.info(f"✅ Virtual try-on completed successfully!")
                return result_image, success_msg
            else:
                error_msg = result.get('error', 'Unknown error')
                gradio_logger.error(f"❌ API returned error: {error_msg}")
                return None, f"❌ API Error: {error_msg}"
        else:
            # Handle HTTP errors
            error_data = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get('error', 'Unknown error')
            except:
                error_msg = error_data[:200] + "..." if len(error_data) > 200 else error_data
            
            return None, f"❌ API Error ({response.status_code}): {error_msg}"
            
    except requests.exceptions.Timeout:
        return None, f"⏰ Request timeout after {TIMEOUT}s. Try reducing DDIM steps or image size."
    except requests.exceptions.ConnectionError:
        return None, "🔌 Cannot connect to API server. Make sure it's running on localhost:5000"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ================================================================================
# UI STYLING
# ================================================================================

custom_css = """
.gradio-container {
    max-width: 1200px !important;
}

.image-preview {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.status-box {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}

.parameter-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 0.9em;
}
"""

# ================================================================================
# GRADIO INTERFACE DEFINITION
# ================================================================================

def create_gradio_interface():
    """
    Create and configure the Gradio web interface
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(
        title="MV-VTON: Multi-View Virtual Try-On", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # ============================================================================
        # HEADER SECTION
        # ============================================================================
        gr.Markdown("""
        <div style="text-align: center;">
            <h1>🔥 MV-VTON: Multi-View Virtual Try-On</h1>
            <p style="font-size: 1.2em; color: #666;">
                Experience the future of fashion with AI-powered virtual try-on technology
            </p>
            <div style="background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin: 20px 0; font-size: 0.9em;">
                <strong>🚀 Enhanced Preprocessing Active:</strong> OpenPose 18-point detection • SCHP Human Parsing • Person-Agnostic Generation • Improved Quality
            </div>
        </div>
        """)
        
        # ============================================================================
        # API STATUS SECTION
        # ============================================================================
        with gr.Row():
            with gr.Column(scale=2):
                api_status = gr.Textbox(
                    label="🔍 API Server Status", 
                    value="Checking connection...", 
                    interactive=False,
                    elem_classes=["status-box"]
                )
            with gr.Column(scale=2):
                model_info = gr.Textbox(
                    label="🤖 Model Information", 
                    value="Loading model info...", 
                    interactive=False,
                    elem_classes=["status-box"]
                )
            with gr.Column(scale=1):
                status_button = gr.Button(
                    "🔄 Refresh Status", 
                    variant="secondary",
                    size="sm"
                )
        
        # ============================================================================
        # MAIN INTERFACE SECTION
        # ============================================================================
        gr.Markdown("---")
        
        with gr.Row(equal_height=True):
            # Input Column
            with gr.Column():
                gr.Markdown("### 📸 Input Images")
                
                person_input = gr.Image(
                    label="👤 Person Image",
                    type="pil",
                    # height=350,  # REMOVED - causes image compression
                    elem_classes=["image-preview"]
                )
                
                gr.Markdown("**Sample Person Images**")
                person_gallery = gr.Gallery(
                    label="Click to select",
                    show_label=False,
                    elem_id="person_gallery",
                    columns=3,
                    rows=2,
                    height=180,
                    object_fit="cover"
                )
            
            # Garment Column  
            with gr.Column():
                gr.Markdown("### 👕 Clothing Selection")
                
                cloth_input = gr.Image(
                    label="👕 Clothing (Front View)",
                    type="pil",
                    # height=300,  # REMOVED - causes image compression
                    elem_classes=["image-preview"]
                )
                
                
                cloth_back_input = gr.Image(
                    label="👕 Clothing (Back View) - Optional",
                    type="pil",
                    # height=150,  # REMOVED - causes image compression
                    elem_classes=["image-preview"]
                )
                
                gr.Markdown("**Sample Garment Images**")
                cloth_gallery = gr.Gallery(
                    label="Click to select",
                    show_label=False,
                    elem_id="cloth_gallery", 
                    columns=4,
                    rows=2,
                    height=120,
                    object_fit="cover"
                )
            
            # Output Column
            with gr.Column():
                gr.Markdown("### 🎨 Generated Result")
                
                result_image = gr.Image(
                    label="🖼️ Virtual Try-On Result",
                    type="pil",
                    height=400,
                    elem_classes=["image-preview"]
                )
                
                submit_btn = gr.Button(
                    "✨ Generate Virtual Try-On", 
                    variant="primary", 
                    size="lg"
                )
                
                result_message = gr.Textbox(
                    label="📊 Processing Status",
                    value="Upload images and click 'Generate Virtual Try-On' to start",
                    interactive=False,
                    lines=3
                )
        
        # Advanced Parameters Section (moved below main interface)
        with gr.Accordion("⚙️ Advanced Parameters", open=False):
            gr.Markdown("Adjust these parameters to control the generation process:")
            
            with gr.Row():
                ddim_steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=DEFAULT_DDIM_STEPS,
                    step=5,
                    label="🔄 DDIM Steps (Higher = Better Quality, Slower)",
                    info="Number of denoising steps. More steps = better quality but longer processing time."
                )
                
                scale = gr.Slider(
                    minimum=0.5,
                    maximum=3.0,
                    value=DEFAULT_SCALE,
                    step=0.1,
                    label="🎯 Guidance Scale",
                    info="Controls how closely the model follows the input. Higher = more faithful to input."
                )
            
            with gr.Row():
                height = gr.Slider(
                    minimum=256,
                    maximum=768,
                    value=DEFAULT_HEIGHT,
                    step=64,
                    label="📏 Output Height"
                )
                width = gr.Slider(
                    minimum=256,
                    maximum=512,
                    value=DEFAULT_WIDTH,
                    step=64,
                    label="📐 Output Width"
                )
        
        # ============================================================================
        # INSTRUCTIONS AND EXAMPLES SECTION
        # ============================================================================
        gr.Markdown("""
        ### 📚 Usage Instructions
        
        1. **Upload a person image**: Clear, front-facing photo works best
        2. **Upload clothing image(s)**: Front view is required, back view is optional
        3. **Adjust parameters** (optional): Use advanced settings for fine-tuning
        4. **Click Generate**: Wait for the AI to process your request
        
        **🚀 Enhanced Processing Pipeline:**
        - **18-point OpenPose** automatically detects body structure
        - **SCHP Human Parsing** segments body parts with ATR dataset precision
        - **Person-Agnostic Generation** removes clothing regions properly
        - **DressCode Methodology** ensures research-grade quality
        
        **Tips for better results:**
        - Use high-quality, well-lit images
        - Person should be facing forward with clear pose
        - Clothing should be clearly visible against background
        - Enhanced preprocessing now handles complex poses automatically
        """)
        
        
        # ============================================================================
        # EVENT HANDLERS
        # ============================================================================
        
        # Load sample images for galleries
        def load_sample_images():
            person_samples = []
            cloth_samples = []
            
            # Load all person samples from assets directory
            person_dir = '/home/ubuntu/MV-VTON/assets/person/'
            if os.path.exists(person_dir):
                person_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                person_files.sort()  # Sort for consistent order
                for filename in person_files[:9]:  # Limit for gallery performance
                    person_samples.append(os.path.join(person_dir, filename))
            
            # Load all cloth samples from assets directory  
            cloth_dir = '/home/ubuntu/MV-VTON/assets/cloth/'
            if os.path.exists(cloth_dir):
                cloth_files = [f for f in os.listdir(cloth_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                cloth_files.sort()  # Sort for consistent order
                for filename in cloth_files[:16]:  # Limit for gallery performance
                    cloth_samples.append(os.path.join(cloth_dir, filename))
                    
            return person_samples, cloth_samples
        
        # Gallery selection handlers
        def select_person_from_gallery(evt: gr.SelectData):
            person_samples, _ = load_sample_images()
            if evt.index < len(person_samples):
                return Image.open(person_samples[evt.index])
            return None
        
        def select_cloth_from_gallery(evt: gr.SelectData):
            _, cloth_samples = load_sample_images()
            if evt.index < len(cloth_samples):
                return Image.open(cloth_samples[evt.index])
            return None
        
        # Status refresh handler
        def refresh_status():
            """Refresh both API status and model info"""
            api_status_text = check_api_health()
            model_info_text = get_model_info()
            return api_status_text, model_info_text
        
        # Set up galleries on load
        demo.load(fn=lambda: load_sample_images(), outputs=[person_gallery, cloth_gallery])
        demo.load(fn=refresh_status, outputs=[api_status, model_info])
        
        # Gallery click handlers
        person_gallery.select(fn=select_person_from_gallery, outputs=[person_input])
        cloth_gallery.select(fn=select_cloth_from_gallery, outputs=[cloth_input])
        
        # Main try-on button click handler
        submit_btn.click(
            fn=virtual_try_on,
            inputs=[person_input, cloth_input, cloth_back_input, ddim_steps, scale, height, width],
            outputs=[result_image, result_message],
            show_progress=True
        )
        
        # Status refresh handler
        status_button.click(
            fn=refresh_status,
            outputs=[api_status, model_info]
        )
        
        # ============================================================================
        # FOOTER SECTION
        # ============================================================================
        gr.Markdown("""
        <div class="footer">
            <p>🚀 <strong>MV-VTON API Server</strong> | Built with ❤️ using Gradio</p>
            <p>Make sure the API server is running: <code>./start_api_server.sh</code></p>
            <p><small>For technical details, see API_USAGE.md</small></p>
        </div>
        """)
    
    return demo

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """
    Main function to launch the Gradio web interface with environment isolation
    """
    print("🚀 Starting MV-VTON Gradio Interface (Environment Isolation)...")
    print(f"🔗 API Server: {API_URL} (mv-vton conda environment)")
    print("🌐 Web Interface: http://localhost:7860 (base environment)")
    print("\n" + "="*60)
    print("📋 ENVIRONMENT ISOLATION ACTIVE:")
    print("   ✅ API Server: mv-vton conda environment (ML models)")
    print("   ✅ Gradio UI: base Python environment (web framework)")
    print("   ✅ Package Conflicts: Resolved via environment separation")
    print("\n📋 PREREQUISITES:")
    print("1. API server should be running in mv-vton environment:")
    print("   conda activate mv-vton && ./start_api_server.sh")
    print("2. API accessible at localhost:5000")
    print("="*60 + "\n")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Set to True for public Gradio link
        debug=False,
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )

if __name__ == "__main__":
    main()
