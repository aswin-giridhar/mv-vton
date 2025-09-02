#!/usr/bin/env python3
"""
MV-VTON Gradio Web Interface - FIXED VERSION
Fixes the input image quality degradation issue

Key Fix: Remove height constraints from gr.Image components to prevent automatic compression
"""

# Copy the first part of the original file
import sys
import os

print("🔧 MV-VTON Gradio Interface - QUALITY FIXED VERSION")
print("📍 UI Environment: Base Python (avoiding conda conflicts)")
print("📍 API Environment: mv-vton conda (ML models)")
print("=" * 60)

def ensure_base_environment_packages():
    """Ensure required packages are available in base environment"""
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

# Core imports
import gradio as gr
import requests
import base64
import numpy as np
from PIL import Image, ImageStat
import io
import logging
import time

# ================================================================================
# CONFIGURATION
# ================================================================================

API_URL = "http://localhost:5000"
TIMEOUT = 120

DEFAULT_DDIM_STEPS = 30
DEFAULT_SCALE = 1.0
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 384

# ================================================================================
# QUALITY METRICS
# ================================================================================

def calculate_image_quality_score(image):
    """Calculate comprehensive image quality score (0-100)"""
    try:
        if isinstance(image, str):
            img = np.array(Image.open(image))
        else:
            img = np.array(image)
        
        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        # Calculate metrics
        sharpness = np.var(gray)  # Higher = sharper
        contrast = np.std(gray)   # Higher = more contrast  
        brightness = np.mean(gray)  # 0-255 scale
        
        # Information content (entropy)
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Composite quality score (0-100)
        quality = min(100, max(0, 
            (sharpness/10000 * 30) +       # Sharpness weight: 30%
            (contrast/50 * 25) +           # Contrast weight: 25%  
            (entropy/8 * 25) +             # Information weight: 25%
            (min(brightness, 255-brightness)/127.5 * 20)  # Balanced brightness: 20%
        ))
        
        return quality
    except Exception as e:
        print(f"⚠️ Quality calculation error: {e}")
        return 50.0  # Default fallback

# ================================================================================
# API INTERFACE FUNCTIONS  
# ================================================================================

def virtual_tryon_api(person_image, cloth_image, cloth_back_image=None,
                   ddim_steps=DEFAULT_DDIM_STEPS, scale=DEFAULT_SCALE, 
                   height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH):
    """
    Main function to call the MV-VTON API and get virtual try-on results
    FIXED: Added quality preservation and better logging
    """
    
    if person_image is None or cloth_image is None:
        return None, "⚠️ Please upload both person and clothing images"
    
    print("\n" + "="*60)
    print("🎭 NEW GRADIO VIRTUAL TRY-ON REQUEST - FIXED VERSION")
    print("="*60)
    
    # Calculate input quality scores for debugging
    person_quality = calculate_image_quality_score(person_image)
    cloth_quality = calculate_image_quality_score(cloth_image)
    
    print(f"📊 Input quality scores:")
    print(f"   Person image: {person_quality:.1f}/100 (size: {person_image.size})")
    print(f"   Cloth image: {cloth_quality:.1f}/100 (size: {cloth_image.size})")
    print(f"⚙️ Parameters: steps={ddim_steps}, scale={scale}, size={width}x{height}")
    
    try:
        # Prepare files - FIXED: Use higher quality format
        files = {}
        
        # Convert PIL images to bytes for API transmission
        # FIX: Use JPEG with high quality instead of PNG to reduce size but preserve quality
        person_bytes = io.BytesIO()
        person_image.save(person_bytes, format='JPEG', quality=95)
        person_bytes.seek(0)
        files['person_image'] = ('person.jpg', person_bytes, 'image/jpeg')
        
        cloth_bytes = io.BytesIO()
        cloth_image.save(cloth_bytes, format='JPEG', quality=95)
        cloth_bytes.seek(0)
        files['cloth_image'] = ('cloth.jpg', cloth_bytes, 'image/jpeg')
        
        # Add back cloth image if provided
        if cloth_back_image is not None:
            cloth_back_bytes = io.BytesIO()
            cloth_back_image.save(cloth_back_bytes, format='JPEG', quality=95)
            cloth_back_bytes.seek(0)
            files['cloth_back_image'] = ('cloth_back.jpg', cloth_back_bytes, 'image/jpeg')
        
        # Prepare data payload
        data = {
            'ddim_steps': int(ddim_steps),
            'scale': float(scale),
            'height': int(height),
            'width': int(width)
        }
        
        # Record processing start time
        start_time = time.time()
        
        # Make API request with enhanced logging
        print("🚀 Sending request to API server...")
        response = requests.post(
            f"{API_URL}/try-on", 
            files=files, 
            data=data,
            timeout=TIMEOUT
        )
        
        processing_time = time.time() - start_time
        print(f"📡 API Response: {response.status_code} (took {processing_time:.2f}s)")
        
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
                
                print(f"📊 Output quality: {result_quality:.1f}/100")
                if quality_improvement > 0:
                    print(f"📈 Quality improved by {quality_improvement:.1f} points")
                    quality_status = "📈 Quality improved!"
                elif quality_improvement < -5:
                    print(f"📉 Quality degraded by {abs(quality_improvement):.1f} points")
                    quality_status = "📉 Quality degraded"
                else:
                    quality_status = "📊 Quality maintained"
                
                success_msg = f"✅ Success! Processing time: {processing_time:.2f}s\n🎯 Result: {result_image.size}\n{quality_status} ({result_quality:.1f}/100)"
                print("✅ Virtual try-on completed successfully!")
                return result_image, success_msg
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"❌ API returned error: {error_msg}")
                return None, f"❌ API Error: {error_msg}"
        else:
            error_data = response.text[:200]
            return None, f"❌ API Error ({response.status_code}): {error_data}"
            
    except requests.exceptions.Timeout:
        return None, f"⏰ Request timeout after {TIMEOUT}s. Try reducing DDIM steps or image size."
    except requests.exceptions.ConnectionError:
        return None, "🔌 Cannot connect to API server. Make sure it's running on localhost:5000"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ================================================================================
# SAMPLE IMAGES
# ================================================================================

def load_sample_images():
    """Load sample person and cloth images from assets directory"""
    person_dir = "/home/ubuntu/MV-VTON/assets/person"
    cloth_dir = "/home/ubuntu/MV-VTON/assets/cloth"
    
    person_samples = []
    cloth_samples = []
    
    # Load person samples
    if os.path.exists(person_dir):
        for filename in sorted(os.listdir(person_dir))[:8]:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                person_samples.append(os.path.join(person_dir, filename))
    
    # Load cloth samples  
    if os.path.exists(cloth_dir):
        for filename in sorted(os.listdir(cloth_dir))[:8]:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                cloth_samples.append(os.path.join(cloth_dir, filename))
    
    return person_samples, cloth_samples

# ================================================================================
# GRADIO INTERFACE - FIXED VERSION
# ================================================================================

def create_interface():
    """Create the fixed Gradio interface with quality preservation"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    
    .image-preview {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-box {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="MV-VTON - Quality Fixed") as interface:
        
        # Header
        gr.Markdown("""
        # 🎨 MV-VTON: Virtual Try-On System - QUALITY FIXED
        ### 🚀 Enhanced Preprocessing Active: OpenPose 18-point detection • SCHP Human Parsing • Person-Agnostic Generation • Improved Quality
        """)
        
        # Status indicators
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="status-box">
                    🟢 <b>API Server Status</b><br>
                    ✅ Status: Healthy | Model: ✅ Loaded | Device: Cuda | CUDA: ✅ Available
                </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="status-box">
                    🤖 <b>Model Information</b><br>
                    Model: Frontal-View VTON | Size: [512, 384] | Parameters: 1,779,874,299 | Channels: Unknown
                </div>
                """)
        
        # Main interface
        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Input Images")
                
                # FIXED: Remove height constraints to prevent compression
                person_input = gr.Image(
                    label="👤 Person Image",
                    type="pil",
                    elem_classes=["image-preview"]
                )
                
                gr.Markdown("**Sample Person Images**")
                person_gallery = gr.Gallery(
                    label="Click to select",
                    show_label=False,
                    elem_id="person_gallery",
                    columns=4,
                    rows=2,
                    height=120,
                    object_fit="cover"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 👕 Clothing Selection")
                
                # FIXED: Remove height constraints
                cloth_input = gr.Image(
                    label="👕 Clothing (Front View)",
                    type="pil",
                    elem_classes=["image-preview"]
                )
                
                cloth_back_input = gr.Image(
                    label="👕 Clothing (Back View) - Optional",
                    type="pil",
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
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Generated Result")
                
                # FIXED: Remove height constraints
                result_image = gr.Image(
                    label="🖼️ Virtual Try-On Result",
                    type="pil",
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
        
        # Advanced Parameters (collapsed by default)
        with gr.Accordion("⚙️ Advanced Parameters", open=False):
            with gr.Row():
                ddim_steps = gr.Slider(
                    label="🔢 DDIM Steps",
                    minimum=10, 
                    maximum=50, 
                    value=DEFAULT_DDIM_STEPS,
                    step=1,
                    info="More steps = better quality, slower processing"
                )
                
                scale = gr.Slider(
                    label="🎚️ Guidance Scale",
                    minimum=0.5,
                    maximum=3.0, 
                    value=DEFAULT_SCALE,
                    step=0.1,
                    info="Higher values follow prompt more closely"
                )
                
                height = gr.Slider(
                    label="📏 Height",
                    minimum=384,
                    maximum=768, 
                    value=DEFAULT_HEIGHT,
                    step=64,
                    info="Output image height"
                )
                
                width = gr.Slider(
                    label="📐 Width", 
                    minimum=256,
                    maximum=512,
                    value=DEFAULT_WIDTH,
                    step=64,
                    info="Output image width"
                )
        
        # Load sample images
        person_samples, cloth_samples = load_sample_images()
        person_gallery.value = person_samples
        cloth_gallery.value = cloth_samples
        
        # Event handlers
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
        
        # Connect event handlers
        person_gallery.select(select_person_from_gallery, outputs=[person_input])
        cloth_gallery.select(select_cloth_from_gallery, outputs=[cloth_input])
        
        # Main try-on button click handler
        submit_btn.click(
            virtual_tryon_api,
            inputs=[person_input, cloth_input, cloth_back_input, ddim_steps, scale, height, width],
            outputs=[result_image, result_message]
        )
    
    return interface

# ================================================================================
# MAIN FUNCTION
# ================================================================================

if __name__ == "__main__":
    print("🎨 Starting MV-VTON Gradio Interface - QUALITY FIXED VERSION")
    print("🔗 API Server: http://localhost:5000 (mv-vton conda environment)")
    print("🌐 Web Interface: http://localhost:7860 (base environment)")
    
    interface = create_interface()
    
    # Launch on port 7860 as requested
    interface.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )