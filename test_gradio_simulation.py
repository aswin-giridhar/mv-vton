#!/usr/bin/env python3
"""
Simulate exact Gradio app behavior to identify quality degradation cause
"""

import requests
import base64
import io
import time
from PIL import Image, ImageStat
import numpy as np
import logging

def calculate_image_quality_score(image):
    """Exact same quality function as in Gradio app"""
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
        return 50.0  # Default fallback

def simulate_gradio_exact():
    """Simulate the exact Gradio app request process"""
    
    print("🎭 Simulating Exact Gradio App Behavior")
    print("=" * 50)
    
    # Constants from Gradio app  
    API_URL = "http://localhost:5000"
    DEFAULT_DDIM_STEPS = 30
    DEFAULT_SCALE = 1.0
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 384
    TIMEOUT = 120
    
    # Load images exactly as Gradio would
    person_img_path = "/home/ubuntu/MV-VTON/assets/person/00010_00.jpg"
    cloth_img_path = "/home/ubuntu/MV-VTON/assets/cloth/00086_00.jpg"
    
    person_image = Image.open(person_img_path)
    cloth_image = Image.open(cloth_img_path)
    cloth_back_image = None  # Same as screenshot - no back image
    
    print(f"👤 Person image loaded: {person_image.size}")
    print(f"👕 Cloth image loaded: {cloth_image.size}")
    
    # Calculate input quality scores (like Gradio does)
    person_quality = calculate_image_quality_score(person_image)
    cloth_quality = calculate_image_quality_score(cloth_image)
    
    print(f"📊 Person input quality: {person_quality:.1f}/100")  
    print(f"📊 Cloth input quality: {cloth_quality:.1f}/100")
    
    try:
        # Prepare files exactly as Gradio app does
        files = {}
        
        # Convert PIL images to bytes for API transmission (EXACT same code)
        person_bytes = io.BytesIO()
        person_image.save(person_bytes, format='PNG')
        person_bytes.seek(0)
        files['person_image'] = ('person.png', person_bytes, 'image/png')
        
        cloth_bytes = io.BytesIO()
        cloth_image.save(cloth_bytes, format='PNG')
        cloth_bytes.seek(0)
        files['cloth_image'] = ('cloth.png', cloth_bytes, 'image/png')
        
        # Add back cloth image if provided (same logic)
        if cloth_back_image is not None:
            cloth_back_bytes = io.BytesIO()
            cloth_back_image.save(cloth_back_bytes, format='PNG')
            cloth_back_bytes.seek(0)
            files['cloth_back_image'] = ('cloth_back.png', cloth_back_bytes, 'image/png')
        
        # Prepare data payload (exact same)
        data = {
            'ddim_steps': int(DEFAULT_DDIM_STEPS),
            'scale': float(DEFAULT_SCALE),
            'height': int(DEFAULT_HEIGHT),
            'width': int(DEFAULT_WIDTH)
        }
        
        print(f"🚀 Sending request with params: {data}")
        
        # Record processing start time
        start_time = time.time()
        
        # Make API request with enhanced logging (exact same)
        response = requests.post(
            f"{API_URL}/try-on", 
            files=files, 
            data=data,
            timeout=TIMEOUT
        )
        
        processing_time = time.time() - start_time
        print(f"📡 API Response: {response.status_code} (took {processing_time:.2f}s)")
        
        # Handle API response (exact same logic)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                # Decode base64 image from API response
                image_data = base64.b64decode(result['result_image'])
                result_image = Image.open(io.BytesIO(image_data))
                
                # Save result
                result_image.save("/home/ubuntu/MV-VTON/gradio_simulation_result.jpg")
                
                # Calculate and log result quality (exact same)
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
                print("\n" + success_msg)
                
                print(f"\n🔍 Comparison vs Gradio App Screenshot:")
                print(f"   Gradio App (screenshot): 48.6/100")
                print(f"   This simulation:         {result_quality:.1f}/100")
                print(f"   Difference:              {result_quality - 48.6:+.1f} points")
                
                return result_image, success_msg
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"❌ API returned error: {error_msg}")
                return None, f"❌ API Error: {error_msg}"
        else:
            print(f"❌ API Error ({response.status_code}): {response.text}")
            return None, f"❌ API Error ({response.status_code}): {response.text}"
            
    except requests.exceptions.Timeout:
        return None, f"⏰ Request timeout after {TIMEOUT}s"
    except requests.exceptions.ConnectionError:
        return None, "🔌 Cannot connect to API server"
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, f"❌ Error: {str(e)}"

if __name__ == "__main__":
    simulate_gradio_exact()