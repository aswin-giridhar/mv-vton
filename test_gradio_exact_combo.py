#!/usr/bin/env python3
"""
Test the exact same image combination from Gradio app via API
to confirm if the quality difference is consistent
"""

import requests
import json
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def calculate_image_quality(image):
    """Calculate comprehensive quality metrics for comparison"""
    if isinstance(image, str):
        img = np.array(Image.open(image))
    else:
        img = np.array(image)
    
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
    
    # Calculate metrics
    sharpness = np.var(gray)  # Higher = sharper
    contrast = np.std(gray)   # Higher = more contrast
    brightness = np.mean(gray)  # 0-255 scale
    entropy = -np.sum((np.histogram(gray, bins=256)[0]/gray.size) * 
                     np.log2(np.histogram(gray, bins=256)[0]/gray.size + 1e-10))
    
    # Composite quality score (0-100)
    quality = min(100, max(0, 
        (sharpness/10000 * 30) +       # Sharpness weight: 30%
        (contrast/50 * 25) +           # Contrast weight: 25%  
        (entropy/8 * 25) +             # Information weight: 25%
        (min(brightness, 255-brightness)/127.5 * 20)  # Balanced brightness: 20%
    ))
    
    return {
        'sharpness': sharpness,
        'contrast': contrast,
        'brightness': brightness,
        'entropy': entropy,
        'quality': quality
    }

def test_exact_gradio_combination():
    """Test the exact image combination from Gradio screenshot"""
    api_url = "http://localhost:5000"
    
    # Exact same files as used in Gradio
    person_img = "/home/ubuntu/MV-VTON/assets/person/00010_00.jpg"
    cloth_img = "/home/ubuntu/MV-VTON/assets/cloth/00086_00.jpg"
    
    print("🧪 Testing Exact Gradio Combination via API")
    print(f"👤 Person: {person_img}")  
    print(f"👕 Cloth: {cloth_img}")
    print("=" * 60)
    
    # Verify files exist
    if not os.path.exists(person_img):
        print(f"❌ Person image not found: {person_img}")
        return
    if not os.path.exists(cloth_img):
        print(f"❌ Cloth image not found: {cloth_img}")
        return
        
    try:
        # Test via API with same parameters as Gradio default
        files = {
            'person_image': ('person.jpg', open(person_img, 'rb'), 'image/jpeg'),
            'cloth_image': ('cloth.jpg', open(cloth_img, 'rb'), 'image/jpeg')
        }
        data = {
            'ddim_steps': 30,    # Default from Gradio
            'scale': 1.0,        # Default from Gradio  
            'height': 512,
            'width': 384
        }
        
        print("🚀 Making API request...")
        response = requests.post(f"{api_url}/try-on", files=files, data=data, timeout=120)
        
        # Close file handles
        files['person_image'][1].close()
        files['cloth_image'][1].close()
        
        if response.status_code == 200:
            # Parse response
            result = response.json()
            
            if 'result_image' in result:
                # Decode and save result
                img_data = base64.b64decode(result['result_image'])
                result_img = Image.open(BytesIO(img_data))
                
                output_path = "/home/ubuntu/MV-VTON/api_test_exact_gradio_combo.jpg"
                result_img.save(output_path)
                
                # Calculate quality
                quality_metrics = calculate_image_quality(result_img)
                
                print("✅ API Test Results:")
                print(f"📊 Quality Score: {quality_metrics['quality']:.1f}/100")
                print(f"🖼️  Result saved: {output_path}")
                print()
                print("📈 Detailed Metrics:")
                print(f"   Sharpness: {quality_metrics['sharpness']:.0f}")
                print(f"   Contrast: {quality_metrics['contrast']:.1f}")
                print(f"   Brightness: {quality_metrics['brightness']:.1f}")
                print(f"   Entropy: {quality_metrics['entropy']:.2f}")
                print()
                
                # Compare with expected Gradio result
                gradio_quality = 48.6  # From screenshot
                api_quality = quality_metrics['quality']
                
                print("🔍 Comparison Analysis:")
                print(f"   Gradio App Quality: {gradio_quality}/100")
                print(f"   API Direct Quality: {api_quality:.1f}/100")
                print(f"   Difference: {api_quality - gradio_quality:+.1f} points")
                
                if abs(api_quality - gradio_quality) < 5:
                    print("✅ Quality levels are consistent - same underlying model")
                else:
                    print("⚠️  Significant quality difference detected")
                
                return True
            else:
                print(f"❌ No result image in response: {result}")
                return False
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_exact_gradio_combination()