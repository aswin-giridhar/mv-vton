#!/usr/bin/env python3
"""
Test if PNG conversion is causing the quality degradation in Gradio app
"""

import requests
import base64
import io
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

def test_png_vs_jpeg():
    """Test API with JPEG vs PNG conversion like Gradio"""
    api_url = "http://localhost:5000"
    
    # Same files as before
    person_img_path = "/home/ubuntu/MV-VTON/assets/person/00010_00.jpg"
    cloth_img_path = "/home/ubuntu/MV-VTON/assets/cloth/00086_00.jpg"
    
    print("🧪 Testing PNG Conversion Impact")
    print("=" * 50)
    
    # Load original images
    person_img = Image.open(person_img_path)
    cloth_img = Image.open(cloth_img_path)
    
    results = {}
    
    # Test 1: Direct JPEG (like my previous API test)
    print("\n🔸 Test 1: Direct JPEG (Original Method)")
    try:
        files = {
            'person_image': ('person.jpg', open(person_img_path, 'rb'), 'image/jpeg'),
            'cloth_image': ('cloth.jpg', open(cloth_img_path, 'rb'), 'image/jpeg')
        }
        data = {'ddim_steps': 30, 'scale': 1.0, 'height': 512, 'width': 384}
        
        response = requests.post(f"{api_url}/try-on", files=files, data=data, timeout=120)
        files['person_image'][1].close()
        files['cloth_image'][1].close()
        
        if response.status_code == 200:
            result = response.json()
            img_data = base64.b64decode(result['result_image'])
            result_img = Image.open(io.BytesIO(img_data))
            result_img.save("/home/ubuntu/MV-VTON/test_jpeg_method.jpg")
            
            quality = calculate_image_quality(result_img)['quality']
            results['jpeg_direct'] = quality
            print(f"   Quality: {quality:.1f}/100")
        else:
            print(f"   Error: {response.status_code}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: PNG Conversion (like Gradio app)
    print("\n🔸 Test 2: PNG Conversion (Gradio Method)")
    try:
        # Convert to PNG format like Gradio
        person_bytes = io.BytesIO()
        person_img.save(person_bytes, format='PNG')
        person_bytes.seek(0)
        
        cloth_bytes = io.BytesIO()
        cloth_img.save(cloth_bytes, format='PNG')
        cloth_bytes.seek(0)
        
        files = {
            'person_image': ('person.png', person_bytes, 'image/png'),
            'cloth_image': ('cloth.png', cloth_bytes, 'image/png')
        }
        data = {'ddim_steps': 30, 'scale': 1.0, 'height': 512, 'width': 384}
        
        response = requests.post(f"{api_url}/try-on", files=files, data=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            img_data = base64.b64decode(result['result_image'])
            result_img = Image.open(io.BytesIO(img_data))
            result_img.save("/home/ubuntu/MV-VTON/test_png_method.jpg")
            
            quality = calculate_image_quality(result_img)['quality']
            results['png_conversion'] = quality
            print(f"   Quality: {quality:.1f}/100")
        else:
            print(f"   Error: {response.status_code}")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Analysis
    print("\n📊 Analysis:")
    if 'jpeg_direct' in results and 'png_conversion' in results:
        jpeg_qual = results['jpeg_direct']
        png_qual = results['png_conversion']
        diff = png_qual - jpeg_qual
        
        print(f"   JPEG Direct:     {jpeg_qual:.1f}/100")
        print(f"   PNG Conversion:  {png_qual:.1f}/100")
        print(f"   Difference:      {diff:+.1f} points")
        
        if abs(diff) < 2:
            print("   ✅ PNG conversion has minimal impact")
        elif diff < -5:
            print("   ⚠️  PNG conversion significantly degrades quality")
        else:
            print("   📈 PNG conversion improves quality")
    
    return results

if __name__ == "__main__":
    test_png_vs_jpeg()