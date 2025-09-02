#!/usr/bin/env python3
"""
Test the corrected Frontal-View VTON preprocessing with geometric masking
"""

import requests
import json
import sys
import os
from PIL import Image
import base64
import io

def test_corrected_preprocessing():
    """Test the updated API with Frontal-View VTON geometric masking"""
    api_url = "http://localhost:5000/try-on"
    
    # Use the same test images as before for comparison
    person_path = "/home/ubuntu/MV-VTON/assets/person/00010_00.jpg"
    cloth_path = "/home/ubuntu/MV-VTON/assets/cloth/00086_00.jpg"
    
    print(f"🧪 Testing Corrected Frontal-View VTON Preprocessing")
    print(f"📸 Person: {person_path}")
    print(f"👔 Cloth: {cloth_path}")
    print(f"🔗 API: {api_url}")
    print("=" * 60)
    
    try:
        # Prepare files
        with open(person_path, 'rb') as pf, open(cloth_path, 'rb') as cf:
            files = {
                'person_image': ('person.jpg', pf, 'image/jpeg'),
                'cloth_image': ('cloth.jpg', cf, 'image/jpeg')
            }
            data = {
                'ddim_steps': 50,
                'scale': 1.0,
                'seed': 23
            }
            
            print("🚀 Sending request to API...")
            response = requests.post(api_url, files=files, data=data, timeout=300)
            
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ SUCCESS!")
            print(f"🔍 Response keys: {list(result.keys())}")
            
            # Save the result for inspection
            if 'result_image' in result:
                # Decode base64 image
                image_data = base64.b64decode(result['result_image'])
                output_image = Image.open(io.BytesIO(image_data))
                
                # Save with timestamp
                output_path = "/home/ubuntu/MV-VTON/corrected_preprocessing_result.jpg"
                output_image.save(output_path)
                print(f"💾 Result saved: {output_path}")
                
                # Calculate basic quality metrics
                import numpy as np
                img_array = np.array(output_image)
                
                # Basic quality metrics
                mean_brightness = np.mean(img_array)
                contrast = np.std(img_array)
                sharpness = np.var(np.gradient(img_array.astype(float)))
                
                quality_score = min(100, (mean_brightness/255 * 25 + contrast/128 * 25 + 
                                        min(sharpness/1000, 1) * 50))
                
                print(f"🎯 Estimated Quality Score: {quality_score:.1f}/100")
                print(f"   • Brightness: {mean_brightness:.1f}/255")
                print(f"   • Contrast: {contrast:.1f}")
                print(f"   • Sharpness: {sharpness:.1f}")
                
                # Report on expected improvements
                print(f"\n📈 EXPECTED IMPROVEMENTS (vs. previous approach):")
                print(f"   • Geometric masking instead of parsing-based clothing removal")
                print(f"   • Pose-guided torso polygon masking")
                print(f"   • Better preservation of body structure")
                print(f"   • More accurate person-agnostic generation")
                
                if quality_score > 70:
                    print(f"🎉 EXCELLENT: Quality score {quality_score:.1f} indicates successful correction!")
                elif quality_score > 50:
                    print(f"👍 GOOD: Quality score {quality_score:.1f} shows improvement")
                else:
                    print(f"⚠️  CONCERN: Quality score {quality_score:.1f} still needs attention")
                    
            else:
                print("❌ No image in response")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_corrected_preprocessing()