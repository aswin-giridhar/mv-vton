# MV-VTON API Server Usage Guide

This guide provides comprehensive instructions for using the integrated MV-VTON API server.

## Quick Start

### 1. Start the Server
```bash
# Activate conda environment and start server
./start_api_server.sh

# Or manually:
conda activate mv-vton
python mvvton_api_server.py
```

### 2. Check Server Health
```bash
curl http://localhost:5000/health
```

## API Endpoints

### 1. Health Check
**GET** `/health`

Returns server status and configuration information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-31T22:54:00.123456",
  "model_loaded": true,
  "sampler_ready": true,
  "device": "cuda:0",
  "cuda_available": true,
  "gpu_count": 1,
  "pytorch_version": "2.0.0",
  "api_version": "1.0.0"
}
```

### 2. Model Information
**GET** `/model/info`

Returns detailed model configuration.

**Response:**
```json
{
  "model_type": "LatentTryOnDiffusion",
  "config_path": "configs/viton512.yaml",
  "image_size": [512, 384],
  "default_ddim_steps": 30,
  "conditioning_type": "crossattn",
  "model_parameters": 859123456
}
```

### 3. Virtual Try-On (Base64)
**POST** `/try-on`

Main virtual try-on endpoint that returns the result as base64-encoded image.

**Required Form Data:**
- `person_image` (file): Person image (JPEG/PNG)
- `cloth_image` (file): Clothing item image (JPEG/PNG)

**Optional Form Data:**
- `cloth_back_image` (file): Back view of clothing (optional)
- `ddim_steps` (int): Number of denoising steps (1-200, default: 30)
- `scale` (float): Guidance scale (0.0-10.0, default: 1.0)
- `height` (int): Output height (256-1024, default: 512)
- `width` (int): Output width (256-1024, default: 384)

**Example using curl:**
```bash
curl -X POST \
  -F "person_image=@person.jpg" \
  -F "cloth_image=@cloth.jpg" \
  -F "ddim_steps=50" \
  -F "scale=1.5" \
  http://localhost:5000/try-on
```

**Response:**
```json
{
  "success": true,
  "result_image": "iVBORw0KGgoAAAANSUhEUgAA...", 
  "message": "MV-VTON virtual try-on completed successfully",
  "processing_time": "12.34s",
  "parameters": {
    "ddim_steps": 50,
    "scale": 1.5,
    "output_size": [512, 384],
    "person_input_size": [512, 768],
    "cloth_input_size": [512, 512]
  }
}
```

### 4. Virtual Try-On (File)
**POST** `/try-on-file`

Returns the result as a downloadable PNG file.

**Same parameters as `/try-on`**

**Example:**
```bash
curl -X POST \
  -F "person_image=@person.jpg" \
  -F "cloth_image=@cloth.jpg" \
  -o result.png \
  http://localhost:5000/try-on-file
```

## Python Client Example

```python
import requests
import base64
from PIL import Image
import io

def try_on_clothes(person_image_path, cloth_image_path, 
                   server_url="http://localhost:5000"):
    """
    Perform virtual try-on using the MV-VTON API
    """
    
    # Prepare files
    files = {
        'person_image': open(person_image_path, 'rb'),
        'cloth_image': open(cloth_image_path, 'rb')
    }
    
    # Optional parameters
    data = {
        'ddim_steps': 50,
        'scale': 1.2,
        'height': 512,
        'width': 384
    }
    
    try:
        # Make API call
        response = requests.post(
            f"{server_url}/try-on", 
            files=files, 
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                # Decode base64 image
                img_data = base64.b64decode(result['result_image'])
                img = Image.open(io.BytesIO(img_data))
                
                # Save result
                output_path = "try_on_result.png"
                img.save(output_path)
                print(f"✅ Result saved to: {output_path}")
                print(f"⏱️  Processing time: {result['processing_time']}")
                
                return img
            else:
                print(f"❌ API Error: {result.get('message', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            
    finally:
        # Close files
        for file in files.values():
            file.close()
    
    return None

# Usage example
if __name__ == "__main__":
    # Check server health
    health = requests.get("http://localhost:5000/health")
    print("Server Health:", health.json()['status'])
    
    # Perform try-on
    result_img = try_on_clothes("person.jpg", "cloth.jpg")
```

## JavaScript/Node.js Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function tryOnClothes(personImagePath, clothImagePath) {
    const form = new FormData();
    
    // Add files
    form.append('person_image', fs.createReadStream(personImagePath));
    form.append('cloth_image', fs.createReadStream(clothImagePath));
    
    // Add parameters
    form.append('ddim_steps', '50');
    form.append('scale', '1.2');
    
    try {
        const response = await axios.post('http://localhost:5000/try-on', form, {
            headers: form.getHeaders(),
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        
        if (response.data.success) {
            console.log('✅ Try-on completed!');
            console.log(`⏱️  Processing time: ${response.data.processing_time}`);
            
            // Save base64 image
            const base64Data = response.data.result_image;
            const buffer = Buffer.from(base64Data, 'base64');
            fs.writeFileSync('result.png', buffer);
            
            return true;
        }
    } catch (error) {
        console.error('❌ Error:', error.response?.data || error.message);
        return false;
    }
}

// Usage
tryOnClothes('person.jpg', 'cloth.jpg');
```

## Error Handling

The API returns structured error responses:

```json
{
  "error": "Error description",
  "timestamp": "2025-08-31T22:54:00.123456"
}
```

Common error codes:
- `400`: Bad request (missing files, invalid parameters)
- `500`: Server error (model loading failed, inference error)

## Performance Tips

1. **Image Sizes**: Larger images take longer to process. Consider resizing input images to reasonable sizes.

2. **DDIM Steps**: More steps = better quality but slower processing
   - Fast: 20-30 steps
   - Balanced: 30-50 steps  
   - High Quality: 50-100 steps

3. **Batch Processing**: For multiple images, reuse the same server instance rather than restarting.

4. **GPU Memory**: Monitor GPU memory usage. Restart the server if memory issues occur.

## Troubleshooting

### Server Won't Start
1. Check conda environment: `conda activate mv-vton`
2. Verify model files exist: `ls -la checkpoint/mvg.ckpt`
3. Run setup test: `python test_api_setup.py`

### Inference Fails  
1. Check image formats (RGB JPEG/PNG recommended)
2. Verify image sizes aren't too large (>2048px)
3. Monitor server logs for detailed error messages

### Poor Results
1. This implementation uses simplified preprocessing
2. For production quality, integrate:
   - Human parsing for accurate segmentation
   - Pose detection for skeleton conditioning
   - Cloth warping preprocessing
   - Trained ControlNet models

## Technical Notes

This implementation provides:
✅ Proper model loading from the original MV-VTON codebase
✅ Complete DDIM/PLMS sampling pipeline  
✅ ControlNet integration for pose conditioning
✅ Proper image preprocessing and postprocessing
✅ Production-ready API with error handling

Simplified components (for demonstration):
⚠️ Basic human parsing (should use trained models)
⚠️ Simple pose detection (should use OpenPose/MediaPipe)
⚠️ Basic cloth warping (should use trained warping networks)

For production deployment, integrate these components with their respective trained models.