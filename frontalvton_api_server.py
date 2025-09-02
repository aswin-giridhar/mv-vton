#!/usr/bin/env python3
"""
Frontal-View VTON API Server
FastAPI server for frontal virtual try-on using the Frontal-View VTON approach.
Focused on high-quality frontal clothing transfer.
"""

import os
import sys
import torch
import base64
import traceback
from io import BytesIO
from typing import Optional
import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision.transforms import Resize
from omegaconf import OmegaConf

# Add Frontal-View VTON to path
frontal_vton_path = "/home/ubuntu/MV-VTON/Frontal-View VTON"
sys.path.insert(0, frontal_vton_path)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# Constants
DEFAULT_DDIM_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_IMAGE_SIZE = (512, 384)  # Height x Width
DEFAULT_HEIGHT, DEFAULT_WIDTH = DEFAULT_IMAGE_SIZE

app = FastAPI(
    title="Frontal-View VTON API Server",
    description="High-quality frontal virtual try-on API using Frontal-View VTON approach",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variables
model = None
sampler = None
device = None

def get_transforms():
    """Get image transformation functions for Frontal-View VTON"""
    from torchvision import transforms
    
    # Transform for input images  
    transform = transforms.Compose([
        transforms.Resize((DEFAULT_HEIGHT, DEFAULT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] range
    ])
    
    # Transform for masks
    mask_transform = transforms.Compose([
        transforms.Resize((DEFAULT_HEIGHT, DEFAULT_WIDTH)),
        transforms.ToTensor()
    ])
    
    return transform, mask_transform

def load_frontal_vton_model():
    """Load the Frontal-View VTON model"""
    global model, sampler, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config from Frontal-View VTON
    config_path = os.path.join(frontal_vton_path, "configs", "viton512.yaml")
    config = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    
    # Load model checkpoint from Frontal-View VTON
    checkpoint_path = os.path.join(frontal_vton_path, "checkpoint", "vitonhd.ckpt")
    if not os.path.exists(checkpoint_path):
        # Fallback to main checkpoint if vitonhd.ckpt doesn't exist
        checkpoint_path = "/home/ubuntu/MV-VTON/checkpoint/mvg.ckpt" 
    
    print(f"Loading model from {checkpoint_path}")
    
    # Instantiate model
    model = instantiate_from_config(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "") if key.startswith("model.") else key
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    # Create DDIM sampler
    sampler = DDIMSampler(model)
    
    print("✅ Frontal-View VTON model loaded successfully!")
    print("✅ DDIM sampler created!")
    return True

def create_simple_agnostic_image(person_image: Image.Image, target_size=DEFAULT_IMAGE_SIZE):
    """Create a simple person-agnostic image for frontal VTON"""
    H, W = target_size
    
    # Convert to numpy for processing
    person_array = np.array(person_image.resize((W, H)))
    
    # Create a simple rectangular mask for the upper body clothing area
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Define clothing region (upper torso area)
    start_y = H // 4  # Start from 1/4 down
    end_y = 3 * H // 4  # End at 3/4 down
    start_x = W // 6  # Start from 1/6 from left
    end_x = 5 * W // 6  # End at 5/6 from left
    
    mask[start_y:end_y, start_x:end_x] = 255
    
    # Apply mask to create agnostic image
    agnostic = person_array.copy()
    mask_3d = np.stack([mask] * 3, axis=2) / 255.0
    
    # Blend with neutral color in clothing region
    neutral_color = np.array([128, 128, 128])  # Gray
    agnostic = agnostic * (1 - mask_3d) + neutral_color * mask_3d
    
    agnostic_image = Image.fromarray(agnostic.astype(np.uint8))
    inpaint_mask = Image.fromarray(mask)
    
    return agnostic_image, inpaint_mask

def preprocess_images(person_image: Image.Image, cloth_image: Image.Image):
    """Preprocess images for Frontal-View VTON"""
    transform, mask_transform = get_transforms()
    
    # Process person image
    person_tensor = transform(person_image)
    
    # Process clothing image  
    cloth_tensor = transform(cloth_image)
    
    # Create agnostic image and mask
    agnostic_image, inpaint_mask = create_simple_agnostic_image(person_image)
    
    # Convert to tensors
    agnostic_tensor = transform(agnostic_image)
    mask_tensor = mask_transform(inpaint_mask)
    
    return {
        'person': person_tensor,
        'cloth': cloth_tensor,
        'agnostic': agnostic_tensor,
        'inpaint_mask': mask_tensor,
        'raw_images': {
            'person': person_image,
            'cloth': cloth_image,
            'agnostic': agnostic_image,
            'inpaint_mask': inpaint_mask
        }
    }

def frontal_vton_inference(person_image: Image.Image, cloth_image: Image.Image, 
                          ddim_steps: int = DEFAULT_DDIM_STEPS, 
                          guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
                          target_size: tuple = DEFAULT_IMAGE_SIZE):
    """Run Frontal-View VTON inference"""
    global model, sampler, device
    
    try:
        H, W = target_size
        print(f"🚀 Starting Frontal-View VTON inference with steps={ddim_steps}, scale={guidance_scale}, size={W}x{H}")
        
        # Preprocess images
        print("🔄 Preprocessing images...")
        processed = preprocess_images(person_image, cloth_image)
        
        # Move tensors to device and add batch dimension
        person_tensor = processed['person'].unsqueeze(0).to(device)
        cloth_tensor = processed['cloth'].unsqueeze(0).to(device)  
        agnostic_tensor = processed['agnostic'].unsqueeze(0).to(device)
        mask_tensor = processed['inpaint_mask'].unsqueeze(0).to(device)
        
        print(f"  📐 Tensor shapes:")
        print(f"     person: {person_tensor.shape}")
        print(f"     cloth: {cloth_tensor.shape}")
        print(f"     agnostic: {agnostic_tensor.shape}")
        print(f"     mask: {mask_tensor.shape}")
        
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    # Prepare conditioning - use cloth image for conditioning
                    cond = model.get_learned_conditioning(cloth_tensor)
                    
                    # Prepare unconditional conditioning
                    if guidance_scale != 1.0:
                        uncond = model.learnable_vector.repeat(cloth_tensor.size(0), 1, 1)
                    else:
                        uncond = None
                    
                    # Prepare model inputs for Frontal-View VTON
                    # The model expects 9-channel input: [agnostic(3) + person(3) + mask(1) + cloth(3)]
                    print("🔧 Preparing 9-channel input for Frontal-View VTON...")
                    
                    # Ensure mask has correct dimensions for concatenation
                    if mask_tensor.dim() == 3:  # [batch, height, width]
                        mask_tensor = mask_tensor.unsqueeze(1)  # Add channel dim -> [batch, 1, height, width]
                    
                    # Create 9-channel input
                    model_input = torch.cat([
                        agnostic_tensor,  # 3 channels
                        person_tensor,    # 3 channels  
                        mask_tensor,      # 1 channel
                        cloth_tensor      # 3 channels
                    ], dim=1)  # Total: 9 channels
                    
                    print(f"  🔧 Model input shape: {model_input.shape}")
                    
                    # Encode to latent space
                    print("📦 Encoding to latent space...")
                    latent_input = model.encode_first_stage(model_input)
                    latent_input = model.get_first_stage_encoding(latent_input).detach()
                    
                    print(f"  📦 Latent shape: {latent_input.shape}")
                    
                    # DDIM sampling
                    print("🎨 Running DDIM sampling...")
                    shape = latent_input.shape[1:]  # Remove batch dimension
                    
                    samples_ddim, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=cond,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=guidance_scale,
                        unconditional_conditioning=uncond,
                        eta=0.0,
                        x_T=None  # Start from noise
                    )
                    
                    # Decode from latent space
                    print("📤 Decoding from latent space...")
                    result_images = model.decode_first_stage(samples_ddim)
                    result_images = torch.clamp((result_images + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    # Convert to PIL Image
                    result_array = result_images[0].cpu().permute(1, 2, 0).numpy()
                    result_array = (result_array * 255).astype(np.uint8)
                    result_img = Image.fromarray(result_array)
                    
                    print(f"  📤 Final result: size={result_img.size}, mode={result_img.mode}")
                    
                    print("✅ Frontal-View VTON inference completed successfully!")
                    return result_img
                    
    except Exception as e:
        print(f"❌ Error in Frontal-View VTON inference: {e}")
        traceback.print_exc()
        return None

# API Endpoints

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_type": "Frontal-View VTON",
        "model_loaded": model is not None,
        "sampler_ready": sampler is not None,
        "device": str(device) if device else "unknown",
        "cuda_available": torch.cuda.is_available(),
        "default_image_size": DEFAULT_IMAGE_SIZE,
        "api_version": "1.0.0"
    }

@app.get('/model/info')
async def model_info():
    """Get model information"""
    return {
        "model_type": "Frontal-View VTON",
        "architecture": "LatentTryOnDiffusion", 
        "input_channels": 9,  # Key difference from MV-VTON
        "image_size": DEFAULT_IMAGE_SIZE,
        "focus": "frontal_upper_body_clothing",
        "approach": "single_view_optimized"
    }

@app.post('/try-on')
async def virtual_try_on(
    person_image: UploadFile = File(..., description="Person image file"),
    cloth_image: UploadFile = File(..., description="Clothing image file"),
    ddim_steps: int = Form(DEFAULT_DDIM_STEPS, description="Number of DDIM sampling steps"),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE, description="Guidance scale for sampling"),
    image_width: int = Form(DEFAULT_WIDTH, description="Output image width"),
    image_height: int = Form(DEFAULT_HEIGHT, description="Output image height")
):
    """Virtual try-on endpoint returning base64 encoded image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read images
        person_pil = Image.open(BytesIO(await person_image.read())).convert('RGB')
        cloth_pil = Image.open(BytesIO(await cloth_image.read())).convert('RGB')
        
        print(f"📸 Processing images: person={person_pil.size}, cloth={cloth_pil.size}")
        
        # Run inference
        result_img = frontal_vton_inference(
            person_pil, cloth_pil,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            target_size=(image_height, image_width)
        )
        
        if result_img is None:
            raise HTTPException(status_code=500, detail="Virtual try-on failed")
        
        # Convert to base64
        buffer = BytesIO()
        result_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "image": img_base64,
            "format": "PNG",
            "size": result_img.size,
            "model": "Frontal-View VTON"
        }
        
    except Exception as e:
        print(f"❌ Virtual try-on error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/try-on-file')
async def virtual_try_on_file(
    person_image: UploadFile = File(..., description="Person image file"),
    cloth_image: UploadFile = File(..., description="Clothing image file"),
    ddim_steps: int = Form(DEFAULT_DDIM_STEPS, description="Number of DDIM sampling steps"),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE, description="Guidance scale for sampling"),
    image_width: int = Form(DEFAULT_WIDTH, description="Output image width"), 
    image_height: int = Form(DEFAULT_HEIGHT, description="Output image height")
):
    """Virtual try-on endpoint returning image file"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read images
        person_pil = Image.open(BytesIO(await person_image.read())).convert('RGB')
        cloth_pil = Image.open(BytesIO(await cloth_image.read())).convert('RGB')
        
        # Run inference
        result_img = frontal_vton_inference(
            person_pil, cloth_pil,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            target_size=(image_height, image_width)
        )
        
        if result_img is None:
            raise HTTPException(status_code=500, detail="Virtual try-on failed")
        
        # Return as image file
        buffer = BytesIO()
        result_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=virtual_tryon_result.png"}
        )
        
    except Exception as e:
        print(f"❌ Virtual try-on error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    print("🚀 Starting Frontal-View VTON API Server...")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🎯 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"💾 GPU count: {torch.cuda.device_count()}")
        print(f"🔧 GPU name: {torch.cuda.get_device_name()}")
    
    print("\n" + "=" * 60)
    print("🤖 INITIALIZING FRONTAL-VIEW VTON MODEL...")
    print("=" * 60)
    
    success = load_frontal_vton_model()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ FRONTAL-VIEW VTON MODEL READY!")
    else:
        print("❌ FRONTAL-VIEW VTON MODEL FAILED TO LOAD!")
    print("🌐 Starting FastAPI server on 0.0.0.0:5000")
    print("📖 Available endpoints:")
    print("   GET  /health - Health check and system status")
    print("   GET  /model/info - Model information")
    print("   POST /try-on - Virtual try-on (returns base64 image)")
    print("   POST /try-on-file - Virtual try-on (returns image file)")
    print("   GET  /docs - Interactive API documentation")
    print("   GET  /redoc - Alternative API documentation")
    print("=" * 60)

if __name__ == "__main__":
    import uvicorn
    # Set global seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    uvicorn.run(app, host="0.0.0.0", port=5000)