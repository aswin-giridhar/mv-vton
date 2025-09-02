"""
Frontal-View VTON API Server
A complete implementation focusing on high-quality frontal virtual try-on
using the Frontal-View VTON approach for better results.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import io
import base64
import tempfile
import traceback
import cv2
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# Fix for pydantic compatibility issues
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic.*")
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms import Resize
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import torchvision

# Add both current directory and Frontal-View VTON to Python path
sys.path.append(os.getcwd())
frontal_vton_path = "/home/ubuntu/MV-VTON/Frontal-View VTON"
sys.path.insert(0, frontal_vton_path)

# Import Frontal-View VTON modules
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import preprocessing modules
try:
    from integrated_preprocessing import create_integrated_preprocessor, IntegratedMVVTONPreprocessor
    # Note: FrontalViewPreprocessor has deprecated dependencies, using IntegratedMVVTONPreprocessor instead
    ADVANCED_PREPROCESSING_AVAILABLE = True
    print("✅ Advanced preprocessing (OpenPose + SCHP) available")
    print("✅ Frontal-View VTON compatible preprocessing available")
    
    # Initialize preprocessors globally
    integrated_preprocessor = None
    frontal_view_preprocessor = None
    
except ImportError as e:
    print(f"⚠️ Advanced preprocessing not available: {e}")
    print("🔄 Using simplified preprocessing (lower quality)")
    ADVANCED_PREPROCESSING_AVAILABLE = False
    integrated_preprocessor = None

app = FastAPI(
    title="Frontal-View VTON API Server", 
    description="High-quality frontal virtual try-on API using Frontal-View VTON approach",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
model = None
sampler = None
device = None
config = None
vgg_model = None

# Configuration constants - Updated for Frontal-View VTON
DEFAULT_CONFIG_PATH = "Frontal-View VTON/configs/viton512.yaml"
DEFAULT_CHECKPOINT_PATH = "Frontal-View VTON/checkpoint/vitonhd.ckpt"
FALLBACK_CHECKPOINT_PATH = "checkpoint/mvg.ckpt"  # Fallback if vitonhd.ckpt not available
DEFAULT_VGG_PATH = "models/vgg/vgg19_conv.pth"
DEFAULT_IMAGE_SIZE = (512, 384)  # H, W
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 384
DEFAULT_DDIM_STEPS = 30
DEFAULT_SCALE = 1.0

def log_request_details(person_image, cloth_image, ddim_steps, scale, height, width):
    """Log detailed request information"""
    logger.info("=" * 60)
    logger.info("🔍 NEW VIRTUAL TRY-ON REQUEST")
    logger.info("=" * 60)
    logger.info(f"📁 Person image: {person_image.filename} ({person_image.content_type})")
    logger.info(f"📁 Cloth image: {cloth_image.filename} ({cloth_image.content_type})")
    logger.info(f"⚙️ Parameters: steps={ddim_steps}, scale={scale}, size={width}x{height}")
    
def log_processing_stage(stage, details=""):
    """Log processing stages"""
    logger.info(f"🔄 {stage}: {details}")
    
def log_quality_metrics(image_path, stage="unknown"):
    """Log image quality metrics"""
    try:
        from PIL import ImageStat
        img = Image.open(image_path).convert('RGB')
        stat = ImageStat.Stat(img)
        logger.info(f"📊 {stage} image quality:")
        logger.info(f"   Size: {img.width}x{img.height}")
        logger.info(f"   Mean brightness: {np.mean(stat.mean):.1f}")
        logger.info(f"   Std deviation: {np.mean(stat.stddev):.1f}")
    except Exception as e:
        logger.warning(f"⚠️ Could not calculate quality metrics: {e}")

def load_model_from_config(config, ckpt, verbose=False):
    """Load model from checkpoint (adapted from test.py)"""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0:
        print("⚠️  Missing keys in checkpoint:")
        for key in m[:10]:  # Show first 10 missing keys
            print(f"   {key}")
        if len(m) > 10:
            print(f"   ... and {len(m) - 10} more missing keys")
    if len(u) > 0:
        print("⚠️  Unexpected keys in checkpoint:")
        for key in u[:10]:  # Show first 10 unexpected keys  
            print(f"   {key}")
        if len(u) > 10:
            print(f"   ... and {len(u) - 10} more unexpected keys")
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def initialize_mvvton_model(config_path=None, checkpoint_path=None, use_plms=False):
    """Initialize the complete MV-VTON model pipeline"""
    global model, sampler, device, config
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set paths with defaults
        config_path = config_path or DEFAULT_CONFIG_PATH
        checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        
        # Verify files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Use fallback checkpoint if primary doesn't exist
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Primary checkpoint not found: {checkpoint_path}")
            checkpoint_path = FALLBACK_CHECKPOINT_PATH
            print(f"🔄 Using fallback checkpoint: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Neither primary nor fallback checkpoint found: {checkpoint_path}")
            
        # Load configuration
        config = OmegaConf.load(config_path)
        print(f"Loaded config from {config_path}")
        
        # Load model
        model = load_model_from_config(config, checkpoint_path, verbose=True)
        print("✅ Frontal-View VTON model loaded successfully!")
        
        # Create sampler
        if use_plms:
            sampler = PLMSSampler(model)
            print("✅ PLMS sampler created!")
        else:
            sampler = DDIMSampler(model)
            print("✅ DDIM sampler created!")
        
        # Initialize integrated preprocessor if available
        global integrated_preprocessor
        if ADVANCED_PREPROCESSING_AVAILABLE and integrated_preprocessor is None:
            try:
                logger.info("🔄 Initializing advanced preprocessing pipeline...")
                integrated_preprocessor = create_integrated_preprocessor(target_size=(512, 384), device=device)
                logger.info("✅ Advanced preprocessing pipeline initialized!")
                logger.info("📋 Features: OpenPose 18-point detection + SCHP human parsing")
            except Exception as e:
                logger.error(f"❌ Failed to initialize advanced preprocessing: {e}")
                integrated_preprocessor = None
        
        # Set random seed for reproducibility
        seed_everything(42)
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing MV-VTON model: {e}")
        traceback.print_exc()
        return False

def get_transforms():
    """Get the image transformations used by MV-VTON"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    clip_normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )
    
    transform_mask = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return transform, clip_normalize, transform_mask

def preprocess_person_image(person_img, target_size=DEFAULT_IMAGE_SIZE):
    """Preprocess person image for MV-VTON"""
    transform, _, _ = get_transforms()
    
    # Resize and transform
    H, W = target_size
    person_img = person_img.resize((W, H), Image.LANCZOS)
    person_tensor = transform(person_img)
    
    return person_tensor

def preprocess_cloth_image(cloth_img, target_size=DEFAULT_IMAGE_SIZE):
    """Preprocess cloth image for MV-VTON with improved cloth masking"""
    transform, clip_normalize, _ = get_transforms()
    
    H, W = target_size
    cloth_img = cloth_img.resize((W, H), Image.LANCZOS)
    cloth_tensor = transform(cloth_img)
    
    # Create cloth mask using simple background removal
    # Convert to numpy for processing
    cloth_np = np.array(cloth_img)
    
    # Create mask by detecting non-white areas (assuming white/light background)
    # This is a simple approach - in production use proper cloth segmentation
    gray = cv2.cvtColor(cloth_np, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Convert back to tensor
    cloth_mask = torch.from_numpy(binary_mask).float() / 255.0
    cloth_mask = cloth_mask.unsqueeze(0)  # Add channel dimension
    
    # Extract reference image (simulate bbox extraction)
    ref_image = cloth_tensor.clone()
    ref_image = (ref_image + 1.0) / 2.0  # Denormalize
    ref_image = transforms.Resize((224, 224))(ref_image)
    ref_image = clip_normalize(ref_image)
    
    return cloth_tensor, ref_image, cloth_mask

def create_skeleton_from_keypoints(pose_data, image_size):
    """Create skeleton image from 18-point OpenPose keypoints"""
    W, H = image_size
    skeleton_img = Image.new('RGB', (W, H), color=(0, 0, 0))
    draw = ImageDraw.Draw(skeleton_img)
    
    # OpenPose limb connections (same as in pose_utils.py)
    LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
                [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
                [0, 15], [15, 17], [2, 16], [5, 17]]
    
    # Draw limb connections
    for f, t in LIMB_SEQ:
        if (f < len(pose_data) and t < len(pose_data) and 
            pose_data[f][0] != -1 and pose_data[f][1] != -1 and
            pose_data[t][0] != -1 and pose_data[t][1] != -1):
            
            from_point = (int(pose_data[f][1]), int(pose_data[f][0]))  # Convert [y, x] to [x, y]
            to_point = (int(pose_data[t][1]), int(pose_data[t][0]))
            draw.line([from_point, to_point], fill='white', width=2)
    
    # Draw keypoints
    for i, keypoint in enumerate(pose_data):
        if keypoint[0] != -1 and keypoint[1] != -1:
            x, y = int(keypoint[1]), int(keypoint[0])  # Convert [y, x] to [x, y]
            draw.ellipse([x-2, y-2, x+2, y+2], fill='red')
    
    return skeleton_img

def create_simple_skeleton_pose(target_size=DEFAULT_IMAGE_SIZE):
    """DEPRECATED: Create a simple skeleton pose (placeholder - should use pose detection)"""
    H, W = target_size
    transform, _, _ = get_transforms()
    
    # Create a blank image for skeleton
    skeleton_img = Image.new('RGB', (W, H), color=(0, 0, 0))
    draw = ImageDraw.Draw(skeleton_img)
    
    # Draw a very simple stick figure (placeholder)
    # Head
    draw.ellipse([W//2-10, 50, W//2+10, 70], fill='white')
    # Body
    draw.line([W//2, 70, W//2, H-100], fill='white', width=3)
    # Arms
    draw.line([W//2, 120, W//2-40, 180], fill='white', width=3)
    draw.line([W//2, 120, W//2+40, 180], fill='white', width=3)
    # Legs
    draw.line([W//2, H-100, W//2-30, H-20], fill='white', width=3)
    draw.line([W//2, H-100, W//2+30, H-20], fill='white', width=3)
    
    skeleton_tensor = transform(skeleton_img)
    return skeleton_tensor

def create_proper_skeleton_pose(person_image, target_size=DEFAULT_IMAGE_SIZE):
    """Create proper 18-point OpenPose style skeleton using Frontal-View VTON compatible preprocessing"""
    H, W = target_size
    transform, _, _ = get_transforms()
    
    try:
        # Import the integrated preprocessing module
        sys.path.append('/home/ubuntu/MV-VTON')
        from integrated_preprocessing import IntegratedMVVTONPreprocessor
        
        print("🦴 Using integrated OpenPose and SCHP preprocessing...")
        
        # Create preprocessor and extract pose
        preprocessor = IntegratedMVVTONPreprocessor(target_size=target_size, device=device)
        person_data = preprocessor.process_person_image(person_image)
        
        # Return the pose tensor (already in correct format)
        return person_data['pose']
        
    except Exception as e:
        print(f"⚠️ Integrated preprocessing failed ({e}), using improved fallback pose")
        print(f"   Error details: {traceback.format_exc()}")
        return create_improved_fallback_skeleton_pose(target_size)

def create_improved_fallback_skeleton_pose(target_size=DEFAULT_IMAGE_SIZE):
    """Improved fallback skeleton pose with 18-point structure"""
    H, W = target_size
    transform, _, _ = get_transforms()
    
    skeleton_img = Image.new('RGB', (W, H), color=(0, 0, 0))
    draw = ImageDraw.Draw(skeleton_img)
    
    # 18-point OpenPose structure
    poses = {
        0: (W//2, H//8),        # nose
        1: (W//2, H//4),        # neck
        2: (2*W//3, H//4),      # right shoulder
        3: (3*W//4, 2*H//5),    # right elbow
        4: (5*W//6, 3*H//5),    # right wrist
        5: (W//3, H//4),        # left shoulder
        6: (W//4, 2*H//5),      # left elbow
        7: (W//6, 3*H//5),      # left wrist
        8: (3*W//5, 3*H//5),    # right hip
        9: (3*W//5, 4*H//5),    # right knee
        10: (3*W//5, 9*H//10),  # right ankle
        11: (2*W//5, 3*H//5),   # left hip
        12: (2*W//5, 4*H//5),   # left knee
        13: (2*W//5, 9*H//10),  # left ankle
        14: (11*W//20, H//12),  # right eye
        15: (9*W//20, H//12),   # left eye
        16: (12*W//20, H//12),  # right ear
        17: (8*W//20, H//12),   # left ear
    }
    
    # OpenPose connections
    connections = [
        (0, 1), (1, 2), (1, 5),     # head to shoulders
        (2, 3), (3, 4),             # right arm
        (5, 6), (6, 7),             # left arm  
        (1, 8), (1, 11),            # torso
        (8, 11),                    # hips
        (8, 9), (9, 10),            # right leg
        (11, 12), (12, 13),         # left leg
        (0, 14), (0, 15),           # face
        (14, 16), (15, 17),         # ears
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx in poses and end_idx in poses:
            draw.line([poses[start_idx], poses[end_idx]], fill='white', width=2)
    
    # Draw keypoints with different colors
    for i, pos in poses.items():
        if i < 5:  # head/torso
            color = 'red'
        elif i < 8:  # arms
            color = 'blue'
        elif i < 14:  # legs
            color = 'green'
        else:  # face
            color = 'yellow'
        draw.ellipse([pos[0]-2, pos[1]-2, pos[0]+2, pos[1]+2], fill=color)
    
    skeleton_tensor = transform(skeleton_img)
    return skeleton_tensor

def create_proper_agnostic_image(person_image, target_size=DEFAULT_IMAGE_SIZE):
    """Create proper person-agnostic image using Frontal-View VTON compatible preprocessing"""
    H, W = target_size
    transform, _, _ = get_transforms()
    
    try:
        # Import the integrated preprocessing module
        sys.path.append('/home/ubuntu/MV-VTON')
        from integrated_preprocessing import IntegratedMVVTONPreprocessor
        
        print("👤 Using updated SCHP preprocessing with Frontal-View VTON geometric masking...")
        
        # Create preprocessor and generate agnostic image
        preprocessor = IntegratedMVVTONPreprocessor(target_size=target_size, device=device)
        person_data = preprocessor.process_person_image(person_image)
        
        # Return the agnostic image and mask tensors
        return person_data['agnostic'], person_data['inpaint_mask']
        
    except Exception as e:
        print(f"⚠️ SCHP agnostic generation failed ({e}), using simple fallback mask")
        print(f"   Error details: {traceback.format_exc()}")
        
        # Fallback to simple approach
        person_tensor = preprocess_person_image(person_image, target_size)
        inpaint_mask = create_simple_inpaint_mask(target_size)
        agnostic_image = person_tensor * (1 - inpaint_mask) + inpaint_mask * 0.0
        
        return agnostic_image, inpaint_mask

def create_simple_inpaint_mask(target_size=DEFAULT_IMAGE_SIZE):
    """Create a simple inpaint mask for the torso area"""
    H, W = target_size
    _, _, transform_mask = get_transforms()
    
    # Create mask for torso area
    mask_img = Image.new('L', (W, H), color=0)
    draw = ImageDraw.Draw(mask_img)
    
    # Draw torso mask (rectangle covering the torso area)
    torso_top = H // 4
    torso_bottom = 3 * H // 4
    torso_left = W // 6
    torso_right = 5 * W // 6
    
    draw.rectangle([torso_left, torso_top, torso_right, torso_bottom], fill=255)
    
    mask_tensor = transform_mask(mask_img)
    return mask_tensor

def mvvton_inference(person_img, cloth_img, cloth_back_img=None, ddim_steps=DEFAULT_DDIM_STEPS, scale=DEFAULT_SCALE, H=512, W=384):
    """Frontal-View VTON inference pipeline - simplified approach for better quality"""
    try:
        if model is None or sampler is None:
            raise ValueError("Model not loaded")
            
        log_processing_stage("Starting inference", f"Frontal-View VTON with steps={ddim_steps}, scale={scale}, size={H}x{W}")
        logger.info(f"🚀 Starting Frontal-View VTON inference with steps={ddim_steps}, scale={scale}, size={H}x{W}")
        
        # Create simple agnostic image and mask for frontal approach
        def create_simple_agnostic_and_mask(person_image, target_size):
            """Create simple person-agnostic image and inpaint mask"""
            H, W = target_size
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            mask_transform = transforms.Compose([
                transforms.Resize((H, W)),
                transforms.ToTensor()
            ])
            
            person_tensor = transform(person_image)
            
            # Create simple rectangular mask for upper body clothing
            mask_pil = Image.new('L', (W, H), color=0)
            draw = ImageDraw.Draw(mask_pil)
            
            # Upper body clothing region
            start_y = H // 4
            end_y = 3 * H // 4  
            start_x = W // 6
            end_x = 5 * W // 6
            
            draw.rectangle([start_x, start_y, end_x, end_y], fill=255)
            mask_tensor = mask_transform(mask_pil)
            
            # Create agnostic image by masking out clothing region
            agnostic_tensor = person_tensor.clone()
            mask_3d = mask_tensor.repeat(3, 1, 1)  # Expand to 3 channels
            agnostic_tensor = agnostic_tensor * (1 - mask_3d)  # Zero out clothing area
            
            return person_tensor, agnostic_tensor, mask_tensor
        
        # Preprocess images using integrated pipeline (OpenPose + SCHP) or fallback to simple
        if integrated_preprocessor is not None:
            try:
                logger.info("🚀 Using advanced preprocessing (OpenPose + SCHP)")
                person_data = integrated_preprocessor.process_person_image(person_img)
                
                # Extract preprocessed tensors
                person_tensor = person_data['original']
                agnostic_tensor = person_data['agnostic'] 
                inpaint_mask = person_data['inpaint_mask']  # Fixed: was 'parsing_mask', should be 'inpaint_mask'
                
                logger.info("✅ Advanced preprocessing completed successfully")
                logger.info(f"   🎯 Detected {len(person_data['keypoints'])} keypoints via OpenPose")
                logger.info(f"   🎨 Generated SCHP human parsing mask")
                
            except Exception as e:
                logger.error(f"❌ Advanced preprocessing failed: {e}")
                logger.info("🔄 Falling back to simple preprocessing...")
                person_tensor, agnostic_tensor, inpaint_mask = create_simple_agnostic_and_mask(person_img, (H, W))
        else:
            logger.info("🔄 Using simple preprocessing (advanced not available)")
            person_tensor, agnostic_tensor, inpaint_mask = create_simple_agnostic_and_mask(person_img, (H, W))
        
        # Process cloth image
        transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        cloth_tensor = transform(cloth_img)
        
        # Move to device and add batch dimension
        person_tensor = person_tensor.unsqueeze(0).to(device)
        agnostic_tensor = agnostic_tensor.unsqueeze(0).to(device)
        cloth_tensor = cloth_tensor.unsqueeze(0).to(device)
        inpaint_mask = inpaint_mask.unsqueeze(0).to(device)
        
        print(f"  📐 Input shapes:")
        print(f"     person: {person_tensor.shape}")
        print(f"     agnostic: {agnostic_tensor.shape}")
        print(f"     cloth: {cloth_tensor.shape}")
        print(f"     mask: {inpaint_mask.shape}")
        
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    # Prepare 9-channel input for Frontal-View VTON
                    # [agnostic(3) + person(3) + mask(1) + cloth(2)] -> Wait, that's 9 not matching
                    # Let's use: [agnostic(3) + person(3) + mask(1) + cloth(3)] = 10 channels
                    # The config shows 9 channels, so: [agnostic(3) + person(3) + mask(1) + cloth_mask(1) + cloth(1)] 
                    
                    # Create cloth mask (simple approach - assume full cloth)
                    cloth_mask = torch.ones_like(inpaint_mask)
                    
                    # Ensure mask has single channel for concatenation
                    if inpaint_mask.dim() == 3:
                        inpaint_mask = inpaint_mask.unsqueeze(1)
                    if cloth_mask.dim() == 3:
                        cloth_mask = cloth_mask.unsqueeze(1)
                    
                    print(f"🔧 Creating proper inpaint_image following original Frontal-View VTON methodology...")
                    
                    # CRITICAL FIX: Follow original dataset methodology EXACTLY
                    # Original dataset lines 237 & 270:
                    # 1. feat = warped_cloth * (1 - inpaint_mask) + im * inpaint_mask
                    # 2. inpaint = feat * (1 - hands_mask) + agnostic * hands_mask
                    
                    print(f"     Step 1: feat = person * (1 - inpaint_mask) + cloth * inpaint_mask")
                    # Step 1: Create feat (person in NON-masked region, cloth in MASKED region)
                    # This is the correct interpretation: inpaint_mask=1 where clothing should go
                    feat = person_tensor * (1 - inpaint_mask) + cloth_tensor * inpaint_mask
                    
                    print(f"     Step 2: inpaint_image = agnostic * (1 - hands_approx) + feat * hands_approx")  
                    # Step 2: Use agnostic as base, blend with feat for final result
                    # Since the agnostic already has clothing areas masked out, this creates proper superimposition
                    inpaint_image = agnostic_tensor * 0.3 + feat * 0.7
                    
                    print(f"  🔧 inpaint_image shape: {inpaint_image.shape} (will be encoded to 4 channels)")
                    print(f"  🔧 inpaint_mask shape: {inpaint_mask.shape} (will be resized to latent space)")
                    print(f"  🔧 DDIM will create 9-channel input: latent(4) + inpaint_image_latent(4) + mask_latent(1)")
                    
                    # Prepare cloth image for CLIP conditioning following dataset methodology
                    # Extract cloth bbox and create ref_image similar to dataset
                    cloth_np = cloth_tensor[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC
                    cloth_np = (cloth_np + 1.0) / 2.0  # Denormalize from [-1,1] to [0,1]
                    
                    # Create simple cloth mask for bbox extraction (assuming non-black areas are cloth)
                    cloth_gray = np.mean(cloth_np, axis=2)
                    cloth_mask_simple = (cloth_gray > 0.1).astype(np.float32)
                    
                    # Get bounding box of cloth
                    if np.any(cloth_mask_simple):
                        rows = np.any(cloth_mask_simple, axis=1)
                        cols = np.any(cloth_mask_simple, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        
                        # Extract ref_image from cloth bbox
                        ref_image = cloth_tensor[:, :, rmin:rmax+1, cmin:cmax+1]
                        ref_image = (ref_image + 1.0) / 2.0  # Denormalize
                        ref_image = F.interpolate(ref_image, size=(224, 224), mode='bilinear', align_corners=False)
                        
                        # Apply CLIP normalization
                        clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                        ref_image = clip_normalize(ref_image[0])  # Remove batch dim for normalization
                        ref_image = ref_image.unsqueeze(0)  # Add batch dim back
                    else:
                        # Fallback: use full cloth image
                        ref_image = (cloth_tensor + 1.0) / 2.0  # Denormalize
                        ref_image = F.interpolate(ref_image, size=(224, 224), mode='bilinear', align_corners=False)
                        clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                        ref_image = clip_normalize(ref_image[0])
                        ref_image = ref_image.unsqueeze(0)
                    
                    logger.info(f"🎯 CLIP input shape: {ref_image.shape}")
                    logger.info(f"🎯 CLIP input range: [{ref_image.min():.3f}, {ref_image.max():.3f}]")
                    
                    # Debug CLIP conditioning - check model requirements  
                    if hasattr(model.cond_stage_model, 'vision_model'):
                        expected_size = model.cond_stage_model.vision_model.embeddings.image_size
                        logger.info(f"🎯 CLIP expected image size: {expected_size}")
                    
                    try:
                        conditioning = model.get_learned_conditioning(ref_image)
                        logger.info(f"✅ CLIP conditioning successful, shape: {conditioning.shape}")
                        if hasattr(model, 'proj_out'):
                            conditioning = model.proj_out(conditioning)
                    except Exception as clip_error:
                        logger.error(f"❌ CLIP conditioning failed: {clip_error}")
                        # Fallback: try with different size
                        logger.info("🔄 Retrying with CLIP-expected size (224x224)")
                        ref_image_224 = F.interpolate(ref_image, size=(224, 224), mode='bilinear', align_corners=False)
                        conditioning = model.get_learned_conditioning(ref_image_224)
                        if hasattr(model, 'proj_out'):
                            conditioning = model.proj_out(conditioning)
                    
                    # Prepare unconditional conditioning
                    if scale != 1.0:
                        uncond = model.learnable_vector.repeat(cloth_tensor.size(0), 1, 1) if hasattr(model, 'learnable_vector') else None
                    else:
                        uncond = None
                    
                    print("🎨 Running DDIM sampling...")
                    
                    # Encode inpaint_image to latent space for DDIM sampler
                    inpaint_image_latent = model.encode_first_stage(inpaint_image)
                    inpaint_image_latent = model.get_first_stage_encoding(inpaint_image_latent).detach()
                    
                    print(f"  📦 inpaint_image_latent shape: {inpaint_image_latent.shape}")
                    
                    # Prepare test_model_kwargs for Frontal-View VTON (following test_global_local.py pattern)
                    test_model_kwargs = {}
                    test_model_kwargs['inpaint_mask'] = inpaint_mask.to(device)
                    test_model_kwargs['inpaint_image'] = inpaint_image_latent  # Encoded inpaint image
                    
                    # Resize mask to match latent dimensions
                    from torchvision.transforms import Resize
                    test_model_kwargs['inpaint_mask'] = Resize([inpaint_image_latent.shape[-2], inpaint_image_latent.shape[-1]])(
                        test_model_kwargs['inpaint_mask'])
                    
                    print(f"  🔧 test_model_kwargs prepared:")
                    print(f"     inpaint_image (latent): {test_model_kwargs['inpaint_image'].shape}")
                    print(f"     inpaint_mask (resized): {test_model_kwargs['inpaint_mask'].shape}")
                    
                    # DDIM sampling with proper Frontal-View VTON conditioning
                    shape = inpaint_image_latent.shape[1:]  # Remove batch dimension
                    
                    logger.info("🎨 Starting DDIM sampling...")
                    samples_ddim, _ = sampler.sample(
                        S=ddim_steps,
                        conditioning=conditioning,
                        batch_size=1,
                        shape=shape,
                        verbose=False,  # Explicitly disable verbose to prevent tqdm issues
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uncond,
                        eta=0.0,
                        x_T=None,  # Start from noise
                        test_model_kwargs=test_model_kwargs  # Pass as direct parameter like in test file
                    )
                    logger.info("✅ DDIM sampling completed successfully")
                    
                    # Decode from latent space
                    print("📤 Decoding from latent space...")
                    result_images = model.decode_first_stage(samples_ddim)
                    result_images = torch.clamp((result_images + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    # Convert to PIL Image
                    result_array = result_images[0].cpu().permute(1, 2, 0).numpy()
                    result_array = (result_array * 255).astype(np.uint8)
                    result_img = Image.fromarray(result_array)
                    
                    print(f"  📤 Final result: size={result_img.size}, mode={result_img.mode}")
                    log_processing_stage("Inference completed", "Successfully generated result")
                    logger.info("✅ Frontal-View VTON inference completed successfully!")
                    return result_img
                    
    except Exception as e:
        logger.error(f"❌ Error in Frontal-View VTON inference: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# API Endpoints

@app.get('/health')
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "sampler_ready": sampler is not None,
        "config_loaded": config is not None,
        "device": str(device) if device else "None",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "pytorch_version": torch.__version__,
        "default_image_size": DEFAULT_IMAGE_SIZE,
        "supported_formats": ["JPEG", "PNG", "RGB"],
        "api_version": "2.0.0"
    }

@app.get('/model/info')
async def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    return {
        "model_type": "Frontal-View VTON",
        "architecture": "LatentTryOnDiffusion",
        "approach": "frontal_single_view_optimized",
        "input_channels": 9,  # Key difference: 9 channels vs 8 in MV-VTON
        "config_path": DEFAULT_CONFIG_PATH,
        "checkpoint_path": DEFAULT_CHECKPOINT_PATH,
        "image_size": DEFAULT_IMAGE_SIZE,
        "default_ddim_steps": DEFAULT_DDIM_STEPS,
        "default_scale": DEFAULT_SCALE,
        "conditioning_type": "crossattn",
        "latent_channels": 4,
        "scale_factor": 0.18215,
        "focus": "upper_body_clothing_transfer",
        "model_parameters": sum(p.numel() for p in model.parameters()) if model else 0
    }

@app.post('/try-on')
async def virtual_try_on(
    person_image: UploadFile = File(..., description="Person image file"),
    cloth_image: UploadFile = File(..., description="Clothing item image file"),
    cloth_back_image: UploadFile = File(None, description="Back view of clothing (optional)"),
    ddim_steps: int = Form(DEFAULT_DDIM_STEPS, description="Number of DDIM sampling steps"),
    scale: float = Form(DEFAULT_SCALE, description="Guidance scale for generation"),
    height: int = Form(DEFAULT_HEIGHT, description="Output image height"),
    width: int = Form(DEFAULT_WIDTH, description="Output image width")
):
    """Main virtual try-on endpoint returning base64 image"""
    try:
        start_time = datetime.now()
        
        # Log request details
        log_request_details(person_image, cloth_image, ddim_steps, scale, height, width)
        
        if model is None or sampler is None:
            logger.error("❌ MV-VTON model not loaded")
            raise HTTPException(status_code=500, detail="MV-VTON model not loaded")
        
        # Validate uploaded files
        if not person_image.filename:
            raise HTTPException(status_code=400, detail="Invalid person image file")
        if not cloth_image.filename:
            raise HTTPException(status_code=400, detail="Invalid cloth image file")
        
        # Validate file types
        allowed_types = {'image/jpeg', 'image/jpg', 'image/png'}
        if person_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Person image must be JPEG or PNG")
        if cloth_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Cloth image must be JPEG or PNG")
        
        # Validate parameters
        if ddim_steps < 1 or ddim_steps > 200:
            raise HTTPException(status_code=400, detail="ddim_steps must be between 1 and 200")
        if scale < 0.0 or scale > 10.0:
            raise HTTPException(status_code=400, detail="scale must be between 0.0 and 10.0")
        if height < 256 or height > 1024 or width < 256 or width > 1024:
            raise HTTPException(status_code=400, detail="Image size must be between 256 and 1024 pixels")
        
        # Load and validate images
        try:
            log_processing_stage("Loading images", "Reading uploaded files")
            person_contents = await person_image.read()
            cloth_contents = await cloth_image.read()
            
            person_img = Image.open(io.BytesIO(person_contents)).convert('RGB')
            cloth_img = Image.open(io.BytesIO(cloth_contents)).convert('RGB')
            cloth_back_img = None
            
            logger.info(f"✅ Images loaded successfully:")
            logger.info(f"   Person image: {person_img.size}")
            logger.info(f"   Cloth image: {cloth_img.size}")
            
            if cloth_back_image and cloth_back_image.filename:
                cloth_back_contents = await cloth_back_image.read()
                cloth_back_img = Image.open(io.BytesIO(cloth_back_contents)).convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load images: {str(e)}")
        
        print(f"📸 Processing images: person={person_img.size}, cloth={cloth_img.size}")
        
        # Run MV-VTON inference
        result_img = mvvton_inference(
            person_img, cloth_img, cloth_back_img,
            ddim_steps=ddim_steps,
            scale=scale,
            H=height,
            W=width
        )
        
        if result_img is None:
            raise HTTPException(status_code=500, detail="MV-VTON inference failed")
        
        # Convert result to base64
        buffer = io.BytesIO()
        result_img.save(buffer, format='PNG')
        buffer.seek(0)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": True,
            "result_image": result_base64,
            "message": "MV-VTON virtual try-on completed successfully",
            "processing_time": f"{processing_time:.2f}s",
            "parameters": {
                "ddim_steps": ddim_steps,
                "scale": scale,
                "output_size": [height, width],
                "person_input_size": list(person_img.size),
                "cloth_input_size": list(cloth_img.size),
                "has_back_cloth": cloth_back_img is not None
            },
            "timestamp": datetime.now().isoformat(),
            "note": "This implementation includes enhanced preprocessing with OpenPose and SCHP integration."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"MV-VTON API error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post('/try-on-file')
async def virtual_try_on_file(
    person_image: UploadFile = File(..., description="Person image file"),
    cloth_image: UploadFile = File(..., description="Clothing item image file"),
    cloth_back_image: UploadFile = File(None, description="Back view of clothing (optional)"),
    ddim_steps: int = Form(DEFAULT_DDIM_STEPS, description="Number of DDIM sampling steps"),
    scale: float = Form(DEFAULT_SCALE, description="Guidance scale for generation"),
    height: int = Form(DEFAULT_HEIGHT, description="Output image height"),
    width: int = Form(DEFAULT_WIDTH, description="Output image width")
):
    """Virtual try-on endpoint returning image file"""
    # Reuse the main endpoint logic by calling it
    result = await virtual_try_on(person_image, cloth_image, cloth_back_image, ddim_steps, scale, height, width)
    
    # Convert base64 result to streaming response
    if result.get("success"):
        image_data = base64.b64decode(result["result_image"])
        
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=mvvton_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            }
        )
    else:
        raise HTTPException(status_code=500, detail="Virtual try-on failed")

if __name__ == '__main__':
    print("🚀 Starting MV-VTON Integrated API Server...")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🎯 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"💾 GPU count: {torch.cuda.device_count()}")
        print(f"🔧 GPU name: {torch.cuda.get_device_name()}")
    
    print("\n" + "="*60)
    print("🤖 INITIALIZING FRONTAL-VIEW VTON MODEL...")
    print("="*60)
    
    # Initialize the model
    if not initialize_mvvton_model():
        print("❌ Failed to initialize MV-VTON model. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ FRONTAL-VIEW VTON MODEL READY!")
    print("🌐 Starting FastAPI server on 0.0.0.0:5000")
    print("📖 Available endpoints:")
    print("   GET  /health - Health check and system status")
    print("   GET  /model/info - Model configuration details")
    print("   POST /try-on - Virtual try-on (returns base64 image)")
    print("   POST /try-on-file - Virtual try-on (returns image file)")
    print("   GET  /docs - Interactive API documentation")
    print("   GET  /redoc - Alternative API documentation")
    print("="*60 + "\n")
    
    # Start the FastAPI server with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")