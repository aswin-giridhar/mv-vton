# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MV-VTON is a PyTorch implementation of "Multi-View Virtual Try-On with Diffusion Models". It includes both multi-view and frontal-view virtual try-on capabilities using diffusion models.

## Environment Setup

Create conda environment:
```bash
conda env create -f environment.yaml
conda activate mv-vton
```

## Core Architecture

### Directory Structure
- `ldm/`: Core diffusion model implementation
  - `ldm/models/diffusion/ddpm.py`: Main diffusion model (LatentTryOnDiffusion)
  - `ldm/data/`: Dataset implementations for different scenarios
  - `ldm/modules/`: Model components (attention, encoders, etc.)
  - `ldm/ip_adapter/`: IP-Adapter implementation for clothing feature extraction
- `configs/viton512.yaml`: Main model configuration
- `Frontal-View VTON/`: Frontal-view specific implementation
- `checkpoint/`: Model checkpoints directory
- `models/vgg/`: VGG model for perceptual loss

#### Preprocessing Pipeline Components (DressCode Dataset Methodology)
Following DressCode dataset (https://github.com/aimagelab/dress-code), we used OpenPose to extract 18 keypoints for each human body, and used the SCHP model trained on the ATR dataset for parsing results, and then get the corresponding agnostic images.

- `pytorch-openpose/`: OpenPose implementation (https://github.com/Hzzone/pytorch-openpose)
  - **Purpose**: Extract 18 keypoints for human body pose detection
  - **Integration**: Used by `proper_preprocessing.py` and `mvvton_api_server.py`
  - **Models**: 
    - `model/body_pose_model.pth`: Body pose detection (209MB)
    - `model/hand_pose_model.pth`: Hand pose detection (147MB)
  - **Key Files**:
    - `src/`: Core OpenPose implementation modules
    - `demo.py`: Basic pose detection demo
    - `demo_camera.py`: Real-time camera pose detection
  - **Usage**: Generates pose skeletons for conditioning the diffusion model

- `Self-Correction-Human-Parsing/`: SCHP human parsing (https://github.com/PeikeLi/Self-Correction-Human-Parsing)
  - **Purpose**: Segment human body parts for precise clothing region detection
  - **Integration**: Used for generating person-agnostic images and clothing masks
  - **Pre-trained Models**:
    - `checkpoints/atr.pth`: Trained on ATR dataset (267MB) - 18 labels including clothing parts
    - `checkpoints/lip.pth`: Trained on LIP dataset (267MB) - 20 labels for complex scenarios  
    - `checkpoints/pascal.pth`: Trained on Pascal-Person-Part (267MB) - 7 body part labels
  - **Key Files**:
    - `simple_extractor.py`: Easy-to-use parsing extractor
    - `networks/`: Neural network architectures
    - `inputs/` and `outputs/`: Processing directories
  - **ATR Dataset Labels**: Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt, Left-shoe, Right-shoe, Face, Left-leg, Right-leg, Left-arm, Right-arm, Bag, Scarf

#### Integration Files
- `proper_preprocessing.py`: Enhanced preprocessing pipeline combining OpenPose and SCHP
  - Implements DressCode dataset methodology
  - Generates 18-point pose skeletons using OpenPose
  - Creates human parsing masks using SCHP ATR model
  - Produces person-agnostic images by removing clothing regions
- `mvvton_api_server.py`: Main API server with integrated preprocessing pipeline

### Key Components
- **LatentTryOnDiffusion**: Main diffusion model class in `ldm/models/diffusion/ddpm.py`
- **Dataset classes**: 
  - `cp_dataset_mv_paired.py`: Multi-view paired dataset
  - `cp_dataset_mv_unpaired.py`: Multi-view unpaired dataset
- **View-adaptive selection**: Hard and soft selection methods for clothing features
- **Joint attention block**: Aligns and fuses clothing with person features

## Dataset Configuration

The model expects dataset switching via renaming:
- For paired testing: rename `cp_dataset_mv_paired.py` to `cp_dataset.py`
- For unpaired testing: rename `cp_dataset_mv_unpaired.py` to `cp_dataset.py`

### Preprocessing Pipeline Integration

The MV-VTON system requires proper preprocessing following the DressCode dataset methodology:

1. **OpenPose Keypoint Extraction** (`pytorch-openpose/`)
   - Extracts 18 keypoints from person images
   - Generates pose skeleton maps for model conditioning
   - Essential for understanding human body structure

2. **Human Parsing** (`Self-Correction-Human-Parsing/`)
   - Uses SCHP model trained on ATR dataset
   - Segments body parts: clothing regions, arms, legs, face, hair
   - Creates precise masks for clothing region identification

3. **Person-Agnostic Image Generation**
   - Removes clothing from person images based on parsing masks
   - Creates "blank canvas" for virtual try-on
   - Preserves body structure while removing original clothing

4. **Integration Flow**:
   ```
   Input Image → OpenPose (18 keypoints) → SCHP Parsing (body segments) → 
   Person-Agnostic Image → MV-VTON Model → Virtual Try-On Result
   ```

## Common Commands

### Preprocessing Setup
Initialize OpenPose and SCHP models:
```bash
# Test OpenPose keypoint detection
cd pytorch-openpose/
python demo.py --input_dir ../assets/person --output_dir ../outputs/openpose

# Test SCHP human parsing
cd ../Self-Correction-Human-Parsing/
python simple_extractor.py --input_dir ../assets/person --output_dir ../outputs/parsing
```

### API Server Usage (Recommended)
Start the complete system with proper preprocessing:
```bash
# Environment isolation approach (resolves package conflicts)
./start_complete_system.sh

# Option 3: Start both API server and Gradio app
# API Server: mv-vton conda environment (localhost:5000)  
# Gradio UI: base environment (localhost:7860)
```

Individual components:
```bash
# API server only (with enhanced preprocessing)
./start_api_server.sh

# Gradio interface only (requires API server running)
./start_gradio_app.sh
```

### Testing
Multi-view (MVG dataset):
```bash
# Paired setting
sh test.sh

# For unpaired, rename cp_dataset_mv_unpaired.py to cp_dataset.py first
```

Frontal-view (VITON-HD dataset):
```bash
cd "Frontal-View VTON/"
# Paired
sh test.sh
# Unpaired
sh test.sh  # (add --unpaired flag to test.sh)
```

### Training
Multi-view:
```bash
sh train.sh
```

Frontal-view:
```bash
cd "Frontal-View VTON/"
sh train.sh
```

## Model Checkpoints Required

1. **VGG checkpoint**: Download and place in `models/vgg/` (and `Frontal-View VTON/models/vgg/`)
2. **MVG checkpoint**: `mvg.ckpt` in `checkpoint/` 
3. **VITON-HD checkpoint**: `vitonhd.ckpt` in `Frontal-View VTON/checkpoint/`
4. **Paint-by-Example initialization**: Required for training, place in `checkpoints/`

### Preprocessing Model Checkpoints (Required for Quality Results)

**OpenPose Models** (`pytorch-openpose/model/`):
- `body_pose_model.pth`: PyTorch body pose model (209MB)
- `hand_pose_model.pth`: PyTorch hand pose model (147MB)
- Alternative: `body_pose.caffemodel` and `hand_pose.caffemodel` for Caffe backend

**SCHP Human Parsing Models** (`Self-Correction-Human-Parsing/checkpoints/`):
- `atr.pth`: ATR dataset model (267MB) - **Recommended for clothing-focused parsing**
- `lip.pth`: LIP dataset model (267MB) - For complex real-world scenarios  
- `pascal.pth`: Pascal-Person-Part model (267MB) - For body part segmentation

**Note**: The ATR model is specifically recommended as it aligns with the DressCode dataset methodology used in the original MV-VTON paper.

## Configuration

Main config file: `configs/viton512.yaml`
- Model: LatentTryOnDiffusion
- Image size: 512x384
- Uses crossattn conditioning
- Scale factor: 0.18215

## API Servers

Several API server implementations are available:
- `mvvton_api_server.py`: Main API server with integrated preprocessing pipeline
- `proper_preprocessing.py`: Enhanced preprocessing module with OpenPose and SCHP integration
- `gradio_app.py`: Full-featured Gradio interface with advanced parameters
- `test_environment_isolation.py`: System verification and compatibility testing

### Environment Isolation Setup
The system uses environment isolation to resolve package conflicts:
- **API Server**: Runs in `mv-vton` conda environment (PyTorch, CUDA, ML models)
- **Gradio UI**: Runs in base Python environment (modern web frameworks)
- **Benefits**: Eliminates dependency conflicts, optimal performance for both components

### Quality Improvements
Recent enhancements address poor virtual try-on quality by implementing:
- **18-point OpenPose** skeletal detection (replacing simple stick figures)
- **SCHP human parsing** for accurate body segmentation (replacing rectangular masks)
- **Person-agnostic image generation** following DressCode methodology
- **Enhanced clothing region detection** with proper body part labeling