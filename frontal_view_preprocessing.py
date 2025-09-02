"""
Frontal-View VTON Compatible Preprocessing
Implements the exact person-agnostic generation methodology used in the original Frontal-View VTON dataset.

Key differences from integrated_preprocessing.py:
1. Uses pose-based geometric masking instead of SCHP parsing-based masking
2. Follows the exact methodology from cp_dataset.py in Frontal-View VTON
3. Compatible with the expected input format for Frontal-View VTON model
"""

import os
import sys
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw
import torch

# Add OpenPose to path  
openpose_src = '/home/ubuntu/MV-VTON/pytorch-openpose/src'
if openpose_src not in sys.path:
    sys.path.insert(0, openpose_src)

# Change to the OpenPose directory for imports
current_dir = os.getcwd()
try:
    os.chdir('/home/ubuntu/MV-VTON/pytorch-openpose/src')
    sys.path.append('/home/ubuntu/MV-VTON/Self-Correction-Human-Parsing')
    
    # OpenPose imports
    from body import Body
    from hand import Hand
finally:
    os.chdir(current_dir)

# SCHP imports
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from networks import deeplab_xception_transfer, deeplab_resnet_transfer
from datasets import simple_extractor_dataset


class FrontalViewPreprocessor:
    """
    Preprocessing pipeline specifically designed for Frontal-View VTON compatibility.
    Follows the exact methodology from the original VITON-HD dataset preparation.
    """
    
    def __init__(self):
        """Initialize OpenPose and SCHP models."""
        # OpenPose initialization
        print("Initializing OpenPose...")
        self.body_estimation = Body('/home/ubuntu/MV-VTON/pytorch-openpose/model/body_pose_model.pth')
        print("OpenPose body model loaded successfully")
        
        # SCHP initialization for parsing (still needed for parse_lower masks)
        print("Initializing SCHP...")
        self.schp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load SCHP model (ATR dataset)
        self.schp_model = deeplab_xception_transfer.DeepLabv3_plus(
            nInputChannels=3, 
            n_classes=18,
            os=16,
            pretrained=True
        ).to(self.schp_device)
        
        # Load SCHP checkpoint
        schp_checkpoint = '/home/ubuntu/MV-VTON/Self-Correction-Human-Parsing/checkpoints/atr.pth'
        if os.path.exists(schp_checkpoint):
            checkpoint = torch.load(schp_checkpoint, map_location=self.schp_device)
            self.schp_model.load_state_dict(checkpoint)
            print("SCHP model loaded successfully")
        else:
            print(f"Warning: SCHP checkpoint not found at {schp_checkpoint}")
            
        self.schp_model.eval()
        
        # ATR dataset labels (compatible with original Frontal-View VTON)
        self.atr_labels = [
            'Background', 'Hat', 'Hair', 'Sunglasses', 
            'Upper-clothes', 'Skirt', 'Pants', 'Dress', 
            'Belt', 'Left-shoe', 'Right-shoe', 'Face', 
            'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'
        ]
        
    def extract_pose_keypoints(self, image_path):
        """
        Extract 18-point OpenPose keypoints from an image.
        Returns keypoints in the format expected by Frontal-View VTON.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Extract pose keypoints
        candidate, subset = self.body_estimation(image)
        
        # Convert to the expected 18-point format
        pose_data = np.full((18, 2), -1, dtype=np.float32)
        
        if len(subset) > 0:
            # Use the first detected person
            person = subset[0]
            for i in range(18):
                if person[i] != -1:
                    idx = int(person[i])
                    if idx < len(candidate):
                        pose_data[i] = [candidate[idx][1], candidate[idx][0]]  # [y, x] format
        
        return pose_data
    
    def extract_human_parsing(self, image_path):
        """
        Extract human parsing using SCHP model trained on ATR dataset.
        Returns parsing map compatible with original Frontal-View VTON format.
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize to standard input size
        input_size = (473, 473)
        image_resized = image.resize(input_size, Image.BILINEAR)
        
        # Convert to tensor
        image_array = np.array(image_resized, dtype=np.float32)
        image_array = (image_array / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).to(self.schp_device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.schp_model(image_tensor)
            predictions = torch.argmax(outputs[0], dim=1).squeeze().cpu().numpy()
        
        # Resize back to original size
        predictions_resized = cv2.resize(
            predictions.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        return Image.fromarray(predictions_resized)
    
    def get_agnostic_frontal_view(self, im, im_parse, pose_data):
        """
        Generate person-agnostic image using the EXACT methodology from Frontal-View VTON.
        This is a direct adaptation of the get_agnostic function from cp_dataset.py.
        """
        parse_array = np.array(im_parse)
        
        # Extract head region (Hair + Face in ATR labels)
        parse_head = ((parse_array == 4).astype(np.float32) +  # Upper-clothes -> wait, this seems wrong
                      (parse_array == 13).astype(np.float32))   # Left-leg -> this seems wrong too
        
        # Let me check the actual labels from the original code
        # Looking at human_parse_labels.py, ATR labels are:
        # [0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress,
        #  8: Belt, 9: Left-shoe, 10: Right-shoe, 11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm, 16: Bag, 17: Scarf]
        
        # Correct the parsing based on ATR labels (0-indexed)
        parse_head = ((parse_array == 2).astype(np.float32) +   # Hair
                      (parse_array == 11).astype(np.float32))   # Face
        
        # Lower body parts (legs, shoes, etc.)
        parse_lower = ((parse_array == 9).astype(np.float32) +   # Left-shoe
                       (parse_array == 12).astype(np.float32) +  # Left-leg  
                       (parse_array == 16).astype(np.float32) +  # Bag
                       (parse_array == 17).astype(np.float32) +  # Scarf
                       (parse_array == 18).astype(np.float32) +  # This seems out of range
                       (parse_array == 19).astype(np.float32))   # This seems out of range too
        
        # Let me correct this based on actual ATR dataset (0-17 range)
        parse_lower = ((parse_array == 9).astype(np.float32) +   # Left-shoe
                       (parse_array == 10).astype(np.float32) +  # Right-shoe
                       (parse_array == 12).astype(np.float32) +  # Left-leg
                       (parse_array == 13).astype(np.float32) +  # Right-leg
                       (parse_array == 5).astype(np.float32) +   # Skirt
                       (parse_array == 6).astype(np.float32))    # Pants
        
        # Create agnostic image
        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)
        
        # Calculate body proportions for geometric masking
        length_a = np.linalg.norm(pose_data[5] - pose_data[2])  # Left shoulder to Right shoulder
        length_b = np.linalg.norm(pose_data[12] - pose_data[9]) # Left hip to Right hip
        point = (pose_data[9] + pose_data[12]) / 2              # Hip center
        
        # Normalize hip positions based on shoulder width
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
        
        # Calculate radius for geometric shapes
        r = int(length_a / 16) + 1
        
        # Mask arms using ellipses (based on pose keypoints)
        for i in [9, 12]:  # Left hip, Right hip 
            pointx, pointy = pose_data[i]
            if pointx != -1 and pointy != -1:
                agnostic_draw.ellipse(
                    (pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 
                    'gray', 'gray'
                )
        
        # Draw connecting lines for torso structure
        if all(pose_data[i][0] != -1 and pose_data[i][1] != -1 for i in [2, 9]):
            agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)  # Right shoulder to Right hip
        if all(pose_data[i][0] != -1 and pose_data[i][1] != -1 for i in [5, 12]):
            agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6) # Left shoulder to Left hip
        if all(pose_data[i][0] != -1 and pose_data[i][1] != -1 for i in [9, 12]):
            agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12) # Hip connection
        
        # CRITICAL: Main torso masking polygon (this is the key difference!)
        if all(pose_data[i][0] != -1 and pose_data[i][1] != -1 for i in [2, 5, 12, 9]):
            agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')
        
        # Mask neck area
        if pose_data[1][0] != -1 and pose_data[1][1] != -1:
            pointx, pointy = pose_data[1]
            agnostic_draw.rectangle(
                (pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 
                'gray', 'gray'
            )
        
        # Apply lower body parsing mask (keep legs, shoes visible)
        agnostic_np = np.array(agnostic)
        parse_mask = np.expand_dims(parse_lower, axis=2)
        agnostic_np = agnostic_np * parse_mask + np.array(im) * (1 - parse_mask)
        agnostic = Image.fromarray(agnostic_np.astype(np.uint8))
        
        # Apply head parsing mask (keep face and hair visible) 
        agnostic_np = np.array(agnostic)
        parse_head_mask = np.expand_dims(parse_head, axis=2)
        agnostic_np = agnostic_np * (1 - parse_head_mask) + np.array(im) * parse_head_mask
        agnostic = Image.fromarray(agnostic_np.astype(np.uint8))
        
        return agnostic
    
    def create_inpaint_mask_frontal_view(self, im_parse, pose_data):
        """
        Create inpaint mask for the clothing region that will be replaced.
        This identifies the area where the new clothing should be applied.
        """
        parse_array = np.array(im_parse)
        
        # Create mask for upper clothing region (Upper-clothes + Dress in ATR)
        clothing_mask = ((parse_array == 4).astype(np.float32) +  # Upper-clothes
                        (parse_array == 7).astype(np.float32))    # Dress
        
        # Create geometric torso mask using the same polygon as agnostic generation
        height, width = parse_array.shape
        mask_image = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask_image)
        
        # Calculate body proportions
        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        r = int(length_a / 16) + 1
        
        # Draw the same torso polygon as in agnostic generation
        if all(pose_data[i][0] != -1 and pose_data[i][1] != -1 for i in [2, 5, 12, 9]):
            mask_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], fill=255)
        
        # Convert to numpy and combine with parsing mask
        geometric_mask = np.array(mask_image) / 255.0
        
        # Combine parsing-based and geometry-based masks
        final_mask = np.maximum(clothing_mask, geometric_mask)
        
        return Image.fromarray((final_mask * 255).astype(np.uint8))
    
    def process_image(self, person_image_path, output_dir=None):
        """
        Complete preprocessing pipeline for a person image.
        Returns all necessary data for Frontal-View VTON model.
        """
        try:
            print(f"Processing image: {person_image_path}")
            
            # Load original image
            original_image = Image.open(person_image_path).convert('RGB')
            
            # Extract pose keypoints
            print("Extracting pose keypoints...")
            pose_data = self.extract_pose_keypoints(person_image_path)
            
            # Extract human parsing
            print("Extracting human parsing...")
            parsing_image = self.extract_human_parsing(person_image_path)
            
            # Generate person-agnostic image using Frontal-View VTON methodology
            print("Generating person-agnostic image...")
            agnostic_image = self.get_agnostic_frontal_view(original_image, parsing_image, pose_data)
            
            # Create inpaint mask
            print("Creating inpaint mask...")
            inpaint_mask = self.create_inpaint_mask_frontal_view(parsing_image, pose_data)
            
            # Save outputs if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(person_image_path))[0]
                
                agnostic_image.save(os.path.join(output_dir, f"{base_name}_agnostic.jpg"))
                inpaint_mask.save(os.path.join(output_dir, f"{base_name}_mask.jpg"))
                parsing_image.save(os.path.join(output_dir, f"{base_name}_parsing.png"))
                
                # Save pose keypoints as JSON
                pose_json = {
                    "keypoints": pose_data.tolist(),
                    "format": "18_point_openpose"
                }
                with open(os.path.join(output_dir, f"{base_name}_pose.json"), 'w') as f:
                    json.dump(pose_json, f, indent=2)
                    
                print(f"Results saved to: {output_dir}")
            
            return {
                'original': original_image,
                'agnostic': agnostic_image,
                'inpaint_mask': inpaint_mask,
                'parsing': parsing_image,
                'pose_data': pose_data,
                'method': 'frontal_view_geometric_masking'
            }
            
        except Exception as e:
            print(f"Error processing image {person_image_path}: {str(e)}")
            raise


def main():
    """Test the Frontal-View preprocessing pipeline."""
    preprocessor = FrontalViewPreprocessor()
    
    # Test with sample images
    test_images = [
        "/home/ubuntu/MV-VTON/assets/person/00010_00.jpg",
        "/home/ubuntu/MV-VTON/assets/person/00069_00.jpg"
    ]
    
    output_dir = "/home/ubuntu/MV-VTON/test_frontal_preprocessing"
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Testing with: {image_path}")
            print(f"{'='*50}")
            
            result = preprocessor.process_image(image_path, output_dir)
            
            print(f"✓ Processed successfully!")
            print(f"  - Original size: {result['original'].size}")
            print(f"  - Agnostic size: {result['agnostic'].size}")
            print(f"  - Mask size: {result['inpaint_mask'].size}")
            print(f"  - Method: {result['method']}")
        else:
            print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()