#!/usr/bin/env python3
"""
Sample Images Quality Testing for MV-VTON
Tests virtual try-on quality with available sample images from assets directory
"""

import requests
import json
import base64
import time
import os
from PIL import Image, ImageStat
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleImagesQualityTester:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.person_dir = Path("assets/person")
        self.cloth_dir = Path("assets/cloth") 
        self.output_dir = Path("sample_test_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def calculate_image_quality_metrics(self, image_path):
        """Calculate comprehensive image quality metrics"""
        if not os.path.exists(image_path):
            return {"quality_score": 0, "error": "Image not found"}
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Basic statistics
            stat = ImageStat.Stat(img)
            
            metrics = {
                "mean_brightness": float(np.mean(stat.mean)),
                "std_brightness": float(np.std(stat.mean)),
                "contrast": float(np.std(img_array)),
                "sharpness": float(np.var(np.array(img.convert('L')))),
                "entropy": float(img.convert('L').entropy()) if hasattr(img.convert('L'), 'entropy') else 0,
                "aspect_ratio": img.width / img.height,
                "resolution": f"{img.width}x{img.height}",
                "file_size_mb": os.path.getsize(image_path) / (1024 * 1024)
            }
            
            # Calculate overall quality score
            metrics["quality_score"] = self._assess_overall_quality(metrics)
            return metrics
            
        except Exception as e:
            return {"quality_score": 0, "error": str(e)}
    
    def _assess_overall_quality(self, metrics):
        """Assess overall image quality based on multiple factors"""
        try:
            # Normalize metrics to 0-100 scale
            sharpness_score = min(100, (metrics["sharpness"] / 500) * 100)
            contrast_score = min(100, (metrics["contrast"] / 100) * 100)
            entropy_score = min(100, (metrics["entropy"] / 8) * 100)
            
            # Brightness should be moderate (not too dark/bright)
            brightness_penalty = abs(metrics["mean_brightness"] - 128) / 128
            brightness_score = max(0, 100 - (brightness_penalty * 100))
            
            # Weighted average
            quality_score = (
                sharpness_score * 0.3 +
                contrast_score * 0.3 +
                entropy_score * 0.2 +
                brightness_score * 0.2
            )
            
            return float(quality_score)
        except:
            return 0.0
    
    def test_virtual_tryon(self, person_img, cloth_img, test_name):
        """Test virtual try-on with specific images"""
        logger.info(f"🧪 Testing {test_name}")
        logger.info(f"   Person: {person_img.name}")
        logger.info(f"   Cloth: {cloth_img.name}")
        
        start_time = time.time()
        
        try:
            # Prepare multipart form data
            files = {
                'person_image': ('person.jpg', open(person_img, 'rb'), 'image/jpeg'),
                'cloth_image': ('cloth.jpg', open(cloth_img, 'rb'), 'image/jpeg')
            }
            
            data = {
                'ddim_steps': 20,
                'scale': 2.5,
                'height': 512,
                'width': 384
            }
            
            response = requests.post(f"{self.api_url}/try-on", 
                                   files=files,
                                   data=data,
                                   timeout=60)
            
            # Close file handles
            files['person_image'][1].close()
            files['cloth_image'][1].close()
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Save result image
                result_image_data = base64.b64decode(result_data["result_image"])
                result_path = self.output_dir / f"{test_name}_result.jpg"
                
                with open(result_path, "wb") as f:
                    f.write(result_image_data)
                
                # Calculate metrics
                person_metrics = self.calculate_image_quality_metrics(person_img)
                cloth_metrics = self.calculate_image_quality_metrics(cloth_img)  
                result_metrics = self.calculate_image_quality_metrics(result_path)
                
                test_result = {
                    "test_name": test_name,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    "person_image": str(person_img),
                    "cloth_image": str(cloth_img),
                    "processing_time_seconds": processing_time,
                    "api_response_status": response.status_code,
                    "person_metrics": person_metrics,
                    "cloth_metrics": cloth_metrics,
                    "result_metrics": result_metrics,
                    "result_image_path": str(result_path),
                    "quality_improvement": result_metrics.get("quality_score", 0) - person_metrics.get("quality_score", 0),
                    "success": True
                }
                
                logger.info(f"✅ {test_name} completed!")
                logger.info(f"   Processing time: {processing_time:.2f}s")
                logger.info(f"   Person quality: {person_metrics.get('quality_score', 0):.1f}/100")
                logger.info(f"   Cloth quality: {cloth_metrics.get('quality_score', 0):.1f}/100")
                logger.info(f"   Result quality: {result_metrics.get('quality_score', 0):.1f}/100")
                
                return test_result
                
            else:
                logger.error(f"❌ {test_name} failed with status {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return {
                    "test_name": test_name,
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                    "person_image": str(person_img),
                    "cloth_image": str(cloth_img),
                    "processing_time_seconds": processing_time,
                    "api_response_status": response.status_code,
                    "error_message": response.text,
                    "success": False
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ {test_name} exception: {str(e)}")
            return {
                "test_name": test_name,
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                "person_image": str(person_img),
                "cloth_image": str(cloth_img),
                "processing_time_seconds": processing_time,
                "exception": str(e),
                "success": False
            }
    
    def run_comprehensive_test(self):
        """Run comprehensive test with available sample images"""
        logger.info("🚀 Starting comprehensive sample images quality test")
        logger.info("=" * 60)
        
        # Get available images
        person_images = list(self.person_dir.glob("*.jpg"))[:3]  # Test first 3
        cloth_images = list(self.cloth_dir.glob("*.jpg"))[:3]   # Test first 3
        
        logger.info(f"📋 Found {len(person_images)} person images, {len(cloth_images)} cloth images")
        
        if not person_images or not cloth_images:
            logger.error("❌ No sample images found!")
            return
        
        # Test combinations
        test_results = []
        test_count = 0
        
        for i, person_img in enumerate(person_images):
            for j, cloth_img in enumerate(cloth_images):
                test_count += 1
                test_name = f"sample_test_{test_count:02d}_{person_img.stem}_{cloth_img.stem}"
                
                result = self.test_virtual_tryon(person_img, cloth_img, test_name)
                test_results.append(result)
                
                # Brief pause between tests
                time.sleep(1)
                
                # Max 6 tests to avoid overwhelming
                if test_count >= 6:
                    break
            if test_count >= 6:
                break
        
        # Generate summary
        successful_tests = [r for r in test_results if r.get("success", False)]
        
        if successful_tests:
            avg_quality = np.mean([r["result_metrics"]["quality_score"] for r in successful_tests])
            avg_time = np.mean([r["processing_time_seconds"] for r in successful_tests])
            best_quality = max(r["result_metrics"]["quality_score"] for r in successful_tests)
            worst_quality = min(r["result_metrics"]["quality_score"] for r in successful_tests)
            
            summary = {
                "test_summary": {
                    "total_tests": len(test_results),
                    "successful_tests": len(successful_tests),
                    "average_quality_score": avg_quality,
                    "average_processing_time": avg_time,
                    "best_quality_score": best_quality,
                    "worst_quality_score": worst_quality
                },
                "detailed_results": test_results,
                "assessment_timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%f')
            }
            
            # Save results
            results_file = self.output_dir / "sample_images_assessment.json"
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("=" * 60)
            logger.info("📊 SAMPLE IMAGES TEST SUMMARY")
            logger.info(f"   Total tests: {len(test_results)}")
            logger.info(f"   Successful: {len(successful_tests)}")
            logger.info(f"   Average quality: {avg_quality:.1f}/100")
            logger.info(f"   Average time: {avg_time:.1f}s")
            logger.info(f"   Quality range: {worst_quality:.1f} - {best_quality:.1f}")
            logger.info(f"   Results saved: {results_file}")
            
            return summary
        else:
            logger.error("❌ No successful tests completed!")
            return {"error": "All tests failed"}

if __name__ == "__main__":
    tester = SampleImagesQualityTester()
    tester.run_comprehensive_test()