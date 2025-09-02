#!/usr/bin/env python3
"""
Virtual Try-On Quality Assessment Tool
Tests the complete MV-VTON pipeline and evaluates results
"""

import os
import sys
import json
import time
import requests
import base64
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
import cv2

# Configuration
API_URL = "http://localhost:5000"
TEST_RESULTS_DIR = "test_results"
QUALITY_METRICS_FILE = "quality_assessment.json"

class VirtualTryOnQualityTester:
    def __init__(self):
        self.api_url = API_URL
        self.results_dir = Path(TEST_RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)
        self.test_results = []
        
    def log(self, message, level="INFO"):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def encode_image(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            self.log(f"Failed to encode image {image_path}: {e}", "ERROR")
            return None
            
    def check_api_health(self):
        """Check if API server is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                self.log("✅ API server is healthy")
                self.log(f"   Model loaded: {health_data.get('model_loaded', 'Unknown')}")
                self.log(f"   Device: {health_data.get('device', 'Unknown')}")
                self.log(f"   GPU count: {health_data.get('gpu_count', 'Unknown')}")
                return True
            else:
                self.log(f"❌ API health check failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Cannot connect to API server: {e}", "ERROR")
            return False
            
    def calculate_image_quality_metrics(self, image_path):
        """Calculate comprehensive image quality metrics"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Basic statistics
            stat = ImageStat.Stat(img)
            
            # Convert to OpenCV format for advanced metrics
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            metrics = {
                'mean_brightness': float(np.mean(stat.mean)),
                'std_brightness': float(np.mean(stat.stddev)),
                'contrast': float(np.std(gray)),
                'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                'entropy': float(self._calculate_entropy(gray)),
                'aspect_ratio': img.width / img.height,
                'resolution': f"{img.width}x{img.height}",
                'file_size_mb': os.path.getsize(image_path) / (1024 * 1024)
            }
            
            # Quality assessment
            metrics['quality_score'] = self._assess_overall_quality(metrics)
            
            return metrics
            
        except Exception as e:
            self.log(f"Failed to calculate metrics for {image_path}: {e}", "ERROR")
            return {}
            
    def _calculate_entropy(self, image):
        """Calculate image entropy (measure of information content)"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        histogram = histogram[histogram > 0]  # Remove zeros
        if len(histogram) <= 1:
            return 0
        probabilities = histogram / histogram.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
        
    def _assess_overall_quality(self, metrics):
        """Assess overall image quality based on metrics"""
        score = 0
        
        # Sharpness (higher is better, typical range 0-1000+)
        if metrics['sharpness'] > 100:
            score += 25
        elif metrics['sharpness'] > 50:
            score += 15
        elif metrics['sharpness'] > 20:
            score += 10
        
        # Contrast (higher is better, typical range 0-100)
        if metrics['contrast'] > 40:
            score += 25
        elif metrics['contrast'] > 20:
            score += 15
        elif metrics['contrast'] > 10:
            score += 10
            
        # Entropy (higher is better, typical range 0-8)
        if metrics['entropy'] > 6:
            score += 25
        elif metrics['entropy'] > 4:
            score += 15
        elif metrics['entropy'] > 2:
            score += 10
            
        # Brightness balance (closer to 128 is better for RGB mean)
        brightness_score = max(0, 25 - abs(metrics['mean_brightness'] - 128) / 5)
        score += brightness_score
        
        return min(100, score)
        
    def test_virtual_tryon(self, person_image, cloth_image, test_name=""):
        """Test virtual try-on with comprehensive logging"""
        self.log(f"🧪 Starting virtual try-on test: {test_name}")
        
        # Start timing
        start_time = time.time()
        
        # Calculate input image metrics
        person_metrics = self.calculate_image_quality_metrics(person_image)
        cloth_metrics = self.calculate_image_quality_metrics(cloth_image)
        
        self.log(f"📊 Input person image quality score: {person_metrics.get('quality_score', 'N/A')}")
        self.log(f"📊 Input cloth image quality score: {cloth_metrics.get('quality_score', 'N/A')}")
        
        # Encode images
        person_b64 = self.encode_image(person_image)
        cloth_b64 = self.encode_image(cloth_image)
        
        if not person_b64 or not cloth_b64:
            self.log("❌ Failed to encode input images", "ERROR")
            return None
            
        # Prepare API request as multipart form data with proper content type
        files = {
            'person_image': (os.path.basename(person_image), open(person_image, 'rb'), 'image/jpeg'),
            'cloth_image': (os.path.basename(cloth_image), open(cloth_image, 'rb'), 'image/jpeg')
        }
        data = {
            'ddim_steps': 30,
            'scale': 1.0,
            'height': 512,
            'width': 384
        }
        
        try:
            self.log("🚀 Sending request to API server...")
            response = requests.post(
                f"{self.api_url}/try-on",
                files=files,
                data=data,
                timeout=120
            )
            
            processing_time = time.time() - start_time
            self.log(f"⏱️ Processing completed in {processing_time:.2f} seconds")
            
        finally:
            # Close file handles
            for f in files.values():
                if hasattr(f, 'close'):
                    f.close()
                elif len(f) > 1:  # It's a tuple (filename, file, content_type)
                    f[1].close()
                
        try:    
            if response.status_code == 200:
                result_data = response.json()
                
                # Save result image
                if 'result_image' in result_data:
                    result_image_path = self.results_dir / f"{test_name}_result.jpg"
                    with open(result_image_path, "wb") as f:
                        f.write(base64.b64decode(result_data['result_image']))
                    
                    # Calculate output image metrics
                    result_metrics = self.calculate_image_quality_metrics(result_image_path)
                    
                    self.log(f"📊 Output image quality score: {result_metrics.get('quality_score', 'N/A')}")
                    
                    # Compile comprehensive test result
                    test_result = {
                        'test_name': test_name,
                        'timestamp': datetime.now().isoformat(),
                        'person_image': str(person_image),
                        'cloth_image': str(cloth_image),
                        'processing_time_seconds': processing_time,
                        'api_response_status': response.status_code,
                        'person_metrics': person_metrics,
                        'cloth_metrics': cloth_metrics,
                        'result_metrics': result_metrics,
                        'result_image_path': str(result_image_path),
                        'quality_improvement': result_metrics.get('quality_score', 0) - person_metrics.get('quality_score', 0),
                        'success': True
                    }
                    
                    # Quality assessment
                    if result_metrics.get('quality_score', 0) > 70:
                        self.log("✅ HIGH QUALITY result generated!", "SUCCESS")
                    elif result_metrics.get('quality_score', 0) > 50:
                        self.log("⚠️ MEDIUM QUALITY result generated", "WARNING")
                    else:
                        self.log("❌ LOW QUALITY result generated", "ERROR")
                    
                    self.log(f"💾 Result saved to: {result_image_path}")
                    
                    return test_result
                    
                else:
                    self.log("❌ No result image in API response", "ERROR")
                    return None
                    
            else:
                self.log(f"❌ API request failed: {response.status_code} - {response.text}", "ERROR")
                return None
                
        except requests.exceptions.Timeout:
            self.log("⏰ API request timeout", "ERROR")
            return None
        except Exception as e:
            self.log(f"❌ API request failed: {e}", "ERROR")
            return None
            
    def run_comprehensive_tests(self):
        """Run comprehensive virtual try-on tests"""
        self.log("🔍 Starting comprehensive virtual try-on quality assessment")
        
        # Check API health first
        if not self.check_api_health():
            self.log("❌ Cannot proceed - API server not healthy", "ERROR")
            return False
            
        # Find test images
        person_images = list(Path("assets/person").glob("*.jpg"))
        cloth_images = list(Path("assets/cloth").glob("*.jpg"))
        
        if not person_images or not cloth_images:
            self.log("❌ No test images found in assets/", "ERROR")
            return False
            
        self.log(f"📂 Found {len(person_images)} person images and {len(cloth_images)} cloth images")
        
        # Run tests with different combinations
        test_combinations = [
            (person_images[0], cloth_images[0], "test_1_primary"),
            (person_images[1] if len(person_images) > 1 else person_images[0], 
             cloth_images[1] if len(cloth_images) > 1 else cloth_images[0], "test_2_secondary")
        ]
        
        successful_tests = 0
        for person_img, cloth_img, test_name in test_combinations:
            result = self.test_virtual_tryon(person_img, cloth_img, test_name)
            if result:
                self.test_results.append(result)
                successful_tests += 1
            
            # Brief pause between tests
            time.sleep(2)
            
        # Generate summary report
        self.generate_summary_report()
        
        self.log(f"🎯 Testing completed: {successful_tests}/{len(test_combinations)} tests successful")
        return successful_tests > 0
        
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.test_results:
            self.log("No test results to summarize", "WARNING")
            return
            
        # Calculate statistics
        quality_scores = [r['result_metrics'].get('quality_score', 0) for r in self.test_results]
        processing_times = [r['processing_time_seconds'] for r in self.test_results]
        
        summary = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'successful_tests': len([r for r in self.test_results if r['success']]),
                'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
                'average_processing_time': np.mean(processing_times) if processing_times else 0,
                'best_quality_score': max(quality_scores) if quality_scores else 0,
                'worst_quality_score': min(quality_scores) if quality_scores else 0,
            },
            'detailed_results': self.test_results,
            'assessment_timestamp': datetime.now().isoformat()
        }
        
        # Save detailed report
        report_file = self.results_dir / QUALITY_METRICS_FILE
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print summary
        self.log("📋 QUALITY ASSESSMENT SUMMARY")
        self.log("=" * 50)
        self.log(f"Total tests: {summary['test_summary']['total_tests']}")
        self.log(f"Successful tests: {summary['test_summary']['successful_tests']}")
        self.log(f"Average quality score: {summary['test_summary']['average_quality_score']:.1f}/100")
        self.log(f"Average processing time: {summary['test_summary']['average_processing_time']:.2f}s")
        self.log(f"Quality range: {summary['test_summary']['worst_quality_score']:.1f} - {summary['test_summary']['best_quality_score']:.1f}")
        self.log(f"📄 Detailed report saved: {report_file}")

def main():
    """Main execution function"""
    print("🔍 MV-VTON Virtual Try-On Quality Assessment Tool")
    print("=" * 60)
    
    tester = VirtualTryOnQualityTester()
    success = tester.run_comprehensive_tests()
    
    if success:
        print("\n✅ Quality assessment completed successfully!")
        print(f"📂 Results available in: {TEST_RESULTS_DIR}/")
    else:
        print("\n❌ Quality assessment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()