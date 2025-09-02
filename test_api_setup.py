#!/usr/bin/env python3
"""
Test script to verify MV-VTON API setup without starting the server
"""

import os
import sys

def test_file_existence():
    """Test that required files exist"""
    print("🔍 Testing file existence...")
    
    required_files = {
        "Frontal-View VTON/configs/viton512.yaml": "Config file",
        "Frontal-View VTON/checkpoint/vitonhd.ckpt": "High-quality Frontal-View VTON checkpoint", 
        "models/vgg/vgg19_conv.pth": "VGG model"
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✅ {description}: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {description}: {file_path} (MISSING)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_imports():
    """Test that required Python modules can be imported"""
    print("\n🐍 Testing Python imports...")
    
    required_modules = [
        "torch", "torchvision", "PIL", "numpy", 
        "fastapi", "uvicorn", "omegaconf", "cv2", "pytorch_lightning"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} (MISSING)")
            missing_modules.append(module)
    
    return len(missing_modules) == 0

def test_mvvton_imports():
    """Test MV-VTON specific imports"""
    print("\n🤖 Testing MV-VTON imports...")
    
    try:
        # Add current directory to path
        sys.path.append(os.getcwd())
        
        from ldm.util import instantiate_from_config
        print("  ✅ ldm.util.instantiate_from_config")
        
        from ldm.models.diffusion.ddim import DDIMSampler
        print("  ✅ ldm.models.diffusion.ddim.DDIMSampler")
        
        from omegaconf import OmegaConf
        print("  ✅ omegaconf.OmegaConf")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ MV-VTON import failed: {e}")
        return False

def test_config_loading():
    """Test loading the config file"""
    print("\n⚙️  Testing config loading...")
    
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load("configs/viton512.yaml")
        
        print(f"  ✅ Config loaded successfully")
        print(f"  📄 Model target: {config.model.target}")
        print(f"  📐 Image size: {config.model.params.get('image_size', 'Not specified')}")
        print(f"  🎯 Channels: {config.model.params.get('channels', 'Not specified')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 MV-VTON API Setup Test")
    print("=" * 50)
    
    tests = [
        ("File Existence", test_file_existence),
        ("Python Imports", test_imports),
        ("MV-VTON Imports", test_mvvton_imports),
        ("Config Loading", test_config_loading)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! API server should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the requirements.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)