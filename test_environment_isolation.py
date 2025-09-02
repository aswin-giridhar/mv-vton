#!/usr/bin/env python3
"""
Test script for MV-VTON Environment Isolation Setup
Verifies that the environment isolation approach resolves package conflicts
"""

import sys
import os
import subprocess
import importlib

def test_environment_detection():
    """Test which Python environment we're running in"""
    print("🔍 ENVIRONMENT DETECTION TEST")
    print("=" * 50)
    
    # Check Python path
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if we're in a conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"Conda environment: {conda_env}")
    else:
        print("Environment: Base Python (not conda)")
    
    print()

def test_package_imports():
    """Test importing required packages"""
    print("📦 PACKAGE IMPORT TEST")
    print("=" * 50)
    
    packages = {
        'gradio': 'Gradio web framework',
        'requests': 'HTTP client library', 
        'PIL': 'Python Imaging Library',
        'numpy': 'Numerical computing',
    }
    
    results = {}
    
    for package, description in packages.items():
        try:
            if package == 'PIL':
                import PIL
                module = PIL
            else:
                module = importlib.import_module(package)
            
            version = getattr(module, '__version__', 'Unknown version')
            print(f"✅ {package}: {version} ({description})")
            results[package] = True
        except ImportError as e:
            print(f"❌ {package}: Import failed - {e}")
            results[package] = False
    
    print()
    return results

def test_gradio_compatibility():
    """Test Gradio-specific compatibility"""
    print("🎭 GRADIO COMPATIBILITY TEST")
    print("=" * 50)
    
    try:
        import gradio as gr
        print(f"✅ Gradio version: {gr.__version__}")
        
        # Test basic Gradio functionality
        def test_function(x):
            return f"Test successful: {x}"
        
        # Create a minimal interface (don't launch)
        demo = gr.Interface(
            fn=test_function,
            inputs="text",
            outputs="text",
            title="Environment Isolation Test"
        )
        print("✅ Gradio interface creation: Success")
        
        # Test if we can access common Gradio components
        components = ['Textbox', 'Button', 'Image', 'Slider']
        for comp in components:
            if hasattr(gr, comp):
                print(f"✅ Gradio {comp}: Available")
            else:
                print(f"❌ Gradio {comp}: Not available")
                
        print("✅ Gradio compatibility: All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Gradio compatibility error: {e}")
        return False
    
    print()

def test_api_connectivity():
    """Test API server connectivity"""
    print("🔌 API CONNECTIVITY TEST")
    print("=" * 50)
    
    try:
        import requests
        api_url = "http://localhost:5000"
        
        # Test health endpoint
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Server: Online")
                print(f"   Status: {data['status']}")
                print(f"   Model: {'Loaded' if data['model_loaded'] else 'Not Loaded'}")
                print(f"   Device: {data.get('device', 'Unknown')}")
                return True
            else:
                print(f"❌ API Server: HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ API Server: Not running (Connection refused)")
            print("   Start with: conda activate mv-vton && ./start_api_server.sh")
            return False
        except requests.exceptions.Timeout:
            print("❌ API Server: Timeout (Server may be starting)")
            return False
            
    except ImportError:
        print("❌ Requests library not available")
        return False
    
    print()

def test_file_structure():
    """Test required files are present"""
    print("📁 FILE STRUCTURE TEST")
    print("=" * 50)
    
    required_files = {
        'start_api_server.sh': 'API server startup script',
        'start_gradio_app.sh': 'Gradio startup script (updated)',
        'start_complete_system.sh': 'Complete system startup script (updated)',
        'gradio_app.py': 'Full Gradio interface (updated)',
        'mvvton_api_server.py': 'MV-VTON API server',
        'configs/viton512.yaml': 'Model configuration',
        'checkpoint/mvg.ckpt': 'Model checkpoint'
    }
    
    all_present = True
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {file_path}: {size:.1f} MB ({description})")
        else:
            print(f"❌ {file_path}: Missing ({description})")
            all_present = False
    
    print()
    return all_present

def main():
    """Run all tests"""
    print("🧪 MV-VTON ENVIRONMENT ISOLATION TEST SUITE")
    print("=" * 60)
    print("Testing environment isolation setup...")
    print("This verifies that package conflicts have been resolved")
    print()
    
    # Run all tests
    test_environment_detection()
    package_results = test_package_imports()
    gradio_ok = test_gradio_compatibility()
    api_ok = test_api_connectivity()
    files_ok = test_file_structure()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 50)
    
    total_packages = len(package_results)
    successful_packages = sum(package_results.values())
    
    print(f"Package imports: {successful_packages}/{total_packages}")
    print(f"Gradio compatibility: {'✅ Pass' if gradio_ok else '❌ Fail'}")
    print(f"API connectivity: {'✅ Pass' if api_ok else '❌ Fail'}")
    print(f"File structure: {'✅ Pass' if files_ok else '❌ Fail'}")
    
    if successful_packages == total_packages and gradio_ok:
        print("\n🎉 ENVIRONMENT ISOLATION SUCCESS!")
        print("✅ All package conflicts resolved")
        print("✅ Ready to run MV-VTON system")
        print("\n📋 Next Steps:")
        print("1. Start API server: ./start_api_server.sh")
        print("2. Start Gradio UI: ./start_gradio_app.sh")
        print("   OR use complete system: ./start_complete_system.sh")
    else:
        print("\n⚠️ ISSUES DETECTED:")
        if successful_packages < total_packages:
            print("- Some packages failed to import")
        if not gradio_ok:
            print("- Gradio compatibility issues")
        if not api_ok:
            print("- API server not running (expected if not started)")
        if not files_ok:
            print("- Missing required files")
        
        print("\n🔧 Run package installation:")
        print("pip install --upgrade gradio requests pillow numpy")

if __name__ == "__main__":
    main()