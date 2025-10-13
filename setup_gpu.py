#!/usr/bin/env python3
"""
GPU Setup Script for Accelerated Training
Installs GPU-optimized packages and checks hardware
"""

import os
import subprocess
import sys
import platform

def check_gpu_availability():
    """Check GPU availability and specifications"""
    print("🔍 GPU HARDWARE DETECTION")
    print("=" * 50)
    
    gpu_info = {}
    
    # Check NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_info[f'gpu_{i}'] = {
                        'name': parts[0],
                        'memory': f"{parts[1]} MB",
                        'driver': parts[2]
                    }
                    print(f"✅ GPU {i}: {parts[0]}")
                    print(f"   Memory: {parts[1]} MB")
                    print(f"   Driver: {parts[2]}")
        else:
            print("❌ NVIDIA GPU not detected or nvidia-smi not available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found - NVIDIA GPU may not be available")
    
    # Check PyTorch GPU support
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            gpu_info['pytorch_cuda'] = True
        else:
            print("⚠️  PyTorch installed but CUDA not available")
            gpu_info['pytorch_cuda'] = False
    except ImportError:
        print("⚠️  PyTorch not installed")
        gpu_info['pytorch_cuda'] = False
    
    return gpu_info

def install_gpu_packages():
    """Install GPU-optimized packages"""
    print("\n📦 INSTALLING GPU-OPTIMIZED PACKAGES")
    print("=" * 50)
    
    # Core packages (always install)
    core_packages = [
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.1.0',
        'tqdm>=4.64.0'
    ]
    
    print("Installing core packages...")
    for package in core_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # GPU-accelerated ML packages
    gpu_packages = []
    
    # XGBoost (supports GPU)
    print("\nInstalling XGBoost...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "xgboost"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ XGBoost (GPU support available)")
        gpu_packages.append('xgboost')
    except:
        print("❌ XGBoost installation failed")
    
    # LightGBM (supports GPU)
    print("Installing LightGBM...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "lightgbm"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ LightGBM (GPU support available)")
        gpu_packages.append('lightgbm')
    except:
        print("❌ LightGBM installation failed")
    
    # PyTorch (GPU version)
    print("Checking PyTorch...")
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch with CUDA already available")
        else:
            print("⚠️  PyTorch available but no CUDA support")
            print("   Consider installing PyTorch with CUDA:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    except ImportError:
        print("Installing PyTorch...")
        try:
            # Try to install PyTorch with CUDA support
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "--index-url", 
                "https://download.pytorch.org/whl/cu118"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ PyTorch with CUDA support")
        except:
            # Fallback to CPU version
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "torch", "torchvision"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("✅ PyTorch (CPU version)")
            except:
                print("❌ PyTorch installation failed")
    
    # Image processing
    print("Installing image processing packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "Pillow"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Pillow")
    except:
        print("❌ Pillow installation failed")
    
    # RAPIDS (advanced GPU acceleration - optional)
    print("\nOptional: RAPIDS cuML/cuDF...")
    print("⚠️  RAPIDS requires conda installation:")
    print("   conda install -c rapidsai -c conda-forge -c nvidia cudf cuml")
    print("   (Skip if you don't have conda)")
    
    return gpu_packages

def create_gpu_config():
    """Create GPU configuration file"""
    config = {
        'gpu_available': False,
        'cuda_available': False,
        'recommended_batch_size': 32,
        'use_mixed_precision': False
    }
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            config['gpu_available'] = True
            config['cuda_available'] = True
            
            # Get GPU memory and recommend batch size
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 8:
                config['recommended_batch_size'] = 64
                config['use_mixed_precision'] = True
            elif gpu_memory >= 4:
                config['recommended_batch_size'] = 32
            else:
                config['recommended_batch_size'] = 16
    except:
        pass
    
    # Save config
    import json
    with open('gpu_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📄 GPU config saved to gpu_config.json")
    print(f"   GPU Available: {config['gpu_available']}")
    print(f"   Recommended batch size: {config['recommended_batch_size']}")

def main():
    print("🚀 GPU-ACCELERATED SETUP")
    print("=" * 60)
    
    # Check system
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check GPU
    gpu_info = check_gpu_availability()
    
    # Install packages
    gpu_packages = install_gpu_packages()
    
    # Create config
    create_gpu_config()
    
    # Final recommendations
    print("\n🎯 SETUP COMPLETE!")
    print("=" * 50)
    
    if gpu_info.get('pytorch_cuda', False):
        print("✅ GPU acceleration ready!")
        print("   Run: python gpu_accelerated_solution.py")
        print("   Expected speedup: 2-5x faster training")
    else:
        print("⚠️  GPU acceleration limited")
        print("   CPU fallback available")
        print("   Run: python quick_solution.py")
    
    print("\n💡 Performance Tips:")
    print("- Close other GPU applications")
    print("- Monitor GPU memory usage")
    print("- Use batch processing for large datasets")
    
    if len(gpu_packages) > 0:
        print(f"\n📦 GPU packages installed: {', '.join(gpu_packages)}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()