#!/usr/bin/env python3
"""
CUDA Installation Guide and Setup Script
Helps install CUDA and GPU-accelerated packages for maximum performance
"""

import os
import subprocess
import sys
import platform
import requests
import json

def check_nvidia_driver():
    """Check if NVIDIA drivers are installed"""
    print("🔍 CHECKING NVIDIA DRIVERS")
    print("=" * 40)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA drivers installed")
            
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'Driver Version' in line:
                    print(f"   {line.strip()}")
                elif '|' in line and 'MiB' in line and 'GeForce' in line:
                    print(f"   GPU: {line.strip()}")
            
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found")
        print("   Please install NVIDIA drivers first!")
        return False

def get_cuda_version():
    """Detect installed CUDA version"""
    print("\n🔍 CHECKING CUDA INSTALLATION")
    print("=" * 40)
    
    cuda_versions = []
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"✅ NVCC: {line.strip()}")
                    # Extract version
                    if 'V' in line:
                        version = line.split('V')[1].split(',')[0]
                        cuda_versions.append(version)
    except FileNotFoundError:
        print("⚠️  nvcc not found")
    
    # Check CUDA runtime
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_ver = line.split('CUDA Version:')[1].strip()
                    print(f"✅ CUDA Runtime: {cuda_ver}")
                    cuda_versions.append(cuda_ver)
    except:
        pass
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch_cuda = torch.version.cuda
            print(f"✅ PyTorch CUDA: {torch_cuda}")
            cuda_versions.append(torch_cuda)
        else:
            print("❌ PyTorch CUDA not available")
    except ImportError:
        print("⚠️  PyTorch not installed")
    
    return cuda_versions

def recommend_cuda_installation():
    """Provide CUDA installation recommendations"""
    print("\n💡 CUDA INSTALLATION RECOMMENDATIONS")
    print("=" * 50)
    
    os_type = platform.system()
    
    if os_type == "Windows":
        print("🪟 WINDOWS INSTALLATION:")
        print("1. Download CUDA Toolkit from:")
        print("   https://developer.nvidia.com/cuda-downloads")
        print("2. Select: Windows > x86_64 > 10/11 > exe (local)")
        print("3. Run the installer as Administrator")
        print("4. Choose 'Express Installation'")
        print("5. Restart your computer after installation")
        print()
        print("📦 Recommended CUDA versions:")
        print("   - CUDA 11.8 (Most compatible)")
        print("   - CUDA 12.1 (Latest, may have compatibility issues)")
        
    elif os_type == "Linux":
        print("🐧 LINUX INSTALLATION:")
        print("1. Update system:")
        print("   sudo apt update && sudo apt upgrade")
        print("2. Install CUDA:")
        print("   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin")
        print("   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600")
        print("   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb")
        print("   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb")
        print("   sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/")
        print("   sudo apt-get update")
        print("   sudo apt-get -y install cuda")
        
    else:
        print("🍎 macOS:")
        print("❌ CUDA not supported on macOS")
        print("   Use CPU-only version: python quick_solution.py")
    
    print("\n🔧 AFTER CUDA INSTALLATION:")
    print("1. Restart your computer")
    print("2. Run: python install_cuda.py --verify")
    print("3. Install PyTorch with CUDA: python install_cuda.py --pytorch")

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("\n🔥 INSTALLING PYTORCH WITH CUDA")
    print("=" * 40)
    
    # Detect CUDA version for PyTorch
    cuda_versions = get_cuda_version()
    
    if not cuda_versions:
        print("❌ No CUDA detected. Installing CPU version...")
        pytorch_cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    else:
        # Use the most common CUDA version
        cuda_ver = cuda_versions[0] if cuda_versions else "11.8"
        
        if "12." in cuda_ver:
            # CUDA 12.x
            print(f"Installing PyTorch for CUDA {cuda_ver}...")
            pytorch_cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
        elif "11.8" in cuda_ver:
            # CUDA 11.8
            print(f"Installing PyTorch for CUDA {cuda_ver}...")
            pytorch_cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            # Default to CUDA 11.8 (most compatible)
            print("Installing PyTorch for CUDA 11.8 (default)...")
            pytorch_cmd = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
    
    try:
        print("Installing PyTorch... (this may take a few minutes)")
        subprocess.check_call(pytorch_cmd)
        print("✅ PyTorch installation completed")
        
        # Verify installation
        try:
            import torch
            print(f"✅ PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"✅ CUDA version: {torch.version.cuda}")
            else:
                print("⚠️  CUDA not available in PyTorch")
        except ImportError:
            print("❌ PyTorch import failed")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")

def install_gpu_ml_packages():
    """Install GPU-accelerated ML packages"""
    print("\n🚀 INSTALLING GPU ML PACKAGES")
    print("=" * 40)
    
    packages = {
        'xgboost': 'XGBoost (GPU support)',
        'lightgbm': 'LightGBM (GPU support)',
        'cupy-cuda11x': 'CuPy (GPU NumPy)',  # Optional
    }
    
    for package, description in packages.items():
        try:
            print(f"Installing {description}...")
            if package == 'cupy-cuda11x':
                # CuPy is optional and may fail
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
                    print(f"✅ {description}")
                except:
                    print(f"⚠️  {description} - optional, skipped")
            else:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ {description}")
        except subprocess.CalledProcessError:
            print(f"❌ {description} - installation failed")
        except subprocess.TimeoutExpired:
            print(f"⚠️  {description} - timeout, may still be installing")

def verify_gpu_setup():
    """Verify complete GPU setup"""
    print("\n✅ VERIFYING GPU SETUP")
    print("=" * 40)
    
    checks = []
    
    # Check NVIDIA driver
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        checks.append(("NVIDIA Driver", True))
    except:
        checks.append(("NVIDIA Driver", False))
    
    # Check CUDA
    try:
        subprocess.run(['nvcc', '--version'], capture_output=True, check=True)
        checks.append(("CUDA Toolkit", True))
    except:
        checks.append(("CUDA Toolkit", False))
    
    # Check PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        checks.append(("PyTorch CUDA", cuda_available))
        if cuda_available:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        checks.append(("PyTorch CUDA", False))
    
    # Check XGBoost
    try:
        import xgboost
        checks.append(("XGBoost", True))
    except ImportError:
        checks.append(("XGBoost", False))
    
    # Check LightGBM
    try:
        import lightgbm
        checks.append(("LightGBM", True))
    except ImportError:
        checks.append(("LightGBM", False))
    
    # Print results
    print("Component Status:")
    all_good = True
    for component, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}")
        if not status:
            all_good = False
    
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 GPU SETUP COMPLETE!")
        print("Ready to run: python hybrid_cpu_gpu_solution.py")
    else:
        print("⚠️  Some components missing")
        print("You can still run: python quick_solution.py")
    
    return all_good

def create_cuda_test_script():
    """Create a simple CUDA test script"""
    test_script = '''#!/usr/bin/env python3
"""
Quick CUDA Test Script
"""

def test_cuda():
    print("🧪 CUDA FUNCTIONALITY TEST")
    print("=" * 30)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Simple GPU computation test
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"✅ GPU computation test passed")
            print(f"   Result shape: {z.shape}")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_cuda()
    if success:
        print("\\n🎉 CUDA is working! Ready for GPU training.")
    else:
        print("\\n⚠️  CUDA issues detected. Use CPU training instead.")
'''
    
    with open('test_cuda.py', 'w') as f:
        f.write(test_script)
    
    print("📄 Created test_cuda.py - run this to test CUDA functionality")

def main():
    """Main CUDA setup function"""
    print("🚀 CUDA INSTALLATION & SETUP GUIDE")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if '--verify' in sys.argv:
            verify_gpu_setup()
            return
        elif '--pytorch' in sys.argv:
            install_pytorch_cuda()
            return
        elif '--packages' in sys.argv:
            install_gpu_ml_packages()
            return
        elif '--test' in sys.argv:
            create_cuda_test_script()
            return
    
    # Full setup process
    print("This script will help you set up CUDA for GPU acceleration.")
    print("Follow the steps below:\n")
    
    # Step 1: Check drivers
    driver_ok = check_nvidia_driver()
    
    if not driver_ok:
        print("\n❌ NVIDIA drivers not found!")
        print("Please install NVIDIA drivers first:")
        print("https://www.nvidia.com/drivers/")
        return
    
    # Step 2: Check CUDA
    cuda_versions = get_cuda_version()
    
    if not cuda_versions:
        print("\n❌ CUDA not found!")
        recommend_cuda_installation()
        print("\nAfter installing CUDA, run:")
        print("python install_cuda.py --pytorch")
        return
    
    # Step 3: Install PyTorch
    print("\n🔥 Installing PyTorch with CUDA...")
    install_pytorch_cuda()
    
    # Step 4: Install ML packages
    print("\n🚀 Installing GPU ML packages...")
    install_gpu_ml_packages()
    
    # Step 5: Verify setup
    print("\n✅ Verifying installation...")
    success = verify_gpu_setup()
    
    # Step 6: Create test script
    create_cuda_test_script()
    
    # Final instructions
    print("\n🎯 NEXT STEPS:")
    if success:
        print("1. Run: python test_cuda.py (verify CUDA works)")
        print("2. Run: python hybrid_cpu_gpu_solution.py (start training)")
    else:
        print("1. Fix any issues shown above")
        print("2. Run: python install_cuda.py --verify")
        print("3. If issues persist, use: python quick_solution.py")

if __name__ == "__main__":
    main()