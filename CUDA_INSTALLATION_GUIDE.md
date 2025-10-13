# 🚀 CUDA Installation Guide for GPU Acceleration

## 🎯 Quick Setup (Recommended)

### **Step 1: Check Your Setup**
```bash
python setup_gpu.py
```
This will tell you exactly what's missing.

### **Step 2: Install Missing Components**
```bash
python install_cuda.py
```
This will guide you through CUDA installation.

### **Step 3: Verify Everything Works**
```bash
python install_cuda.py --verify
python test_cuda.py
```

---

## 📋 Manual Installation Steps

### **🪟 Windows Installation**

#### **1. Install NVIDIA Drivers**
- Go to: https://www.nvidia.com/drivers/
- Download and install the latest driver for your GPU
- Restart your computer

#### **2. Install CUDA Toolkit**
- Go to: https://developer.nvidia.com/cuda-downloads
- Select: **Windows > x86_64 > 10/11 > exe (local)**
- Download the installer (~3GB)
- Run as Administrator
- Choose **"Express Installation"**
- Restart your computer

#### **3. Install PyTorch with CUDA**
```bash
# For CUDA 11.8 (most compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **4. Install GPU ML Packages**
```bash
pip install xgboost lightgbm
```

### **🐧 Linux Installation**

#### **1. Install NVIDIA Drivers**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-525  # or latest version
sudo reboot
```

#### **2. Install CUDA Toolkit**
```bash
# Download CUDA 11.8 (recommended)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### **3. Install PyTorch and ML packages**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost lightgbm
```

---

## 🔍 Verification Steps

### **Check NVIDIA Driver**
```bash
nvidia-smi
```
Should show your GPU information.

### **Check CUDA**
```bash
nvcc --version
```
Should show CUDA compiler version.

### **Check PyTorch CUDA**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU name
```

---

## 🚀 Performance Comparison

| Setup | Training Time | Speedup |
|-------|---------------|---------|
| **CPU Only** | 40 minutes | 1x |
| **GPU + CPU Hybrid** | 12-20 minutes | **2-3x faster** |
| **Pure GPU** | 8-15 minutes | **3-5x faster** |

---

## 🔧 Troubleshooting

### **Common Issues:**

#### **"CUDA out of memory"**
- Reduce batch size in the code
- Close other GPU applications
- Use mixed precision training

#### **"CUDA driver version is insufficient"**
- Update NVIDIA drivers
- Match CUDA version with driver

#### **"torch.cuda.is_available() returns False"**
- Reinstall PyTorch with correct CUDA version
- Check CUDA installation

#### **"nvcc not found"**
- Add CUDA to PATH
- Reinstall CUDA Toolkit

### **Quick Fixes:**

```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check GPU memory usage
nvidia-smi

# Test CUDA functionality
python test_cuda.py
```

---

## 💡 Optimization Tips

### **For Better Performance:**

1. **Close unnecessary applications** that use GPU
2. **Use appropriate batch sizes** based on GPU memory
3. **Enable mixed precision** if GPU supports it
4. **Monitor GPU utilization** with `nvidia-smi`

### **GPU Memory Guidelines:**

| GPU Memory | Recommended Batch Size | Mixed Precision |
|------------|------------------------|-----------------|
| 4GB | 16-32 | No |
| 6GB | 32-64 | Yes |
| 8GB+ | 64+ | Yes |

---

## 🎯 Ready to Train!

Once CUDA is installed and verified:

### **🚀 Best Performance (GPU + CPU)**
```bash
python hybrid_cpu_gpu_solution.py
```

### **🔥 GPU-Focused**
```bash
python gpu_accelerated_solution.py
```

### **💻 CPU Fallback**
```bash
python quick_solution.py
```

---

## 📞 Need Help?

1. **Run diagnostics**: `python setup_gpu.py`
2. **Check CUDA**: `python install_cuda.py --verify`
3. **Test functionality**: `python test_cuda.py`

**Remember**: If GPU setup fails, the CPU-only solution still works great and will give you competitive results!