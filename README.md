# 🎯 Amazon ML Challenge 2025 - Fast Training Solution

Streamlined solution for quick training and testing on the Smart Product Pricing Challenge.

## 🚀 Quick Start Options

### ⚡ **GPU Accelerated (Recommended if you have GPU)**
```bash
# 1. Check GPU Setup
python setup_gpu.py

# 2. Install CUDA (if needed)
python install_cuda.py

# 3. Hybrid CPU+GPU Training (5-10 minutes)
python hybrid_cpu_gpu_solution.py
```

### 🔥 **CPU Fast Training**
```bash
# 1. Setup (30 seconds)
python setup_fast.py

# 2. Fast Training (2-3 minutes)
python train_10k.py

# 3. Full Training (15 minutes)
python quick_solution.py
```

## 📁 Optimized File Structure

```
├── hybrid_cpu_gpu_solution.py    # 🚀 GPU+CPU accelerated (BEST)
├── gpu_accelerated_solution.py   # 🔥 Pure GPU solution
├── quick_solution.py             # 💻 CPU-only full training
├── train_10k.py                  # ⚡ Fast 10K testing
├── install_cuda.py               # 🔧 CUDA installation guide
├── setup_gpu.py                  # 🔧 GPU setup & detection
├── setup_fast.py                 # 🔧 CPU setup
├── CUDA_INSTALLATION_GUIDE.md    # 📚 Detailed CUDA guide
└── archive/                      # Advanced solutions
```

## 🔥 GPU Requirements

### **For GPU Acceleration:**
- **NVIDIA GPU** with 2GB+ VRAM
- **CUDA Toolkit** (11.8 or 12.1)
- **Updated NVIDIA Drivers**

### **Quick CUDA Check:**
```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA version
```

### **Don't have CUDA?**
No problem! Run `python install_cuda.py` for step-by-step installation.

## ⚡ Fast Training Features

### 🎯 **10K Training Script** (`train_10k.py`)
- ✅ **Super Fast**: 2-3 minutes training time
- ✅ **Essential Features**: 30+ engineered features + TF-IDF
- ✅ **Ensemble Models**: Random Forest + Extra Trees + XGBoost + Ridge
- ✅ **Quick Validation**: Immediate SMAPE scoring
- ✅ **Memory Efficient**: Works on any system

### 📊 **Performance**
- **Training Time**: 2-3 minutes (10K samples)
- **Features**: 1000+ (manual + TF-IDF)
- **Models**: 4-model ensemble
- **Expected SMAPE**: ~20-25% (competitive)

## 🔧 What's Included

### Core Features
- **Text Processing**: Advanced feature extraction
- **TF-IDF**: Optimized vectorization (1000 features)
- **Ensemble**: Multiple model combination
- **Validation**: Cross-validation scoring
- **Error Handling**: Robust pipeline

### Models Used
1. **Random Forest** (100 trees)
2. **Extra Trees** (80 trees) 
3. **XGBoost** (if available)
4. **Ridge Regression** (baseline)

## 📈 Training Workflow

### Step 1: Fast Test (10K samples)
```bash
python train_10k.py
```
**Output**: 
- `fast_model_10k.pkl` (trained model)
- `dataset/test_out_10k.csv` (predictions)
- Training SMAPE score

### Step 2: Full Training (75K samples)
```bash
python quick_solution.py
```
**Output**:
- `dataset/test_out.csv` (final predictions)
- Better performance on full dataset

## 🎯 Expected Results

### 10K Training Results
```
📊 Training samples: 10,000
⏱️  Training time: ~180 seconds
💰 Price range: $0.50 - $450.00
📊 Training SMAPE: ~22%
```

### Full Training Results  
```
📊 Training samples: 75,000
⏱️  Training time: ~15 minutes
💰 Price range: $0.10 - $500.00
📊 Expected SMAPE: ~18-22%
```

## 🔍 Troubleshooting

### Quick Fixes
```bash
# Check setup
python diagnostic_tool.py

# Install missing packages
pip install pandas numpy scikit-learn tqdm

# XGBoost (optional but recommended)
pip install xgboost
```

### Common Issues
- **No dataset folder**: Create `dataset/` and add `train.csv`, `test.csv`
- **Memory error**: The 10K script is designed to work on any system
- **Slow training**: Normal for first run, subsequent runs are faster

## 📊 File Sizes & Requirements

### Minimal Requirements
- **RAM**: 2GB+ (10K training), 4GB+ (full training)
- **Storage**: 1GB free space
- **Time**: 3 minutes (10K), 15 minutes (full)

### Generated Files
- **Model file**: ~5-20 MB
- **Predictions**: ~2-5 MB
- **Logs**: Minimal

## 🎉 Next Steps

1. **Start with 10K**: `python train_10k.py`
2. **Validate results**: Check SMAPE score
3. **Scale to full**: `python quick_solution.py`
4. **Advanced features**: Check `archive/` folder

## 🏆 Competition Ready

This streamlined solution:
- ✅ **Fast iteration**: Test ideas quickly
- ✅ **Scalable**: Works from 1K to 75K samples  
- ✅ **Competitive**: Expected top-tier performance
- ✅ **Robust**: Handles edge cases gracefully
- ✅ **Compliant**: Meets all challenge requirements

---

**Start training in under 1 minute! 🚀**