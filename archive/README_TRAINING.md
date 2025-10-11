# Training Progress Monitoring Guide

## ⏱️ Expected Training Times

### For 75K samples (full dataset):
- **Text Processing**: 5-10 minutes
- **Image Download**: 15-30 minutes (depends on internet speed)
- **Image Feature Extraction**: 10-20 minutes
- **Model Training**: 5-15 minutes
- **Predictions**: 2-5 minutes

**Total: 35-75 minutes (0.6-1.25 hours)**

### For smaller datasets:
- **Sample data (100 samples)**: 2-5 minutes
- **10K samples**: 8-15 minutes
- **25K samples**: 15-30 minutes

## 📊 How to Monitor Training Progress

### Option 1: Built-in Progress Bars (Recommended)
The solution includes built-in progress tracking:

```bash
python solution.py
```

You'll see:
- 🔤 Text feature extraction progress
- 📥 Image download status
- 🖼️ Image processing progress
- 🤖 Model training stages
- ⏱️ Time estimates and completion status

### Option 2: External Monitor (Advanced)
Run this in a **separate terminal** while training:

```bash
python monitor_training.py
```

This shows:
- Real-time progress updates
- System resource usage (CPU, Memory)
- File creation status
- ETA calculations
- **Does NOT interfere with training**

### Option 3: Log File Monitoring
Check the automatically created `progress.log` file:

```bash
# Windows
type progress.log

# Linux/Mac
tail -f progress.log
```

## 🚀 Quick Start Commands

### 1. Fast Testing (Text-only, 5-10 minutes)
```bash
python quick_solution.py
```

### 2. Full Solution (Text + Images, 35-75 minutes)
```bash
python solution.py
```

### 3. With External Monitoring
```bash
# Terminal 1: Start training
python solution.py

# Terminal 2: Monitor progress
python monitor_training.py
```

## 📈 Progress Indicators

### What you'll see during training:

1. **Data Loading** (1-2 minutes)
   ```
   📂 Loading datasets...
   ✓ Data loaded in 1.2 seconds
   📊 Training samples: 75,000
   ```

2. **Feature Extraction** (20-40 minutes)
   ```
   🔤 Extracting text features...
   Text features: 100%|████████| 75000/75000 [02:15<00:00, 553.21it/s]
   📥 Downloading images...
   🖼️ Extracting image features...
   Image features: 100%|████████| 75000/75000 [15:30<00:00, 80.65it/s]
   ```

3. **Model Training** (5-15 minutes)
   ```
   🤖 Training models...
     🌳 Training Random Forest...
     ✓ Random Forest trained in 45.2 seconds
     🚀 Training XGBoost...
     ✓ XGBoost trained in 23.1 seconds
   ```

4. **Predictions** (2-5 minutes)
   ```
   🔮 Making predictions on test set...
   ✓ Predictions completed in 3.2 minutes
   ```

## 🛑 How to Stop Training Safely

- **Ctrl+C**: Stops training (you can resume later)
- **Close terminal**: Stops training
- **Monitor script**: Ctrl+C stops monitoring only, training continues

## 📁 Files Created During Training

- `images/`: Downloaded product images
- `trained_model.pkl`: Saved model (created after training)
- `test_out.csv`: Final predictions (created at the end)
- `progress.log`: Training progress log
- Temporary files during processing

## 🔧 Troubleshooting

### Training seems stuck?
- Check `monitor_training.py` output
- Look at `progress.log` file
- Check system resources (CPU/Memory usage)

### Out of memory?
- Use `quick_solution.py` (text-only)
- Reduce batch sizes in the code
- Close other applications

### Slow image download?
- Check internet connection
- Some images may be slow/unavailable (this is normal)
- The process will continue with available images

### Want to resume training?
- If model file exists, you can skip to predictions
- Modify the code to load existing model

## 💡 Tips for Faster Training

1. **Use SSD storage** for faster file I/O
2. **Close unnecessary applications** to free RAM
3. **Use text-only solution first** to test quickly
4. **Run during off-peak hours** for better internet speed
5. **Monitor system resources** to avoid bottlenecks

## 🎯 Expected Performance

After training completes, you should see:
- SMAPE score < 30% (lower is better)
- Price predictions in reasonable range ($0.10 - $500)
- All 75,000 test samples predicted
- Model file size: 50-200 MB

Happy training! 🚀