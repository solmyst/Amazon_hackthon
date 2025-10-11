# 🏆 Amazon ML Challenge 2025 - Smart Product Pricing

A comprehensive multimodal machine learning solution for predicting e-commerce product prices using text descriptions and product images.

## 🎯 Challenge Overview

This solution addresses the **Smart Product Pricing Challenge** where we predict product prices based on:
- **Text Data**: Product titles, descriptions, and specifications
- **Image Data**: Product images from URLs
- **Target**: Accurate price prediction with SMAPE evaluation

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### 2. Run Solutions

#### Fast Testing (Text-only, 5-10 minutes)
```bash
python quick_solution.py
```

#### Full Solution (Text + Images, 35-75 minutes)
```bash
python solution.py
```

#### With Progress Monitoring
```bash
# Terminal 1: Start training
python solution.py

# Terminal 2: Monitor progress
python monitor_training.py
```

## 📊 Solution Architecture

### Multimodal Ensemble Approach
```
Product Data (Text + Images)
         ↓
┌─────────────────┬─────────────────┐
│   Text Branch   │  Image Branch   │
│                 │                 │
│ • Feature Eng.  │ • ResNet50      │
│ • TF-IDF        │ • 2048 features │
│ • 25+ features  │ • Pre-trained   │
└─────────────────┴─────────────────┘
         ↓
    Feature Scaling
         ↓
┌──────┬──────┬──────┬──────┐
│  RF  │ XGB  │  GB  │Ridge │
│ 30%  │ 30%  │ 25%  │ 15%  │
└──────┴──────┴──────┴──────┘
         ↓
   Weighted Ensemble
         ↓
   Price Prediction
```

### Key Components

#### 🔤 Text Processing
- **Feature Engineering**: Quantities, units, categories, quality indicators
- **TF-IDF Vectorization**: 1000 features, 1-2 grams
- **Domain Features**: Food/beauty classification, brand detection

#### 🖼️ Image Processing
- **Model**: ResNet50 (pre-trained on ImageNet)
- **Features**: 2048-dimensional vectors
- **Preprocessing**: Resize, normalize, tensor conversion

#### 🤖 Ensemble Models
- **Random Forest**: 100 trees, max_depth=15 (30% weight)
- **XGBoost**: Gradient boosting, 100 estimators (30% weight)
- **Gradient Boosting**: Scikit-learn implementation (25% weight)
- **Ridge Regression**: Linear baseline with L2 regularization (15% weight)

## 📁 File Structure

```
├── solution.py              # Complete multimodal solution
├── quick_solution.py        # Fast text-only version
├── experiment.ipynb         # Data exploration notebook
├── monitor_training.py      # External progress monitor
├── training_tracker.py      # Progress tracking utilities
├── setup.py                # Environment setup script
├── requirements.txt         # Python dependencies
├── solution_documentation.md # Detailed methodology
├── README_TRAINING.md       # Training guide
└── src/
    ├── utils.py            # Image download utilities
    └── example.ipynb       # Basic usage examples
```

## ⏱️ Performance & Timing

### Expected Training Times
- **Sample data (100 items)**: 2-5 minutes
- **10K samples**: 8-15 minutes
- **Full dataset (75K)**: 35-75 minutes

### Model Performance
- **Target**: SMAPE < 25%
- **Features**: 3000+ combined features
- **Model Size**: ~50-200 MB
- **Inference**: Fast (seconds for 75K predictions)

## 🔧 Features & Innovation

### Smart Feature Engineering
- **Quantity Extraction**: Automatic detection of weights, counts, volumes
- **Unit Recognition**: oz, lb, count, pack, fl oz, etc.
- **Quality Indicators**: Premium, organic, gourmet, natural
- **Category Classification**: Food, beauty, cleaning products
- **Brand Detection**: Brand name and quality tier identification

### Robust Pipeline
- **Error Handling**: Graceful handling of missing/corrupted images
- **Scalability**: Efficient processing of large datasets
- **Monitoring**: Real-time progress tracking and ETA
- **Resumability**: Can resume from checkpoints

### Compliance
- ✅ **No External Data**: Uses only provided training data
- ✅ **Model Limits**: ResNet50 < 8B parameters, MIT licensed
- ✅ **Output Format**: Exact CSV format matching requirements
- ✅ **Positive Prices**: All predictions are positive floats

## 📈 Monitoring & Progress

### Built-in Progress Tracking
- Real-time progress bars for each stage
- Time estimates and completion status
- System resource monitoring
- Automatic logging to `progress.log`

### External Monitoring
Run `python monitor_training.py` in a separate terminal for:
- Live progress updates without interrupting training
- System resource usage (CPU, Memory)
- File creation status and ETA calculations

## 🛠️ Customization

### Adjust Model Weights
```python
# In solution.py, modify ensemble weights
weights = {'rf': 0.3, 'xgb': 0.3, 'gb': 0.25, 'ridge': 0.15}
```

### Feature Engineering
- Add new text patterns in `extract_text_features()`
- Modify TF-IDF parameters in `prepare_features()`
- Experiment with different image models

### Hyperparameter Tuning
- Adjust model parameters in the `train()` method
- Modify cross-validation strategy
- Change feature selection criteria

## 📊 Results Format

Output file: `dataset/test_out.csv`
```csv
sample_id,price
217392,62.08
209156,17.19
262333,96.50
...
```

## 🔍 Troubleshooting

### Common Issues
- **Memory Error**: Use `quick_solution.py` or reduce batch sizes
- **Slow Training**: Check internet connection for image downloads
- **Missing Images**: Normal - pipeline handles gracefully
- **Interrupted Training**: Safe to stop with Ctrl+C, can resume

### Performance Tips
- Use SSD storage for faster I/O
- Close unnecessary applications
- Run during off-peak hours for better download speeds
- Monitor system resources with the monitoring script

## 📚 Documentation

- **[Training Guide](README_TRAINING.md)**: Detailed training instructions
- **[Solution Documentation](solution_documentation.md)**: Complete methodology
- **[Experiment Notebook](experiment.ipynb)**: Data exploration and analysis

## 🏅 Challenge Compliance

This solution fully complies with all challenge requirements:
- Uses only provided training data (no external price lookup)
- Employs MIT/Apache licensed models under 8B parameters
- Outputs positive float prices in exact required format
- Includes comprehensive documentation and methodology

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Place dataset files** in `dataset/` folder
4. **Run setup**: `python setup.py`
5. **Start training**: `python solution.py`
6. **Monitor progress**: `python monitor_training.py` (optional)

## 📞 Support

For questions or issues:
- Check the training guide: `README_TRAINING.md`
- Review the troubleshooting section above
- Examine the experiment notebook for data insights

---

**Good luck with the Amazon ML Challenge! 🎯**