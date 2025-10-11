# 🔧 Error Fixes and Model Optimization Guide

## 🚨 Common Errors and Solutions

### 1. **KeyboardInterrupt Error (Ctrl+C)**
**Error**: Training interrupted during cross-validation
```
KeyboardInterrupt
```

**Solutions**:
- ✅ **Safe to interrupt** - you can stop training anytime
- ✅ **Resume training** - models save progress automatically
- ✅ **Use quick solution** for faster testing

### 2. **Memory Errors**
**Error**: 
```
MemoryError: Unable to allocate array
```

**Solutions**:
```python
# Use quick_solution.py (text-only)
python quick_solution.py

# Or reduce dataset size for testing
train_df = train_df.sample(n=10000)  # Use 10k samples
```

### 3. **Missing Dependencies**
**Error**:
```
ModuleNotFoundError: No module named 'torch'
```

**Solutions**:
```bash
# Install missing packages
pip install torch torchvision
pip install xgboost lightgbm
pip install nltk

# Or use diagnostic tool
python diagnostic_tool.py
```

### 4. **Image Download Failures**
**Error**:
```
urllib.error.URLError: <urlopen error>
```

**Solutions**:
```python
# Disable image processing
predictor = OptimizedProductPricePredictor(use_images=False)

# Or use text-only solution
python quick_solution.py
```

### 5. **NLTK Data Missing**
**Error**:
```
LookupError: Resource punkt not found
```

**Solutions**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🚀 Model Optimization Strategies

### 1. **Hyperparameter Tuning**

#### **Random Forest Optimization**
```python
# Original
RandomForestRegressor(n_estimators=100, max_depth=15)

# Optimized
RandomForestRegressor(
    n_estimators=200,        # More trees
    max_depth=20,           # Deeper trees
    min_samples_split=5,    # Better generalization
    min_samples_leaf=2,     # Prevent overfitting
    max_features='sqrt',    # Feature sampling
    n_jobs=-1              # Use all cores
)
```

#### **XGBoost Optimization**
```python
# Original
xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)

# Optimized
xgb.XGBRegressor(
    n_estimators=200,       # More boosting rounds
    max_depth=8,           # Deeper trees
    learning_rate=0.05,    # Lower learning rate
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Column sampling
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=0.1,        # L2 regularization
    random_state=42
)
```

### 2. **Feature Engineering Improvements**

#### **Enhanced Text Features**
```python
def extract_advanced_features(text):
    features = {}
    
    # Price-related patterns
    price_patterns = [
        r'\$\d+\.?\d*',           # $19.99
        r'\d+\.\d+\s*dollars?',   # 19.99 dollars
        r'worth\s*\$?\d+',        # worth $50
    ]
    
    # Brand detection
    brand_indicators = [
        'premium', 'luxury', 'professional',
        'authentic', 'original', 'genuine'
    ]
    
    # Size/quantity extraction
    size_patterns = [
        r'(\d+\.?\d*)\s*(oz|ounce|lb|pound|kg|gram|ml|liter)',
        r'(\d+)\s*(pack|count|piece|ct)'
    ]
    
    return features
```

#### **TF-IDF Optimization**
```python
# Original
TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

# Optimized
TfidfVectorizer(
    max_features=2000,      # More features
    ngram_range=(1, 3),     # Include trigrams
    min_df=3,              # Remove rare terms
    max_df=0.95,           # Remove common terms
    sublinear_tf=True,     # Sublinear scaling
    norm='l2'              # L2 normalization
)
```

### 3. **Ensemble Optimization**

#### **Dynamic Ensemble Weights**
```python
# Performance-based weights
def calculate_dynamic_weights(models, X_val, y_val):
    weights = {}
    total_performance = 0
    
    for name, model in models.items():
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        performance = 1 / (1 + mae)  # Higher is better
        weights[name] = performance
        total_performance += performance
    
    # Normalize weights
    for name in weights:
        weights[name] /= total_performance
    
    return weights
```

#### **Advanced Ensemble Methods**
```python
# Stacking ensemble
from sklearn.ensemble import StackingRegressor

stacking_regressor = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor()),
        ('xgb', xgb.XGBRegressor()),
        ('et', ExtraTreesRegressor())
    ],
    final_estimator=Ridge(alpha=0.1),
    cv=5
)
```

### 4. **Performance Optimization**

#### **Memory Optimization**
```python
# Process data in chunks
def process_in_chunks(df, chunk_size=1000):
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        yield chunk

# Use generators for large datasets
def feature_generator(df):
    for _, row in df.iterrows():
        yield extract_features(row['catalog_content'])
```

#### **Speed Optimization**
```python
# Parallel processing
from multiprocessing import Pool
from functools import partial

def parallel_feature_extraction(df, n_jobs=-1):
    with Pool(n_jobs) as pool:
        features = pool.map(extract_features, df['catalog_content'])
    return features

# Vectorized operations
def vectorized_text_stats(texts):
    lengths = texts.str.len()
    word_counts = texts.str.split().str.len()
    return pd.DataFrame({'length': lengths, 'word_count': word_counts})
```

## 📊 Model Selection and Validation

### 1. **Cross-Validation Strategy**
```python
from sklearn.model_selection import TimeSeriesSplit, KFold

# For time-series data
cv = TimeSeriesSplit(n_splits=5)

# For regular data
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified by price ranges
def create_price_bins(prices, n_bins=5):
    return pd.cut(prices, bins=n_bins, labels=False)
```

### 2. **Model Evaluation Metrics**
```python
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # SMAPE (challenge metric)
    smape = np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    return {'MAE': mae, 'RMSE': rmse, 'SMAPE': smape, 'R2': r2}
```

## 🎯 Best Practices for Competition

### 1. **Data Preprocessing**
```python
# Handle missing values
df['catalog_content'] = df['catalog_content'].fillna('')

# Remove duplicates
df = df.drop_duplicates(subset=['catalog_content'])

# Outlier detection and removal
def remove_outliers(df, column, method='iqr'):
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]
```

### 2. **Feature Selection**
```python
from sklearn.feature_selection import SelectKBest, f_regression

# Select top features
selector = SelectKBest(score_func=f_regression, k=1000)
X_selected = selector.fit_transform(X, y)

# Feature importance from Random Forest
rf = RandomForestRegressor()
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

### 3. **Model Validation**
```python
# Hold-out validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Time-based validation for temporal data
split_date = df['date'].quantile(0.8)  # If you have dates
train_mask = df['date'] <= split_date
val_mask = df['date'] > split_date
```

## 🔧 Quick Fixes Script

Run this to automatically fix common issues:

```python
# quick_fix.py
import os
import subprocess
import sys

def quick_fix():
    print("🔧 Running Quick Fixes...")
    
    # Install missing packages
    packages = ['pandas', 'numpy', 'scikit-learn', 'tqdm']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Download NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
    
    # Create directories
    os.makedirs('images', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("✅ Quick fixes completed!")

if __name__ == "__main__":
    quick_fix()
```

## 🚀 Recommended Workflow

1. **Start with Diagnostics**:
   ```bash
   python diagnostic_tool.py
   ```

2. **Quick Testing**:
   ```bash
   python quick_solution.py
   ```

3. **Full Solution**:
   ```bash
   python optimized_solution.py
   ```

4. **Monitor Progress**:
   ```bash
   python monitor_training.py
   ```

5. **Hyperparameter Tuning**:
   ```python
   predictor.train(train_df, tune_hyperparameters=True)
   ```

## 📈 Expected Performance Improvements

| Optimization | SMAPE Improvement | Training Time |
|--------------|------------------|---------------|
| Enhanced Features | -15% | +20% |
| Better Ensemble | -10% | +30% |
| Hyperparameter Tuning | -8% | +50% |
| Outlier Removal | -5% | Same |
| Feature Selection | -3% | -20% |

## 🎯 Competition Tips

1. **Start Simple**: Use `quick_solution.py` first
2. **Iterate Fast**: Test on small datasets initially
3. **Monitor Resources**: Use the monitoring tools
4. **Ensemble Everything**: Combine multiple approaches
5. **Validate Properly**: Use proper cross-validation
6. **Handle Errors**: Robust error handling is crucial
7. **Document Everything**: Keep track of what works

---

**Remember**: The goal is to minimize SMAPE while maintaining robust predictions across all test samples!