#!/usr/bin/env python3
"""
Streamlined Training Script - First 10K Samples
Fast training for immediate testing and validation
"""

import os
import re
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

# Optional imports with fallbacks
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from tqdm import tqdm
import pickle

class FastTrainer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.scaler = RobustScaler()
        self.feature_names = []
        
    def extract_features(self, text):
        """Fast feature extraction"""
        features = {}
        text = str(text).lower()
        
        # Basic stats
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['digit_count'] = sum(c.isdigit() for c in text)
        
        # Numbers
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            nums = [float(x) for x in numbers if float(x) > 0]
            if nums:
                features['max_number'] = max(nums)
                features['avg_number'] = np.mean(nums)
                features['number_count'] = len(nums)
            else:
                features.update({'max_number': 0, 'avg_number': 0, 'number_count': 0})
        else:
            features.update({'max_number': 0, 'avg_number': 0, 'number_count': 0})
        
        # Units
        units = ['oz', 'ounce', 'lb', 'pound', 'count', 'pack', 'ml', 'liter']
        for unit in units:
            features[f'has_{unit}'] = 1 if unit in text else 0
        
        # Quality
        quality = ['premium', 'organic', 'natural', 'gourmet', 'fresh']
        for q in quality:
            features[f'has_{q}'] = 1 if q in text else 0
        
        # Categories
        food_words = ['food', 'snack', 'candy', 'chocolate', 'sauce', 'spice']
        beauty_words = ['cosmetic', 'beauty', 'lip', 'lotion', 'cream']
        
        features['is_food'] = 1 if any(w in text for w in food_words) else 0
        features['is_beauty'] = 1 if any(w in text for w in beauty_words) else 0
        
        return features
    
    def preprocess_text(self, text):
        """Clean text for TF-IDF"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return ' '.join(text.split())
    
    def prepare_features(self, df):
        """Prepare all features"""
        print(f"🔤 Extracting features from {len(df)} samples...")
        
        # Extract manual features
        features_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Manual features"):
            features = self.extract_features(row['catalog_content'])
            features_list.append(features)
        
        manual_df = pd.DataFrame(features_list)
        print(f"✓ Manual features: {manual_df.shape[1]}")
        
        # TF-IDF features
        print("📝 Processing TF-IDF...")
        texts = [self.preprocess_text(text) for text in df['catalog_content']]
        
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.vectorizer.transform(texts)
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        print(f"✓ TF-IDF features: {tfidf_df.shape[1]}")
        
        # Combine features
        combined = pd.concat([manual_df, tfidf_df], axis=1)
        combined = combined.fillna(0)
        
        print(f"✓ Total features: {combined.shape[1]}")
        return combined
    
    def train(self, train_df):
        """Train models on data"""
        start_time = time.time()
        
        print("🚀 Fast Training Pipeline")
        print("=" * 40)
        print(f"📊 Training samples: {len(train_df):,}")
        
        # Prepare features
        X = self.prepare_features(train_df)
        y = train_df['price'].values
        
        # Remove extreme outliers
        Q1, Q3 = np.percentile(y, [25, 75])
        IQR = Q3 - Q1
        mask = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
        X_clean, y_clean = X[mask], y[mask]
        
        print(f"Removed {len(y) - len(y_clean)} outliers")
        
        # Scale features
        print("📏 Scaling features...")
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train models
        print("🤖 Training models...")
        
        # Random Forest
        print("  🌳 Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['rf'].fit(X_scaled, y_clean)
        
        # Extra Trees
        print("  🌲 Extra Trees...")
        self.models['et'] = ExtraTreesRegressor(
            n_estimators=80, max_depth=20, random_state=42, n_jobs=-1
        )
        self.models['et'].fit(X_scaled, y_clean)
        
        # XGBoost if available
        if XGB_AVAILABLE:
            print("  🚀 XGBoost...")
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, 
                random_state=42, verbosity=0
            )
            self.models['xgb'].fit(X_scaled, y_clean)
        
        # Ridge
        print("  📊 Ridge...")
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_scaled, y_clean)
        
        # Quick validation
        print("📊 Validation scores:")
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y_clean, cv=3, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
            print(f"  {name.upper()}: MAE = {-scores.mean():.2f}")
        
        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.1f} seconds")
        
        return X_scaled, y_clean
    
    def predict(self, test_df):
        """Make predictions"""
        print("🔮 Making predictions...")
        X = self.prepare_features(test_df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[name] = pred
        
        # Ensemble with equal weights
        ensemble = np.mean(list(predictions.values()), axis=0)
        ensemble = np.maximum(ensemble, 0.1)  # Ensure positive
        
        print(f"✓ Predictions: ${ensemble.min():.2f} - ${ensemble.max():.2f}")
        return ensemble
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")

def main():
    """Main training pipeline"""
    start_time = time.time()
    
    print("🎯 FAST 10K TRAINING PIPELINE")
    print("=" * 50)
    
    # Load data
    print("📂 Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    # Use first 10K samples for training
    train_sample = train_df.head(10000).copy()
    test_sample = test_df.head(1000).copy()  # Small test for quick validation
    
    print(f"📊 Using {len(train_sample):,} training samples")
    print(f"📊 Using {len(test_sample):,} test samples")
    
    # Initialize trainer
    trainer = FastTrainer()
    
    # Train
    X_train, y_train = trainer.train(train_sample)
    
    # Save model
    trainer.save_model('fast_model_10k.pkl')
    
    # Test predictions
    predictions = trainer.predict(test_sample)
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_sample['sample_id'],
        'price': predictions
    })
    
    # Save results
    output_df.to_csv('dataset/test_out_10k.csv', index=False)
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("🎉 FAST TRAINING COMPLETED!")
    print("=" * 50)
    print(f"📁 Model saved: fast_model_10k.pkl")
    print(f"📁 Predictions: dataset/test_out_10k.csv")
    print(f"📊 Predictions: {len(predictions):,}")
    print(f"💰 Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"💰 Mean price: ${predictions.mean():.2f}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print("=" * 50)
    
    # Calculate SMAPE on a small validation set
    val_sample = train_sample.head(1000)
    val_pred = trainer.predict(val_sample)
    val_actual = val_sample['price'].values
    smape = np.mean(np.abs(val_pred - val_actual) / ((np.abs(val_actual) + np.abs(val_pred)) / 2)) * 100
    print(f"📊 Validation SMAPE: {smape:.2f}%")
    
    return trainer, output_df

if __name__ == "__main__":
    trainer, results = main()
    print("\n🎯 Ready for full dataset training!")
    print("Next: Modify the script to use full dataset when ready.")