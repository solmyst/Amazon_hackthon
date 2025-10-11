#!/usr/bin/env python3
"""
Smart Product Pricing Challenge - ML Solution
Multimodal approach using text and image features for price prediction
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime, timedelta

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Image processing
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Progress tracking
from tqdm import tqdm
tqdm.pandas()

# Utils
from src.utils import download_images
import pickle
import json

# Import progress tracker
try:
    from training_tracker import track_progress, show_final_summary
    TRACKING_ENABLED = True
except ImportError:
    TRACKING_ENABLED = False
    def track_progress(step_name, details=""):
        pass
    def show_final_summary():
        pass

class ProductPricePredictor:
    def __init__(self, dataset_folder='dataset/', image_folder='images/'):
        self.dataset_folder = dataset_folder
        self.image_folder = image_folder
        self.text_vectorizer = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
        # Initialize image model
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.image_model.fc = torch.nn.Identity()  # Remove final layer
        self.image_model.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
    
    def extract_text_features(self, catalog_content):
        """Extract features from catalog content"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(str(catalog_content))
        features['word_count'] = len(str(catalog_content).split())
        
        # Extract numerical values (quantities, sizes, etc.)
        numbers = re.findall(r'\d+\.?\d*', str(catalog_content))
        if numbers:
            features['max_number'] = max([float(x) for x in numbers])
            features['min_number'] = min([float(x) for x in numbers])
            features['avg_number'] = np.mean([float(x) for x in numbers])
            features['number_count'] = len(numbers)
        else:
            features['max_number'] = 0
            features['min_number'] = 0
            features['avg_number'] = 0
            features['number_count'] = 0
        
        # Extract units and quantities
        units = ['oz', 'ounce', 'lb', 'pound', 'kg', 'gram', 'ml', 'liter', 'count', 'pack']
        for unit in units:
            features[f'has_{unit}'] = 1 if unit.lower() in str(catalog_content).lower() else 0
        
        # Brand indicators (common brand patterns)
        brand_indicators = ['brand', 'premium', 'organic', 'natural', 'gourmet']
        for indicator in brand_indicators:
            features[f'has_{indicator}'] = 1 if indicator.lower() in str(catalog_content).lower() else 0
        
        # Product categories (basic classification)
        food_keywords = ['food', 'snack', 'candy', 'chocolate', 'sauce', 'spice', 'oil']
        beauty_keywords = ['cosmetic', 'beauty', 'lip', 'skin', 'hair']
        
        features['is_food'] = 1 if any(kw in str(catalog_content).lower() for kw in food_keywords) else 0
        features['is_beauty'] = 1 if any(kw in str(catalog_content).lower() for kw in beauty_keywords) else 0
        
        return features
    
    def print_time_estimate(self, n_samples):
        """Print estimated training time"""
        # Time estimates per 1000 samples
        text_time_per_1k = 0.5  # minutes
        image_time_per_1k = 2.0  # minutes (including download)
        model_time_per_1k = 0.2  # minutes
        
        estimated_minutes = (n_samples / 1000) * (text_time_per_1k + image_time_per_1k + model_time_per_1k)
        
        print(f"\n⏱️  Estimated Training Time: {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)")
        print(f"📊 Processing {n_samples:,} samples")
        print(f"🔄 You can monitor progress with the bars below...")
        print("-" * 60)
    
    def preprocess_text(self, text):
        """Clean and preprocess text for TF-IDF"""
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def extract_image_features(self, image_path):
        """Extract features from product image using ResNet50"""
        try:
            if not os.path.exists(image_path):
                return np.zeros(2048)  # Return zero vector if image not found
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.image_model(image_tensor)
            
            return features.numpy().flatten()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros(2048)
    
    def prepare_features(self, df, is_training=True):
        """Prepare all features for training/prediction"""
        start_time = time.time()
        
        print("🔤 Extracting text features...")
        # Extract text features with progress bar
        text_features_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Text features"):
            text_features = self.extract_text_features(row['catalog_content'])
            text_features_list.append(text_features)
        
        text_features_df = pd.DataFrame(text_features_list)
        print(f"✓ Text features extracted: {text_features_df.shape[1]} features")
        
        # TF-IDF features
        print("📝 Processing TF-IDF features...")
        processed_texts = [self.preprocess_text(text) for text in tqdm(df['catalog_content'], desc="Text preprocessing")]
        
        if is_training:
            self.text_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            tfidf_features = self.text_vectorizer.fit_transform(processed_texts)
        else:
            if self.text_vectorizer is None:
                raise ValueError("Text vectorizer not fitted. Train the model first.")
            tfidf_features = self.text_vectorizer.transform(processed_texts)
        
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        print(f"✓ TF-IDF features: {tfidf_df.shape[1]} features")
        
        # Download images if needed
        if not os.path.exists(self.image_folder):
            print("📥 Downloading images...")
            download_images(df['image_link'].tolist(), self.image_folder)
        else:
            print("✓ Images folder exists, checking for missing images...")
            # Check for missing images
            missing_count = 0
            for link in df['image_link']:
                image_filename = Path(link).name
                image_path = os.path.join(self.image_folder, image_filename)
                if not os.path.exists(image_path):
                    missing_count += 1
            
            if missing_count > 0:
                print(f"📥 Downloading {missing_count} missing images...")
                download_images(df['image_link'].tolist(), self.image_folder)
        
        # Extract image features
        print("🖼️  Extracting image features...")
        image_features_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Image features"):
            image_filename = Path(row['image_link']).name
            image_path = os.path.join(self.image_folder, image_filename)
            image_features = self.extract_image_features(image_path)
            image_features_list.append(image_features)
        
        image_features_df = pd.DataFrame(
            image_features_list,
            columns=[f'img_{i}' for i in range(len(image_features_list[0]))]
        )
        print(f"✓ Image features: {image_features_df.shape[1]} features")
        
        # Combine all features
        print("🔗 Combining features...")
        combined_features = pd.concat([
            text_features_df.reset_index(drop=True),
            tfidf_df.reset_index(drop=True),
            image_features_df.reset_index(drop=True)
        ], axis=1)
        
        if is_training:
            self.feature_names = combined_features.columns.tolist()
        
        elapsed_time = time.time() - start_time
        print(f"✓ Feature extraction completed in {elapsed_time/60:.1f} minutes")
        print(f"📊 Total features: {combined_features.shape[1]}")
        
        return combined_features
    
    def train(self, train_df):
        """Train the ensemble model"""
        start_time = time.time()
        
        # Print time estimate
        self.print_time_estimate(len(train_df))
        
        print("🔧 Preparing training features...")
        X = self.prepare_features(train_df, is_training=True)
        y = train_df['price'].values
        
        print("📏 Scaling features...")
        scaling_start = time.time()
        X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Scaling completed in {time.time() - scaling_start:.1f} seconds")
        
        print("🤖 Training models...")
        training_start = time.time()
        
        # Random Forest
        print("  🌳 Training Random Forest...")
        rf_start = time.time()
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        self.models['rf'].fit(X_scaled, y)
        print(f"    ✓ Random Forest trained in {time.time() - rf_start:.1f} seconds")
        
        # XGBoost
        print("  🚀 Training XGBoost...")
        xgb_start = time.time()
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.models['xgb'].fit(X_scaled, y)
        print(f"    ✓ XGBoost trained in {time.time() - xgb_start:.1f} seconds")
        
        # Gradient Boosting
        print("  📈 Training Gradient Boosting...")
        gb_start = time.time()
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        self.models['gb'].fit(X_scaled, y)
        print(f"    ✓ Gradient Boosting trained in {time.time() - gb_start:.1f} seconds")
        
        # Ridge Regression
        print("  📊 Training Ridge Regression...")
        ridge_start = time.time()
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_scaled, y)
        print(f"    ✓ Ridge trained in {time.time() - ridge_start:.1f} seconds")
        
        total_training_time = time.time() - training_start
        print(f"✓ All models trained in {total_training_time:.1f} seconds")
        
        # Cross-validation scores
        print("📊 Cross-validation scores:")
        for name, model in self.models.items():
            cv_start = time.time()
            scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_absolute_error')
            cv_time = time.time() - cv_start
            print(f"  {name.upper()}: MAE = {-scores.mean():.2f} (±{scores.std() * 2:.2f}) [{cv_time:.1f}s]")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
        print(f"⏰ Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
        print(f"⏰ Finished: {datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}")
    
    def predict(self, test_df):
        """Make predictions using ensemble"""
        print("Preparing test features...")
        X = self.prepare_features(test_df, is_training=False)
        X_scaled = self.scaler.transform(X)
        
        print("Making predictions...")
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[name] = pred
        
        # Ensemble prediction (weighted average)
        weights = {'rf': 0.3, 'xgb': 0.3, 'gb': 0.25, 'ridge': 0.15}
        
        ensemble_pred = np.zeros(len(test_df))
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]
        
        # Ensure positive prices
        ensemble_pred = np.maximum(ensemble_pred, 0.1)
        
        return ensemble_pred
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'text_vectorizer': self.text_vectorizer,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.text_vectorizer = model_data['text_vectorizer']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")

def main():
    """Main training and prediction pipeline"""
    overall_start = time.time()
    print("🚀 Smart Product Pricing Challenge - Training Pipeline")
    print("=" * 60)
    
    predictor = ProductPricePredictor()
    
    # Load data
    print("📂 Loading datasets...")
    load_start = time.time()
    train_df = pd.read_csv(os.path.join(predictor.dataset_folder, 'train.csv'))
    test_df = pd.read_csv(os.path.join(predictor.dataset_folder, 'test.csv'))
    load_time = time.time() - load_start
    
    print(f"✓ Data loaded in {load_time:.1f} seconds")
    print(f"📊 Training samples: {len(train_df):,}")
    print(f"📊 Test samples: {len(test_df):,}")
    
    track_progress("Data Loading", f"{len(train_df):,} train + {len(test_df):,} test samples")
    
    # Train model
    predictor.train(train_df)
    
    # Save model
    print("💾 Saving trained model...")
    save_start = time.time()
    predictor.save_model('trained_model.pkl')
    print(f"✓ Model saved in {time.time() - save_start:.1f} seconds")
    
    track_progress("Model Saving", "Ensemble model saved")
    
    # Make predictions
    print("🔮 Making predictions on test set...")
    pred_start = time.time()
    predictions = predictor.predict(test_df)
    pred_time = time.time() - pred_start
    print(f"✓ Predictions completed in {pred_time/60:.1f} minutes")
    
    track_progress("Predictions", f"{len(predictions):,} predictions completed")
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save predictions
    output_path = os.path.join(predictor.dataset_folder, 'test_out.csv')
    output_df.to_csv(output_path, index=False)
    
    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Predictions saved to: {output_path}")
    print(f"📊 Total predictions: {len(output_df):,}")
    print(f"💰 Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"💰 Mean price: ${predictions.mean():.2f}")
    print(f"⏱️  Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"🕐 Started: {datetime.fromtimestamp(overall_start).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Show final summary if tracking is enabled
    if TRACKING_ENABLED:
        show_final_summary()

if __name__ == "__main__":
    main()