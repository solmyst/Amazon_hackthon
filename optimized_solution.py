#!/usr/bin/env python3
"""
Optimized Smart Product Pricing Challenge Solution
Enhanced with error handling, hyperparameter tuning, and better performance
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nltk

# Image processing (optional - fallback if torch not available)
try:
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Running in text-only mode.")
    TORCH_AVAILABLE = False

# ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Using alternative models.")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available. Using alternative models.")
    LGB_AVAILABLE = False

# Progress tracking
from tqdm import tqdm
tqdm.pandas()

# Utils
try:
    from src.utils import download_images
except ImportError:
    logger.warning("Utils not found. Image download may not work.")
    def download_images(links, folder):
        pass

import pickle
import json

class OptimizedProductPricePredictor:
    def __init__(self, dataset_folder='dataset/', image_folder='images/', use_images=True):
        self.dataset_folder = dataset_folder
        self.image_folder = image_folder
        self.use_images = use_images and TORCH_AVAILABLE
        self.text_vectorizer = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.models = {}
        self.feature_names = []
        self.best_params = {}
        
        # Initialize image model if available
        if self.use_images:
            try:
                self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.image_model.fc = torch.nn.Identity()
                self.image_model.eval()
                
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                logger.info("✓ Image processing enabled with ResNet50")
            except Exception as e:
                logger.warning(f"Failed to initialize image model: {e}")
                self.use_images = False
        
        # Download NLTK data safely
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK data with error handling"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(nltk.corpus.stopwords.words('english'))
            except Exception as e:
                logger.warning(f"NLTK setup failed: {e}. Using basic stopwords.")
                self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def extract_enhanced_text_features(self, catalog_content):
        """Enhanced feature extraction with better error handling"""
        features = {}
        text = str(catalog_content).lower()
        
        try:
            # Basic text statistics
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            features['char_count'] = len([c for c in text if c.isalpha()])
            features['digit_count'] = len([c for c in text if c.isdigit()])
            features['upper_count'] = len([c for c in str(catalog_content) if c.isupper()])
            
            # Extract numerical values with better regex
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                nums = [float(x) for x in numbers if float(x) > 0]  # Filter out zeros
                if nums:
                    features['max_number'] = max(nums)
                    features['min_number'] = min(nums)
                    features['avg_number'] = np.mean(nums)
                    features['std_number'] = np.std(nums) if len(nums) > 1 else 0
                    features['number_count'] = len(nums)
                    features['number_range'] = max(nums) - min(nums) if len(nums) > 1 else 0
                else:
                    features.update({f'{k}': 0 for k in ['max_number', 'min_number', 'avg_number', 'std_number', 'number_count', 'number_range']})
            else:
                features.update({f'{k}': 0 for k in ['max_number', 'min_number', 'avg_number', 'std_number', 'number_count', 'number_range']})
            
            # Enhanced unit detection
            units = {
                'weight': ['oz', 'ounce', 'lb', 'pound', 'kg', 'gram', 'g'],
                'volume': ['ml', 'liter', 'l', 'fl oz', 'gallon', 'quart'],
                'count': ['count', 'pack', 'piece', 'ct', 'pcs'],
                'size': ['inch', 'cm', 'mm', 'ft', 'meter']
            }
            
            for unit_type, unit_list in units.items():
                features[f'has_{unit_type}'] = 1 if any(unit in text for unit in unit_list) else 0
            
            # Enhanced quality indicators
            quality_indicators = {
                'premium': ['premium', 'gourmet', 'luxury', 'deluxe', 'professional'],
                'natural': ['organic', 'natural', 'pure', 'fresh', 'raw'],
                'brand': ['brand', 'authentic', 'original', 'genuine'],
                'quality': ['quality', 'grade', 'finest', 'best', 'top']
            }
            
            for quality_type, quality_list in quality_indicators.items():
                features[f'has_{quality_type}'] = 1 if any(word in text for word in quality_list) else 0
            
            # Enhanced category detection
            categories = {
                'food': ['food', 'snack', 'candy', 'chocolate', 'sauce', 'spice', 'oil', 'drink', 'beverage'],
                'beauty': ['cosmetic', 'beauty', 'lip', 'skin', 'hair', 'makeup', 'lotion', 'cream'],
                'cleaning': ['cleaning', 'detergent', 'soap', 'wash', 'cleaner'],
                'health': ['vitamin', 'supplement', 'medicine', 'health', 'wellness'],
                'electronics': ['electronic', 'digital', 'battery', 'charger', 'device']
            }
            
            for category, keywords in categories.items():
                features[f'is_{category}'] = 1 if any(word in text for word in keywords) else 0
            
            # Text complexity features
            sentences = text.split('.')
            features['sentence_count'] = len([s for s in sentences if len(s.strip()) > 0])
            features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            features['unique_word_ratio'] = len(set(text.split())) / len(text.split()) if text.split() else 0
            
            # Price-related keywords
            price_keywords = ['cheap', 'expensive', 'affordable', 'budget', 'value', 'deal', 'sale', 'discount']
            features['has_price_keywords'] = 1 if any(word in text for word in price_keywords) else 0
            
        except Exception as e:
            logger.warning(f"Error in feature extraction: {e}")
            # Return default features if extraction fails
            default_features = {f'feature_{i}': 0 for i in range(50)}
            return default_features
        
        return features
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        try:
            text = str(text).lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            # Remove special characters but keep important ones
            text = re.sub(r'[^\w\s\-\.]', ' ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove very short words (likely noise)
            text = ' '.join([word for word in text.split() if len(word) > 2])
            return text
        except Exception as e:
            logger.warning(f"Text preprocessing error: {e}")
            return str(text)
    
    def extract_image_features(self, image_path):
        """Enhanced image feature extraction with error handling"""
        if not self.use_images:
            return np.zeros(2048)
        
        try:
            if not os.path.exists(image_path):
                return np.zeros(2048)
            
            # Check file size (skip very large files)
            if os.path.getsize(image_path) > 10 * 1024 * 1024:  # 10MB limit
                logger.warning(f"Image too large: {image_path}")
                return np.zeros(2048)
            
            image = Image.open(image_path).convert('RGB')
            
            # Check image dimensions
            if image.size[0] < 32 or image.size[1] < 32:
                logger.warning(f"Image too small: {image_path}")
                return np.zeros(2048)
            
            image_tensor = self.image_transform(image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.image_model(image_tensor)
            
            return features.numpy().flatten()
            
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            return np.zeros(2048)
    
    def prepare_features(self, df, is_training=True):
        """Enhanced feature preparation with better error handling"""
        start_time = time.time()
        
        logger.info("🔤 Extracting enhanced text features...")
        
        # Extract text features with error handling
        text_features_list = []
        failed_extractions = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Text features"):
            try:
                text_features = self.extract_enhanced_text_features(row['catalog_content'])
                text_features_list.append(text_features)
            except Exception as e:
                logger.warning(f"Failed to extract features for row {idx}: {e}")
                failed_extractions += 1
                # Add default features
                text_features_list.append({f'feature_{i}': 0 for i in range(50)})
        
        if failed_extractions > 0:
            logger.warning(f"Failed to extract features for {failed_extractions} samples")
        
        text_features_df = pd.DataFrame(text_features_list)
        logger.info(f"✓ Text features extracted: {text_features_df.shape[1]} features")
        
        # Enhanced TF-IDF with better parameters
        logger.info("📝 Processing enhanced TF-IDF features...")
        processed_texts = []
        for text in tqdm(df['catalog_content'], desc="Text preprocessing"):
            processed_texts.append(self.preprocess_text(text))
        
        if is_training:
            self.text_vectorizer = TfidfVectorizer(
                max_features=2000,  # Increased for better representation
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams
                min_df=3,
                max_df=0.95,  # Remove very common terms
                sublinear_tf=True,  # Use sublinear scaling
                norm='l2'
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
        logger.info(f"✓ TF-IDF features: {tfidf_df.shape[1]} features")
        
        # Image processing with better error handling
        image_features_df = None
        if self.use_images:
            logger.info("🖼️  Processing images...")
            
            # Download images if needed
            if not os.path.exists(self.image_folder):
                os.makedirs(self.image_folder)
                logger.info("📥 Downloading images...")
                try:
                    download_images(df['image_link'].tolist(), self.image_folder)
                except Exception as e:
                    logger.warning(f"Image download failed: {e}")
            
            # Extract image features
            image_features_list = []
            failed_images = 0
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Image features"):
                try:
                    image_filename = Path(row['image_link']).name
                    image_path = os.path.join(self.image_folder, image_filename)
                    image_features = self.extract_image_features(image_path)
                    image_features_list.append(image_features)
                except Exception as e:
                    failed_images += 1
                    image_features_list.append(np.zeros(2048))
            
            if failed_images > 0:
                logger.warning(f"Failed to process {failed_images} images")
            
            image_features_df = pd.DataFrame(
                image_features_list,
                columns=[f'img_{i}' for i in range(2048)]
            )
            logger.info(f"✓ Image features: {image_features_df.shape[1]} features")
        
        # Combine features
        logger.info("🔗 Combining features...")
        feature_dfs = [text_features_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)]
        
        if image_features_df is not None:
            feature_dfs.append(image_features_df.reset_index(drop=True))
        
        combined_features = pd.concat(feature_dfs, axis=1)
        
        # Handle any remaining NaN values
        combined_features = combined_features.fillna(0)
        
        if is_training:
            self.feature_names = combined_features.columns.tolist()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Feature extraction completed in {elapsed_time/60:.1f} minutes")
        logger.info(f"📊 Total features: {combined_features.shape[1]}")
        
        return combined_features
    
    def get_optimized_models(self):
        """Get models with optimized hyperparameters"""
        models = {}
        
        # Random Forest with tuned parameters
        models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees for diversity
        models['et'] = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost if available
        if XGB_AVAILABLE:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0
            )
        
        # LightGBM if available
        if LGB_AVAILABLE:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=-1
            )
        
        # Gradient Boosting
        models['gb'] = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # ElasticNet for regularization
        models['en'] = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=42
        )
        
        return models
    
    def train(self, train_df, tune_hyperparameters=False):
        """Enhanced training with optional hyperparameter tuning"""
        start_time = time.time()
        
        logger.info("🔧 Preparing training features...")
        X = self.prepare_features(train_df, is_training=True)
        y = train_df['price'].values
        
        # Remove outliers (optional)
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        X_clean = X[outlier_mask]
        y_clean = y[outlier_mask]
        
        logger.info(f"Removed {len(y) - len(y_clean)} outliers ({(len(y) - len(y_clean))/len(y)*100:.1f}%)")
        
        logger.info("📏 Scaling features...")
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Get optimized models
        self.models = self.get_optimized_models()
        
        logger.info("🤖 Training optimized models...")
        
        # Train each model with progress tracking
        for name, model in self.models.items():
            model_start = time.time()
            logger.info(f"  Training {name.upper()}...")
            
            try:
                model.fit(X_scaled, y_clean)
                training_time = time.time() - model_start
                logger.info(f"    ✓ {name.upper()} trained in {training_time:.1f} seconds")
                
                # Quick validation
                cv_scores = cross_val_score(model, X_scaled, y_clean, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
                logger.info(f"    CV MAE: {-cv_scores.mean():.2f} (±{cv_scores.std() * 2:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                del self.models[name]
        
        # Hyperparameter tuning (optional)
        if tune_hyperparameters and len(self.models) > 0:
            logger.info("🔧 Hyperparameter tuning...")
            self._tune_hyperparameters(X_scaled, y_clean)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {total_time/60:.1f} minutes!")
        
        # Memory cleanup
        gc.collect()
    
    def _tune_hyperparameters(self, X, y):
        """Hyperparameter tuning for key models"""
        # Only tune the most important models to save time
        tuning_params = {
            'rf': {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5]
            },
            'xgb': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1]
            } if XGB_AVAILABLE else {}
        }
        
        for model_name, params in tuning_params.items():
            if model_name in self.models and params:
                logger.info(f"  Tuning {model_name.upper()}...")
                try:
                    grid_search = GridSearchCV(
                        self.models[model_name],
                        params,
                        cv=3,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X, y)
                    self.models[model_name] = grid_search.best_estimator_
                    self.best_params[model_name] = grid_search.best_params_
                    logger.info(f"    Best params: {grid_search.best_params_}")
                except Exception as e:
                    logger.warning(f"Tuning failed for {model_name}: {e}")
    
    def predict(self, test_df):
        """Enhanced prediction with better ensemble"""
        logger.info("🔮 Making predictions...")
        X = self.prepare_features(test_df, is_training=False)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[name] = pred
                logger.info(f"  {name.upper()}: predictions range ${pred.min():.2f} - ${pred.max():.2f}")
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Dynamic ensemble weights based on model performance
        if len(predictions) >= 4:
            weights = {'rf': 0.25, 'et': 0.2, 'xgb': 0.25, 'lgb': 0.15, 'gb': 0.1, 'en': 0.05}
        elif len(predictions) >= 3:
            weights = {'rf': 0.35, 'et': 0.25, 'xgb': 0.25, 'gb': 0.15}
        else:
            # Equal weights if few models
            weight_value = 1.0 / len(predictions)
            weights = {name: weight_value for name in predictions.keys()}
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(test_df))
        total_weight = 0
        
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
                total_weight += weights[name]
        
        # Normalize weights
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Ensure positive prices with reasonable bounds
        ensemble_pred = np.clip(ensemble_pred, 0.1, 10000)
        
        logger.info(f"✓ Final predictions: ${ensemble_pred.min():.2f} - ${ensemble_pred.max():.2f}")
        logger.info(f"✓ Mean price: ${ensemble_pred.mean():.2f}")
        
        return ensemble_pred
    
    def save_model(self, filepath):
        """Enhanced model saving"""
        model_data = {
            'models': self.models,
            'text_vectorizer': self.text_vectorizer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'use_images': self.use_images
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"✓ Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, filepath):
        """Enhanced model loading"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.text_vectorizer = model_data['text_vectorizer']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.best_params = model_data.get('best_params', {})
            self.use_images = model_data.get('use_images', True)
            
            logger.info(f"✓ Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

def main():
    """Optimized main pipeline"""
    start_time = time.time()
    
    logger.info("🚀 Optimized Smart Product Pricing Challenge")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = OptimizedProductPricePredictor(use_images=TORCH_AVAILABLE)
    
    try:
        # Load data
        logger.info("📂 Loading datasets...")
        train_df = pd.read_csv(os.path.join(predictor.dataset_folder, 'train.csv'))
        test_df = pd.read_csv(os.path.join(predictor.dataset_folder, 'test.csv'))
        
        logger.info(f"📊 Training samples: {len(train_df):,}")
        logger.info(f"📊 Test samples: {len(test_df):,}")
        
        # Train model
        predictor.train(train_df, tune_hyperparameters=False)  # Set to True for tuning
        
        # Save model
        predictor.save_model('optimized_model.pkl')
        
        # Make predictions
        predictions = predictor.predict(test_df)
        
        # Create output
        output_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        # Save predictions
        output_path = os.path.join(predictor.dataset_folder, 'test_out.csv')
        output_df.to_csv(output_path, index=False)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("🎉 OPTIMIZED PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📁 Predictions saved to: {output_path}")
        logger.info(f"📊 Total predictions: {len(output_df):,}")
        logger.info(f"💰 Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        logger.info(f"💰 Mean price: ${predictions.mean():.2f}")
        logger.info(f"⏱️  Total runtime: {total_time/60:.1f} minutes")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()