#!/usr/bin/env python3
"""
Hybrid CPU+GPU Solution for Smart Product Pricing
Intelligently distributes workload between CPU and GPU for optimal performance
"""

import os
import re
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import logging
from tqdm import tqdm
import gc
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

# GPU imports with fallbacks
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        DEVICE = torch.device('cuda')
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)} ({GPU_MEMORY:.1f}GB)")
    else:
        DEVICE = torch.device('cpu')
        GPU_MEMORY = 0
        logger.info("⚠️  GPU not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = 'cpu'
    GPU_MEMORY = 0
    logger.warning("⚠️  PyTorch not available")

# XGBoost with GPU support
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# LightGBM with GPU support
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

import pickle

class HybridCPUGPUPredictor:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.scaler = RobustScaler()
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        self.cpu_cores = cpu_count()
        
        # Load or create configuration
        self.config = self.load_config()
        
        # Initialize image model if available
        if TORCH_AVAILABLE:
            self.setup_image_model()
        
        logger.info(f"🔧 Hybrid Setup: {self.cpu_cores} CPU cores, GPU: {self.gpu_available}")
    
    def load_config(self):
        """Load GPU configuration"""
        default_config = {
            'batch_size': 32 if GPU_AVAILABLE else 16,
            'use_mixed_precision': GPU_AVAILABLE and GPU_MEMORY >= 6,
            'parallel_workers': min(self.cpu_cores, 8),
            'gpu_models': ['xgb', 'lgb'] if GPU_AVAILABLE else [],
            'cpu_models': ['rf', 'et', 'ridge']
        }
        
        try:
            with open('gpu_config.json', 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            return default_config
    
    def setup_image_model(self):
        """Setup image processing model"""
        try:
            self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.image_model.fc = torch.nn.Identity()
            self.image_model = self.image_model.to(self.device)
            self.image_model.eval()
            
            # Enable mixed precision if available
            if self.config['use_mixed_precision']:
                self.image_model = self.image_model.half()
            
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info(f"✅ Image model on {self.device}")
            self.use_images = True
        except Exception as e:
            logger.warning(f"Image model setup failed: {e}")
            self.use_images = False
    
    def extract_features_parallel(self, texts):
        """Parallel feature extraction using CPU cores"""
        def extract_single_feature(text):
            features = {}
            try:
                text = str(text).lower()
                
                # Basic stats
                features['text_length'] = len(text)
                features['word_count'] = len(text.split())
                features['char_count'] = sum(c.isalpha() for c in text)
                features['digit_count'] = sum(c.isdigit() for c in text)
                
                # Numbers
                numbers = re.findall(r'\d+\.?\d*', text)
                if numbers:
                    nums = [float(x) for x in numbers if float(x) > 0]
                    if nums:
                        features.update({
                            'max_number': max(nums),
                            'min_number': min(nums),
                            'avg_number': np.mean(nums),
                            'std_number': np.std(nums) if len(nums) > 1 else 0,
                            'number_count': len(nums)
                        })
                    else:
                        features.update({k: 0 for k in ['max_number', 'min_number', 'avg_number', 'std_number', 'number_count']})
                else:
                    features.update({k: 0 for k in ['max_number', 'min_number', 'avg_number', 'std_number', 'number_count']})
                
                # Units and categories (optimized)
                unit_patterns = {
                    'weight': r'\b(?:oz|ounce|lb|pound|kg|gram|g)\b',
                    'volume': r'\b(?:ml|liter|l|fl oz|gallon)\b',
                    'count': r'\b(?:count|pack|piece|ct|pcs)\b'
                }
                
                for unit_type, pattern in unit_patterns.items():
                    features[f'has_{unit_type}'] = 1 if re.search(pattern, text) else 0
                
                # Quality indicators
                quality_words = ['premium', 'organic', 'natural', 'gourmet', 'fresh', 'luxury']
                features['quality_score'] = sum(1 for word in quality_words if word in text)
                
                # Categories
                categories = {
                    'food': ['food', 'snack', 'candy', 'chocolate', 'sauce'],
                    'beauty': ['cosmetic', 'beauty', 'lip', 'lotion', 'cream'],
                    'health': ['vitamin', 'supplement', 'medicine']
                }
                
                for category, keywords in categories.items():
                    features[f'is_{category}'] = 1 if any(word in text for word in keywords) else 0
                
                # Text complexity
                words = text.split()
                if words:
                    features['avg_word_length'] = np.mean([len(word) for word in words])
                    features['unique_word_ratio'] = len(set(words)) / len(words)
                else:
                    features['avg_word_length'] = 0
                    features['unique_word_ratio'] = 0
                
            except Exception as e:
                # Return default features on error
                features = {f'feature_{i}': 0 for i in range(20)}
            
            return features
        
        # Use ProcessPoolExecutor for CPU-intensive feature extraction
        with ProcessPoolExecutor(max_workers=self.config['parallel_workers']) as executor:
            results = list(tqdm(
                executor.map(extract_single_feature, texts),
                total=len(texts),
                desc="CPU feature extraction"
            ))
        
        return pd.DataFrame(results)
    
    def extract_images_gpu_batch(self, image_paths):
        """GPU batch processing for images"""
        if not self.use_images:
            return np.zeros((len(image_paths), 2048))
        
        batch_size = self.config['batch_size']
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="GPU image processing"):
            batch_paths = image_paths[i:i+batch_size]
            batch_features = []
            
            # Load batch of images
            batch_tensors = []
            for path in batch_paths:
                try:
                    if os.path.exists(path):
                        image = Image.open(path).convert('RGB')
                        tensor = self.image_transform(image)
                        batch_tensors.append(tensor)
                    else:
                        # Placeholder for missing image
                        batch_tensors.append(torch.zeros(3, 224, 224))
                except Exception as e:
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            if batch_tensors:
                # Process batch on GPU
                try:
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    
                    if self.config['use_mixed_precision']:
                        batch_tensor = batch_tensor.half()
                    
                    with torch.no_grad():
                        features = self.image_model(batch_tensor)
                    
                    # Move back to CPU
                    batch_features = features.cpu().float().numpy()
                    all_features.extend(batch_features)
                    
                    # Clear GPU cache
                    del batch_tensor, features
                    if self.gpu_available:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"GPU batch processing failed: {e}")
                    # Fallback to zeros
                    batch_features = np.zeros((len(batch_tensors), 2048))
                    all_features.extend(batch_features)
            
        return np.array(all_features)
    
    def prepare_features_hybrid(self, df, is_training=True):
        """Hybrid CPU+GPU feature preparation"""
        start_time = time.time()
        
        logger.info(f"🔄 Hybrid feature extraction: {len(df)} samples")
        logger.info(f"   CPU cores: {self.config['parallel_workers']}")
        logger.info(f"   GPU batch size: {self.config['batch_size']}")
        
        # CPU: Text feature extraction (parallel)
        logger.info("🔤 CPU: Extracting text features...")
        text_features_df = self.extract_features_parallel(df['catalog_content'])
        logger.info(f"   ✓ Text features: {text_features_df.shape[1]}")
        
        # CPU: TF-IDF processing
        logger.info("📝 CPU: Processing TF-IDF...")
        processed_texts = [self.preprocess_text(text) for text in df['catalog_content']]
        
        if is_training:
            self.vectorizer = TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.95,
                sublinear_tf=True
            )
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        else:
            tfidf_matrix = self.vectorizer.transform(processed_texts)
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        logger.info(f"   ✓ TF-IDF features: {tfidf_df.shape[1]}")
        
        # GPU: Image processing (if available)
        image_features_df = None
        if self.use_images:
            logger.info("🖼️  GPU: Processing images...")
            
            # Prepare image paths
            image_folder = 'images'
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
                # Download images in background
                logger.info("   📥 Downloading images...")
                try:
                    from src.utils import download_images
                    download_images(df['image_link'].tolist(), image_folder)
                except Exception as e:
                    logger.warning(f"Image download failed: {e}")
            
            image_paths = [
                os.path.join(image_folder, os.path.basename(link))
                for link in df['image_link']
            ]
            
            # GPU batch processing
            image_features = self.extract_images_gpu_batch(image_paths)
            image_features_df = pd.DataFrame(
                image_features,
                columns=[f'img_{i}' for i in range(image_features.shape[1])]
            )
            logger.info(f"   ✓ Image features: {image_features_df.shape[1]}")
        
        # Combine all features
        logger.info("🔗 Combining features...")
        feature_dfs = [text_features_df, tfidf_df]
        if image_features_df is not None:
            feature_dfs.append(image_features_df)
        
        combined_features = pd.concat(feature_dfs, axis=1)
        combined_features = combined_features.fillna(0)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Hybrid feature extraction: {elapsed_time/60:.1f} min")
        logger.info(f"📊 Total features: {combined_features.shape[1]}")
        
        return combined_features
    
    def preprocess_text(self, text):
        """Fast text preprocessing"""
        try:
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            return ' '.join(text.split())
        except:
            return str(text)
    
    def train_hybrid_models(self, train_df):
        """Train models using optimal CPU/GPU distribution"""
        start_time = time.time()
        
        logger.info("🚀 HYBRID CPU+GPU TRAINING")
        logger.info("=" * 50)
        
        # Feature preparation
        X = self.prepare_features_hybrid(train_df, is_training=True)
        y = train_df['price'].values
        
        # Outlier removal
        Q1, Q3 = np.percentile(y, [25, 75])
        IQR = Q3 - Q1
        mask = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
        X_clean, y_clean = X[mask], y[mask]
        
        logger.info(f"Removed {len(y) - len(y_clean)} outliers")
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train CPU models (parallel)
        logger.info("🤖 Training CPU models...")
        
        # Random Forest (CPU parallel)
        logger.info("  🌳 Random Forest (CPU)...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self.models['rf'].fit(X_scaled, y_clean)
        
        # Extra Trees (CPU parallel)
        logger.info("  🌲 Extra Trees (CPU)...")
        self.models['et'] = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        self.models['et'].fit(X_scaled, y_clean)
        
        # Ridge (CPU)
        logger.info("  📊 Ridge (CPU)...")
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_scaled, y_clean)
        
        # Train GPU models if available
        if self.gpu_available:
            logger.info("🚀 Training GPU models...")
            
            # XGBoost with GPU
            if XGB_AVAILABLE:
                logger.info("  ⚡ XGBoost (GPU)...")
                self.models['xgb'] = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method='gpu_hist',
                    gpu_id=0,
                    random_state=42,
                    verbosity=0
                )
                self.models['xgb'].fit(X_scaled, y_clean)
            
            # LightGBM with GPU
            if LGB_AVAILABLE:
                logger.info("  💡 LightGBM (GPU)...")
                self.models['lgb'] = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    device='gpu',
                    random_state=42,
                    verbosity=-1
                )
                self.models['lgb'].fit(X_scaled, y_clean)
        
        # Model validation
        logger.info("📊 Cross-validation...")
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y_clean, cv=3, 
                                       scoring='neg_mean_absolute_error', n_jobs=-1)
                logger.info(f"  {name.upper()}: MAE = {-scores.mean():.2f}")
            except Exception as e:
                logger.warning(f"CV failed for {name}: {e}")
        
        training_time = time.time() - start_time
        logger.info(f"🎉 Hybrid training completed: {training_time/60:.1f} min")
        
        # Cleanup
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        return X_scaled, y_clean
    
    def predict_hybrid(self, test_df):
        """Hybrid prediction using all available models"""
        logger.info("🔮 Hybrid prediction...")
        
        X = self.prepare_features_hybrid(test_df, is_training=False)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[name] = pred
                logger.info(f"  {name.upper()}: ${pred.min():.2f} - ${pred.max():.2f}")
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
        
        # Smart ensemble weighting
        if len(predictions) >= 4:
            # More models available
            weights = {'rf': 0.25, 'et': 0.2, 'xgb': 0.25, 'lgb': 0.2, 'ridge': 0.1}
        elif len(predictions) >= 3:
            weights = {'rf': 0.35, 'et': 0.25, 'xgb': 0.3, 'ridge': 0.1}
        else:
            # Equal weights for few models
            weight_value = 1.0 / len(predictions)
            weights = {name: weight_value for name in predictions.keys()}
        
        # Ensemble prediction
        ensemble_pred = np.zeros(len(test_df))
        total_weight = 0
        
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
                total_weight += weights[name]
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Ensure positive prices
        ensemble_pred = np.clip(ensemble_pred, 0.1, 10000)
        
        logger.info(f"✓ Final ensemble: ${ensemble_pred.min():.2f} - ${ensemble_pred.max():.2f}")
        
        return ensemble_pred
    
    def save_model(self, filepath):
        """Save hybrid model"""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'config': self.config,
            'gpu_available': self.gpu_available
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Hybrid model saved: {filepath}")

def main():
    """Main hybrid pipeline"""
    start_time = time.time()
    
    logger.info("🚀 HYBRID CPU+GPU SMART PRODUCT PRICING")
    logger.info("=" * 60)
    
    # Initialize hybrid predictor
    predictor = HybridCPUGPUPredictor()
    
    try:
        # Load data
        logger.info("📂 Loading datasets...")
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')
        
        logger.info(f"📊 Training: {len(train_df):,} samples")
        logger.info(f"📊 Test: {len(test_df):,} samples")
        
        # Train hybrid model
        X_train, y_train = predictor.train_hybrid_models(train_df)
        
        # Save model
        predictor.save_model('hybrid_cpu_gpu_model.pkl')
        
        # Make predictions
        predictions = predictor.predict_hybrid(test_df)
        
        # Create output
        output_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        # Save results
        output_path = 'dataset/test_out_hybrid.csv'
        output_df.to_csv(output_path, index=False)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("🎉 HYBRID PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"📊 Predictions: {len(output_df):,}")
        logger.info(f"💰 Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        logger.info(f"💰 Mean: ${predictions.mean():.2f}")
        logger.info(f"⏱️  Runtime: {total_time/60:.1f} minutes")
        logger.info(f"🔧 Models: {list(predictor.models.keys())}")
        logger.info(f"🚀 GPU used: {predictor.gpu_available}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Hybrid pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()