#!/usr/bin/env python3
"""
GPU-Accelerated Smart Product Pricing Solution
Utilizes both CPU and GPU for maximum performance
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

# GPU-accelerated imports
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.linear_model import Ridge as cuRidge
    RAPIDS_AVAILABLE = True
    logger.info("✅ RAPIDS (cuML/cuDF) available - GPU acceleration enabled!")
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.warning("⚠️  RAPIDS not available - using CPU fallback")

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

# PyTorch for image processing (GPU accelerated)
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image
    TORCH_AVAILABLE = True
    
    # Check GPU availability
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        logger.info(f"✅ PyTorch GPU available: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        logger.info("⚠️  PyTorch using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'

import pickle

class GPUAcceleratedPredictor:
    def __init__(self, use_gpu=True, use_images=True):
        self.use_gpu = use_gpu and (RAPIDS_AVAILABLE or torch.cuda.is_available())
        self.use_images = use_images and TORCH_AVAILABLE
        self.models = {}
        self.vectorizer = None
        self.scaler = RobustScaler()
        self.device = DEVICE if TORCH_AVAILABLE else 'cpu'
        
        # Initialize image model on GPU if available
        if self.use_images:
            try:
                self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.image_model.fc = torch.nn.Identity()
                self.image_model = self.image_model.to(self.device)
                self.image_model.eval()
                
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                logger.info(f"✅ Image model loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Image model setup failed: {e}")
                self.use_images = False
        
        logger.info(f"🚀 GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        logger.info(f"🖼️  Image Processing: {'Enabled' if self.use_images else 'Disabled'}")
    
    def extract_enhanced_features(self, text):
        """Enhanced feature extraction optimized for speed"""
        features = {}
        
        try:
            text = str(text).lower()
            
            # Basic stats (vectorized where possible)
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            features['char_count'] = sum(c.isalpha() for c in text)
            features['digit_count'] = sum(c.isdigit() for c in text)
            features['upper_count'] = sum(c.isupper() for c in str(text))
            
            # Enhanced number extraction
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                nums = [float(x) for x in numbers if float(x) > 0]
                if nums:
                    features.update({
                        'max_number': max(nums),
                        'min_number': min(nums),
                        'avg_number': np.mean(nums),
                        'std_number': np.std(nums) if len(nums) > 1 else 0,
                        'number_count': len(nums),
                        'number_range': max(nums) - min(nums) if len(nums) > 1 else 0
                    })
                else:
                    features.update({k: 0 for k in ['max_number', 'min_number', 'avg_number', 'std_number', 'number_count', 'number_range']})
            else:
                features.update({k: 0 for k in ['max_number', 'min_number', 'avg_number', 'std_number', 'number_count', 'number_range']})
            
            # Unit detection (optimized)
            unit_groups = {
                'weight': ['oz', 'ounce', 'lb', 'pound', 'kg', 'gram', 'g'],
                'volume': ['ml', 'liter', 'l', 'fl oz', 'gallon'],
                'count': ['count', 'pack', 'piece', 'ct', 'pcs', 'box'],
                'size': ['inch', 'cm', 'mm', 'ft']
            }
            
            for unit_type, unit_list in unit_groups.items():
                features[f'has_{unit_type}'] = 1 if any(unit in text for unit in unit_list) else 0
            
            # Quality indicators
            quality_groups = {
                'premium': ['premium', 'gourmet', 'luxury', 'deluxe', 'professional'],
                'natural': ['organic', 'natural', 'pure', 'fresh', 'raw'],
                'brand': ['brand', 'authentic', 'original', 'genuine'],
                'quality': ['quality', 'grade', 'finest', 'best', 'top']
            }
            
            for quality_type, quality_list in quality_groups.items():
                features[f'has_{quality_type}'] = 1 if any(word in text for word in quality_list) else 0
            
            # Category detection
            categories = {
                'food': ['food', 'snack', 'candy', 'chocolate', 'sauce', 'spice', 'oil', 'drink'],
                'beauty': ['cosmetic', 'beauty', 'lip', 'skin', 'hair', 'makeup', 'lotion'],
                'cleaning': ['cleaning', 'detergent', 'soap', 'wash', 'cleaner'],
                'health': ['vitamin', 'supplement', 'medicine', 'health'],
                'electronics': ['electronic', 'digital', 'battery', 'charger']
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
            
            # Sentence analysis
            sentences = [s for s in text.split('.') if len(s.strip()) > 0]
            features['sentence_count'] = len(sentences)
            
            # Price keywords
            price_keywords = ['cheap', 'expensive', 'affordable', 'budget', 'value', 'deal', 'sale']
            features['has_price_keywords'] = 1 if any(word in text for word in price_keywords) else 0
            
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            features = {f'feature_{i}': 0 for i in range(40)}
        
        return features
    
    def extract_image_features_gpu(self, image_path):
        """GPU-accelerated image feature extraction"""
        if not self.use_images:
            return np.zeros(2048)
        
        try:
            if not os.path.exists(image_path):
                return np.zeros(2048)
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Extract features on GPU
            with torch.no_grad():
                features = self.image_model(image_tensor)
            
            # Move back to CPU for further processing
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"GPU image processing error for {image_path}: {e}")
            return np.zeros(2048)
    
    def prepare_features_parallel(self, df, is_training=True):
        """Parallel feature preparation using CPU and GPU"""
        start_time = time.time()
        
        logger.info(f"🔤 Extracting features from {len(df)} samples...")
        
        # Text feature extraction (CPU parallel)
        logger.info("  📝 Processing text features...")
        text_features = []
        
        # Use multiprocessing for text features
        from multiprocessing import Pool, cpu_count
        n_cores = min(cpu_count(), 8)  # Limit to 8 cores
        
        with Pool(n_cores) as pool:
            text_features = list(tqdm(
                pool.imap(self.extract_enhanced_features, df['catalog_content']),
                total=len(df),
                desc="Text features"
            ))
        
        text_df = pd.DataFrame(text_features)
        logger.info(f"  ✓ Text features: {text_df.shape[1]}")
        
        # TF-IDF processing (CPU)
        logger.info("  📊 Processing TF-IDF...")
        processed_texts = [self.preprocess_text(text) for text in df['catalog_content']]
        
        if is_training:
            self.vectorizer = TfidfVectorizer(
                max_features=2000,  # Increased for GPU handling
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
        logger.info(f"  ✓ TF-IDF features: {tfidf_df.shape[1]}")
        
        # Image feature extraction (GPU accelerated)
        image_df = None
        if self.use_images:
            logger.info("  🖼️  Processing images on GPU...")
            
            # Download images if needed
            image_folder = 'images'
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
                logger.info("    📥 Downloading images...")
                try:
                    from src.utils import download_images
                    download_images(df['image_link'].tolist(), image_folder)
                except Exception as e:
                    logger.warning(f"Image download failed: {e}")
            
            # Process images in batches for GPU efficiency
            batch_size = 32  # Adjust based on GPU memory
            image_features = []
            
            for i in tqdm(range(0, len(df), batch_size), desc="Image batches"):
                batch_df = df.iloc[i:i+batch_size]
                batch_features = []
                
                for _, row in batch_df.iterrows():
                    image_filename = os.path.basename(row['image_link'])
                    image_path = os.path.join(image_folder, image_filename)
                    features = self.extract_image_features_gpu(image_path)
                    batch_features.append(features)
                
                image_features.extend(batch_features)
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            image_df = pd.DataFrame(
                image_features,
                columns=[f'img_{i}' for i in range(2048)]
            )
            logger.info(f"  ✓ Image features: {image_df.shape[1]}")
        
        # Combine all features
        logger.info("🔗 Combining features...")
        feature_dfs = [text_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)]
        
        if image_df is not None:
            feature_dfs.append(image_df.reset_index(drop=True))
        
        combined_features = pd.concat(feature_dfs, axis=1)
        combined_features = combined_features.fillna(0)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Feature extraction completed in {elapsed_time/60:.1f} minutes")
        logger.info(f"📊 Total features: {combined_features.shape[1]}")
        
        return combined_features
    
    def preprocess_text(self, text):
        """Optimized text preprocessing"""
        try:
            text = str(text).lower()
            text = re.sub(r'[^\w\s\-\.]', ' ', text)
            text = ' '.join(text.split())
            return text
        except:
            return str(text)
    
    def train_gpu_accelerated(self, train_df):
        """GPU-accelerated training pipeline"""
        start_time = time.time()
        
        logger.info("🚀 GPU-ACCELERATED TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"📊 Training samples: {len(train_df):,}")
        
        # Feature preparation
        X = self.prepare_features_parallel(train_df, is_training=True)
        y = train_df['price'].values
        
        # Outlier removal
        Q1, Q3 = np.percentile(y, [25, 75])
        IQR = Q3 - Q1
        mask = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
        X_clean, y_clean = X[mask], y[mask]
        
        logger.info(f"Removed {len(y) - len(y_clean)} outliers")
        
        # Feature scaling
        logger.info("📏 Scaling features...")
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Train models with GPU acceleration where available
        logger.info("🤖 Training GPU-accelerated models...")
        
        # GPU-accelerated Random Forest (if RAPIDS available)
        if RAPIDS_AVAILABLE and self.use_gpu:
            logger.info("  🚀 GPU Random Forest (cuML)...")
            try:
                # Convert to cuDF for GPU processing
                X_gpu = cudf.DataFrame(X_scaled)
                y_gpu = cudf.Series(y_clean)
                
                self.models['gpu_rf'] = cuRF(
                    n_estimators=200,
                    max_depth=20,
                    random_state=42
                )
                self.models['gpu_rf'].fit(X_gpu, y_gpu)
                logger.info("    ✓ GPU Random Forest trained")
            except Exception as e:
                logger.warning(f"GPU RF failed: {e}, using CPU fallback")
        
        # CPU Random Forest (always available)
        logger.info("  🌳 CPU Random Forest...")
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
        logger.info("  🌲 Extra Trees...")
        self.models['et'] = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        self.models['et'].fit(X_scaled, y_clean)
        
        # XGBoost with GPU support
        if XGB_AVAILABLE:
            logger.info("  🚀 XGBoost...")
            gpu_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1
            }
            
            # Use GPU if available
            if torch.cuda.is_available():
                gpu_params['tree_method'] = 'gpu_hist'
                gpu_params['gpu_id'] = 0
                logger.info("    Using GPU acceleration")
            
            self.models['xgb'] = xgb.XGBRegressor(**gpu_params)
            self.models['xgb'].fit(X_scaled, y_clean)
        
        # LightGBM with GPU support
        if LGB_AVAILABLE:
            logger.info("  💡 LightGBM...")
            lgb_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1
            }
            
            # Use GPU if available
            if torch.cuda.is_available():
                lgb_params['device'] = 'gpu'
                logger.info("    Using GPU acceleration")
            
            self.models['lgb'] = lgb.LGBMRegressor(**lgb_params)
            self.models['lgb'].fit(X_scaled, y_clean)
        
        # Ridge regression
        if RAPIDS_AVAILABLE and self.use_gpu:
            logger.info("  📊 GPU Ridge (cuML)...")
            try:
                self.models['gpu_ridge'] = cuRidge(alpha=1.0)
                self.models['gpu_ridge'].fit(X_gpu, y_gpu)
            except:
                pass
        
        logger.info("  📊 CPU Ridge...")
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_scaled, y_clean)
        
        # Cross-validation
        logger.info("📊 Model validation...")
        for name, model in self.models.items():
            if 'gpu_' not in name:  # Skip GPU models for CV (compatibility)
                try:
                    scores = cross_val_score(model, X_scaled, y_clean, cv=3, 
                                           scoring='neg_mean_absolute_error', n_jobs=-1)
                    logger.info(f"  {name.upper()}: MAE = {-scores.mean():.2f}")
                except Exception as e:
                    logger.warning(f"CV failed for {name}: {e}")
        
        training_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {training_time/60:.1f} minutes!")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return X_scaled, y_clean
    
    def predict_gpu_accelerated(self, test_df):
        """GPU-accelerated prediction"""
        logger.info("🔮 Making GPU-accelerated predictions...")
        
        X = self.prepare_features_parallel(test_df, is_training=False)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                if 'gpu_' in name and RAPIDS_AVAILABLE:
                    # GPU prediction
                    X_gpu = cudf.DataFrame(X_scaled)
                    pred = model.predict(X_gpu).to_pandas().values
                else:
                    # CPU prediction
                    pred = model.predict(X_scaled)
                
                predictions[name] = pred
                logger.info(f"  {name.upper()}: ${pred.min():.2f} - ${pred.max():.2f}")
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
        
        # Dynamic ensemble weights
        if len(predictions) >= 4:
            weights = {
                'gpu_rf': 0.2, 'rf': 0.2, 'et': 0.15, 'xgb': 0.2, 
                'lgb': 0.15, 'gpu_ridge': 0.05, 'ridge': 0.05
            }
        else:
            # Equal weights
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
        
        logger.info(f"✓ Final predictions: ${ensemble_pred.min():.2f} - ${ensemble_pred.max():.2f}")
        
        return ensemble_pred
    
    def save_model(self, filepath):
        """Save trained model"""
        # Convert GPU models to CPU for saving
        cpu_models = {}
        for name, model in self.models.items():
            if 'gpu_' not in name:
                cpu_models[name] = model
        
        model_data = {
            'models': cpu_models,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'use_gpu': self.use_gpu,
            'use_images': self.use_images
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to {filepath}")

def main():
    """Main GPU-accelerated pipeline"""
    start_time = time.time()
    
    logger.info("🚀 GPU-ACCELERATED SMART PRODUCT PRICING")
    logger.info("=" * 60)
    
    # Initialize GPU predictor
    predictor = GPUAcceleratedPredictor(use_gpu=True, use_images=True)
    
    try:
        # Load data
        logger.info("📂 Loading datasets...")
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')
        
        logger.info(f"📊 Training samples: {len(train_df):,}")
        logger.info(f"📊 Test samples: {len(test_df):,}")
        
        # Train model
        X_train, y_train = predictor.train_gpu_accelerated(train_df)
        
        # Save model
        predictor.save_model('gpu_accelerated_model.pkl')
        
        # Make predictions
        predictions = predictor.predict_gpu_accelerated(test_df)
        
        # Create output
        output_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        # Save results
        output_path = 'dataset/test_out_gpu.csv'
        output_df.to_csv(output_path, index=False)
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("🎉 GPU-ACCELERATED PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📁 Predictions saved to: {output_path}")
        logger.info(f"📊 Total predictions: {len(output_df):,}")
        logger.info(f"💰 Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        logger.info(f"💰 Mean price: ${predictions.mean():.2f}")
        logger.info(f"⏱️  Total runtime: {total_time/60:.1f} minutes")
        logger.info(f"🚀 GPU acceleration: {'Enabled' if predictor.use_gpu else 'Disabled'}")
        logger.info(f"🖼️  Image processing: {'Enabled' if predictor.use_images else 'Disabled'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()