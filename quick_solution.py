#!/usr/bin/env python3
"""
Improved Quick Solution for Smart Product Pricing Challenge
Robust text-only approach with better error handling and performance
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional packages
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available")

def extract_enhanced_features(catalog_content):
    """Extract enhanced features from catalog content with error handling"""
    features = {}
    
    try:
        text = str(catalog_content).lower()
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len([c for c in text if c.isalpha()])
        features['digit_count'] = len([c for c in text if c.isdigit()])
        
        # Extract numbers with better handling
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            nums = [float(x) for x in numbers if float(x) > 0]  # Filter zeros
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
        unit_groups = {
            'weight': ['oz', 'ounce', 'lb', 'pound', 'kg', 'gram', 'g'],
            'volume': ['ml', 'liter', 'l', 'fl oz', 'gallon'],
            'count': ['count', 'pack', 'piece', 'ct', 'pcs', 'box'],
            'size': ['inch', 'cm', 'mm', 'ft']
        }
        
        for unit_type, unit_list in unit_groups.items():
            features[f'has_{unit_type}'] = 1 if any(unit in text for unit in unit_list) else 0
        
        # Enhanced quality indicators
        quality_groups = {
            'premium': ['premium', 'gourmet', 'luxury', 'deluxe', 'professional'],
            'natural': ['organic', 'natural', 'pure', 'fresh', 'raw'],
            'brand': ['brand', 'authentic', 'original', 'genuine'],
            'quality': ['quality', 'grade', 'finest', 'best', 'top']
        }
        
        for quality_type, quality_list in quality_groups.items():
            features[f'has_{quality_type}'] = 1 if any(word in text for word in quality_list) else 0
        
        # Enhanced category detection
        categories = {
            'food': ['food', 'snack', 'candy', 'chocolate', 'sauce', 'spice', 'oil', 'drink'],
            'beauty': ['cosmetic', 'beauty', 'lip', 'skin', 'hair', 'makeup', 'lotion'],
            'cleaning': ['cleaning', 'detergent', 'soap', 'wash', 'cleaner'],
            'health': ['vitamin', 'supplement', 'medicine', 'health'],
            'electronics': ['electronic', 'digital', 'battery', 'charger']
        }
        
        for category, keywords in categories.items():
            features[f'is_{category}'] = 1 if any(word in text for word in keywords) else 0
        
        # Text complexity features
        words = text.split()
        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
            features['unique_word_ratio'] = len(set(words)) / len(words)
        else:
            features['avg_word_length'] = 0
            features['unique_word_ratio'] = 0
        
        # Sentence count
        sentences = [s for s in text.split('.') if len(s.strip()) > 0]
        features['sentence_count'] = len(sentences)
        
        # Price-related keywords
        price_keywords = ['cheap', 'expensive', 'affordable', 'budget', 'value', 'deal', 'sale']
        features['has_price_keywords'] = 1 if any(word in text for word in price_keywords) else 0
        
    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
        # Return default features on error
        features = {f'feature_{i}': 0 for i in range(30)}
    
    return features

def preprocess_text(text):
    """Clean text for TF-IDF"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return ' '.join(text.split())

def train_and_predict_improved():
    """Improved main pipeline with better error handling and ensemble"""
    DATASET_FOLDER = 'dataset/'
    
    try:
        logger.info("🚀 Starting Improved Quick Solution")
        logger.info("=" * 50)
        
        # Load data with error handling
        logger.info("📂 Loading data...")
        train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
        test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
        
        logger.info(f"📊 Training samples: {len(train_df):,}")
        logger.info(f"📊 Test samples: {len(test_df):,}")
        
        # Data validation
        if train_df.empty or test_df.empty:
            raise ValueError("Empty dataset found")
        
        # Extract enhanced features for training
        logger.info("🔤 Extracting enhanced training features...")
        train_features = []
        failed_extractions = 0
        
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training features"):
            try:
                features = extract_enhanced_features(row['catalog_content'])
                train_features.append(features)
            except Exception as e:
                failed_extractions += 1
                train_features.append({f'feature_{i}': 0 for i in range(30)})
        
        if failed_extractions > 0:
            logger.warning(f"Failed to extract features for {failed_extractions} training samples")
        
        train_features_df = pd.DataFrame(train_features)
        logger.info(f"✓ Extracted {train_features_df.shape[1]} engineered features")
        
        # Enhanced TF-IDF features
        logger.info("📝 Processing enhanced TF-IDF...")
        train_texts = [preprocess_text(text) for text in tqdm(train_df['catalog_content'], desc="Text preprocessing")]
        
        vectorizer = TfidfVectorizer(
            max_features=1500,  # Increased for better representation
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=3,
            max_df=0.95,
            sublinear_tf=True
        )
        
        train_tfidf = vectorizer.fit_transform(train_texts)
        train_tfidf_df = pd.DataFrame(
            train_tfidf.toarray(),
            columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
        )
        logger.info(f"✓ Extracted {train_tfidf_df.shape[1]} TF-IDF features")
        
        # Combine features
        X_train = pd.concat([train_features_df, train_tfidf_df], axis=1)
        y_train = train_df['price'].values
        
        # Remove outliers
        Q1 = np.percentile(y_train, 25)
        Q3 = np.percentile(y_train, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (y_train >= lower_bound) & (y_train <= upper_bound)
        X_train_clean = X_train[outlier_mask]
        y_train_clean = y_train[outlier_mask]
        
        logger.info(f"Removed {len(y_train) - len(y_train_clean)} outliers")
        
        # Scale features with robust scaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        
        # Train ensemble of models
        logger.info("🤖 Training ensemble models...")
        models = {}
        
        # Random Forest
        logger.info("  Training Random Forest...")
        models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        models['rf'].fit(X_train_scaled, y_train_clean)
        
        # Extra Trees
        logger.info("  Training Extra Trees...")
        models['et'] = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        models['et'].fit(X_train_scaled, y_train_clean)
        
        # XGBoost if available
        if XGB_AVAILABLE:
            logger.info("  Training XGBoost...")
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            models['xgb'].fit(X_train_scaled, y_train_clean)
        
        # Ridge Regression
        logger.info("  Training Ridge Regression...")
        models['ridge'] = Ridge(alpha=1.0)
        models['ridge'].fit(X_train_scaled, y_train_clean)
        
        # Cross-validation
        logger.info("📊 Cross-validation scores:")
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_train_scaled, y_train_clean, 
                                       cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
                logger.info(f"  {name.upper()}: MAE = {-scores.mean():.2f} (±{scores.std() * 2:.2f})")
            except Exception as e:
                logger.warning(f"CV failed for {name}: {e}")
        
        # Extract test features
        logger.info("🔍 Extracting test features...")
        test_features = []
        failed_test_extractions = 0
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test features"):
            try:
                features = extract_enhanced_features(row['catalog_content'])
                test_features.append(features)
            except Exception as e:
                failed_test_extractions += 1
                test_features.append({f'feature_{i}': 0 for i in range(30)})
        
        if failed_test_extractions > 0:
            logger.warning(f"Failed to extract features for {failed_test_extractions} test samples")
        
        test_features_df = pd.DataFrame(test_features)
        
        # Test TF-IDF
        test_texts = [preprocess_text(text) for text in tqdm(test_df['catalog_content'], desc="Test text preprocessing")]
        test_tfidf = vectorizer.transform(test_texts)
        test_tfidf_df = pd.DataFrame(
            test_tfidf.toarray(),
            columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])]
        )
        
        # Combine test features
        X_test = pd.concat([test_features_df, test_tfidf_df], axis=1)
        X_test_scaled = scaler.transform(X_test)
        
        # Make ensemble predictions
        logger.info("🔮 Making ensemble predictions...")
        predictions = {}
        
        for name, model in models.items():
            try:
                pred = model.predict(X_test_scaled)
                predictions[name] = pred
                logger.info(f"  {name.upper()}: ${pred.min():.2f} - ${pred.max():.2f}")
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
        
        # Ensemble with dynamic weights
        if len(predictions) >= 3:
            weights = {'rf': 0.35, 'et': 0.25, 'xgb': 0.25, 'ridge': 0.15}
        else:
            # Equal weights if fewer models
            weight_value = 1.0 / len(predictions)
            weights = {name: weight_value for name in predictions.keys()}
        
        ensemble_pred = np.zeros(len(test_df))
        total_weight = 0
        
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
                total_weight += weights[name]
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Ensure positive prices with reasonable bounds
        ensemble_pred = np.clip(ensemble_pred, 0.1, 10000)
        
        # Create output
        output_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': ensemble_pred
        })
        
        # Save results
        output_path = os.path.join(DATASET_FOLDER, 'test_out.csv')
        output_df.to_csv(output_path, index=False)
        
        # Final summary
        logger.info("=" * 50)
        logger.info("🎉 IMPROVED QUICK SOLUTION COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"📁 Predictions saved to: {output_path}")
        logger.info(f"📊 Total predictions: {len(output_df):,}")
        logger.info(f"💰 Price range: ${ensemble_pred.min():.2f} - ${ensemble_pred.max():.2f}")
        logger.info(f"💰 Mean price: ${ensemble_pred.mean():.2f}")
        logger.info(f"🤖 Models used: {list(models.keys())}")
        logger.info("=" * 50)
        
        return output_df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    try:
        results = train_and_predict_improved()
        logger.info("🎉 Done!")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        print("\n🔧 Try running the diagnostic tool first:")
        print("python diagnostic_tool.py")
        raise