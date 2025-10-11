#!/usr/bin/env python3
"""
Quick Solution for Smart Product Pricing Challenge
Simplified approach focusing on text features for faster execution
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def extract_features(catalog_content):
    """Extract basic features from catalog content"""
    features = {}
    text = str(catalog_content).lower()
    
    # Basic text statistics
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Extract numbers (quantities, sizes, weights)
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        nums = [float(x) for x in numbers]
        features['max_number'] = max(nums)
        features['avg_number'] = np.mean(nums)
        features['number_count'] = len(nums)
    else:
        features['max_number'] = 0
        features['avg_number'] = 0
        features['number_count'] = 0
    
    # Unit indicators
    units = ['oz', 'ounce', 'lb', 'pound', 'count', 'pack', 'fl oz']
    for unit in units:
        features[f'has_{unit.replace(" ", "_")}'] = 1 if unit in text else 0
    
    # Quality indicators
    quality_words = ['premium', 'gourmet', 'organic', 'natural', 'fresh']
    for word in quality_words:
        features[f'has_{word}'] = 1 if word in text else 0
    
    # Category indicators
    food_words = ['food', 'snack', 'candy', 'sauce', 'spice', 'chocolate']
    beauty_words = ['cosmetic', 'beauty', 'lip', 'lotion']
    
    features['is_food'] = 1 if any(word in text for word in food_words) else 0
    features['is_beauty'] = 1 if any(word in text for word in beauty_words) else 0
    
    return features

def preprocess_text(text):
    """Clean text for TF-IDF"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return ' '.join(text.split())

def train_and_predict():
    """Main pipeline"""
    DATASET_FOLDER = 'dataset/'
    
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Extract features for training
    print("Extracting training features...")
    train_features = []
    for _, row in train_df.iterrows():
        features = extract_features(row['catalog_content'])
        train_features.append(features)
    
    train_features_df = pd.DataFrame(train_features)
    
    # TF-IDF features
    print("Processing TF-IDF...")
    train_texts = [preprocess_text(text) for text in train_df['catalog_content']]
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5
    )
    
    train_tfidf = vectorizer.fit_transform(train_texts)
    train_tfidf_df = pd.DataFrame(
        train_tfidf.toarray(),
        columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    )
    
    # Combine features
    X_train = pd.concat([train_features_df, train_tfidf_df], axis=1)
    y_train = train_df['price'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Extract test features
    print("Extracting test features...")
    test_features = []
    for _, row in test_df.iterrows():
        features = extract_features(row['catalog_content'])
        test_features.append(features)
    
    test_features_df = pd.DataFrame(test_features)
    
    # Test TF-IDF
    test_texts = [preprocess_text(text) for text in test_df['catalog_content']]
    test_tfidf = vectorizer.transform(test_texts)
    test_tfidf_df = pd.DataFrame(
        test_tfidf.toarray(),
        columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])]
    )
    
    # Combine test features
    X_test = pd.concat([test_features_df, test_tfidf_df], axis=1)
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    predictions = np.maximum(predictions, 0.1)  # Ensure positive prices
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save results
    output_path = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Mean price: ${predictions.mean():.2f}")
    
    return output_df

if __name__ == "__main__":
    results = train_and_predict()
    print("Done!")