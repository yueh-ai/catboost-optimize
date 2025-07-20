#!/usr/bin/env python3

import json
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

def train_diamonds_model():
    print("🏗️  Training CatBoost model on Diamonds dataset...")
    
    # Load diamonds dataset
    print("📊 Loading Diamonds dataset...")
    diamonds = sns.load_dataset('diamonds')
    print(f"   Dataset shape: {diamonds.shape}")
    
    # Prepare features and target
    X = diamonds.drop('price', axis=1)
    y = diamonds['price']
    
    # Handle categorical features
    categorical_features = ['cut', 'color', 'clarity']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train CatBoost model
    print("\n🚀 Training CatBoost model...")
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        cat_features=categorical_features,
        random_seed=42,
        verbose=100
    )
    
    # Create Pool objects for proper C++ export
    from catboost import Pool
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, y_test, cat_features=categorical_features)
    
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
        early_stopping_rounds=50
    )
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\n📈 Model performance:")
    print(f"   Train R²: {train_score:.4f}")
    print(f"   Test R²: {test_score:.4f}")
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    
    # Save as CBM (CatBoost native format)
    model_path = '../models/baseline.cbm'
    model.save_model(model_path)
    print(f"\n💾 Saved model to: {model_path}")
    
    # Export to C++
    cpp_path = '../models/baseline.cpp'
    model.save_model(cpp_path, format='cpp', pool=train_pool)
    print(f"   Exported C++ to: {cpp_path}")
    
    # Save metadata
    metadata = {
        'features': list(X.columns),
        'categorical_features': categorical_features,
        'numeric_features': [f for f in X.columns if f not in categorical_features],
        'feature_ranges': {},
        'categorical_values': {},
        'model_params': {
            'iterations': model.tree_count_,
            'depth': model.get_param('depth'),
            'learning_rate': model.get_param('learning_rate')
        },
        'performance': {
            'train_r2': float(train_score),
            'test_r2': float(test_score),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    }
    
    # Calculate feature ranges
    for feature in X.columns:
        if feature in categorical_features:
            metadata['categorical_values'][feature] = list(X[feature].unique())
        else:
            metadata['feature_ranges'][feature] = {
                'min': float(X[feature].min()),
                'max': float(X[feature].max()),
                'mean': float(X[feature].mean()),
                'std': float(X[feature].std())
            }
    
    metadata_path = '../models/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata to: {metadata_path}")
    
    print("\n✅ Model training completed successfully!")
    return model, X, y

if __name__ == "__main__":
    train_diamonds_model()