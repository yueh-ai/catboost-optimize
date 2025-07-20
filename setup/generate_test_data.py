#!/usr/bin/env python3

import json
import struct
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def generate_test_data():
    print("🎲 Generating test data...")
    
    # Load metadata
    with open('../models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of test samples
    n_samples = 1_000_000
    print(f"   Generating {n_samples:,} test vectors")
    
    # Generate test data
    test_data = {}
    
    # Generate numeric features
    for feature, ranges in metadata['feature_ranges'].items():
        # Use normal distribution within reasonable bounds
        mean = ranges['mean']
        std = ranges['std']
        min_val = ranges['min']
        max_val = ranges['max']
        
        # Generate values with normal distribution
        values = np.random.normal(mean, std, n_samples)
        # Clip to valid range
        values = np.clip(values, min_val, max_val)
        test_data[feature] = values
        print(f"   Generated {feature}: [{min_val:.2f}, {max_val:.2f}]")
    
    # Generate categorical features
    for feature, categories in metadata['categorical_values'].items():
        # Sample with realistic distribution
        if feature == 'cut':
            # Assume quality distribution (more common cuts)
            weights = [0.05, 0.15, 0.25, 0.30, 0.25]  # Fair, Good, Very Good, Premium, Ideal
        elif feature == 'color':
            # Assume normal distribution around middle grades
            n_colors = len(categories)
            weights = np.exp(-0.5 * ((np.arange(n_colors) - n_colors/2) / 2)**2)
            weights = weights / weights.sum()
        elif feature == 'clarity':
            # Assume quality distribution
            weights = [0.02, 0.08, 0.15, 0.25, 0.25, 0.15, 0.08, 0.02]
        else:
            weights = None
        
        values = np.random.choice(categories, n_samples, p=weights)
        test_data[feature] = values
        print(f"   Generated {feature}: {len(categories)} categories")
    
    # Create DataFrame
    test_df = pd.DataFrame(test_data)
    
    # Reorder columns to match training data
    test_df = test_df[metadata['features']]
    
    # Load model and generate ground truth predictions
    print("\n🔮 Generating ground truth predictions...")
    model = CatBoostRegressor()
    model.load_model('../models/baseline.cbm')
    
    # Generate predictions in batches to avoid memory issues
    batch_size = 10000
    predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch = test_df.iloc[i:i+batch_size]
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)
        if i % 100000 == 0:
            print(f"   Progress: {i:,}/{n_samples:,}")
    
    predictions = np.array(predictions, dtype=np.float32)
    print(f"   Generated {len(predictions):,} predictions")
    
    # Save test data in binary format
    print("\n💾 Saving test data...")
    
    # Convert categorical to numeric codes for binary storage
    encoded_data = []
    encoding_map = {}
    
    for feature in metadata['features']:
        if feature in metadata['categorical_features']:
            # Create encoding
            categories = metadata['categorical_values'][feature]
            encoding = {cat: i for i, cat in enumerate(categories)}
            encoding_map[feature] = encoding
            
            # Encode values
            values = test_df[feature].map(encoding).values.astype(np.float32)
        else:
            values = test_df[feature].values.astype(np.float32)
        
        encoded_data.append(values)
    
    # Stack features
    features_array = np.column_stack(encoded_data)
    
    # Binary format: 
    # - Header: magic number (4 bytes), version (4 bytes), n_samples (4 bytes), n_features (4 bytes)
    # - Data: features (n_samples x n_features x 4 bytes) + predictions (n_samples x 4 bytes)
    
    binary_path = '../models/test_data.bin'
    with open(binary_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', 0xCAFEBABE))  # Magic number
        f.write(struct.pack('I', 1))            # Version
        f.write(struct.pack('I', n_samples))    # Number of samples
        f.write(struct.pack('I', len(metadata['features'])))  # Number of features
        
        # Write features
        features_array.astype(np.float32).tofile(f)
        
        # Write predictions
        predictions.astype(np.float32).tofile(f)
    
    file_size_mb = os.path.getsize(binary_path) / (1024 * 1024)
    print(f"   Saved binary data: {binary_path} ({file_size_mb:.1f} MB)")
    
    # Save encoding map
    encoding_path = '../models/encoding_map.json'
    with open(encoding_path, 'w') as f:
        json.dump(encoding_map, f, indent=2)
    print(f"   Saved encoding map: {encoding_path}")
    
    # Save sample of test data as CSV for debugging
    sample_path = '../models/test_data_sample.csv'
    sample_df = test_df.head(1000).copy()
    sample_df['prediction'] = predictions[:1000]
    sample_df.to_csv(sample_path, index=False)
    print(f"   Saved sample CSV: {sample_path}")
    
    # Print statistics
    print("\n📊 Test data statistics:")
    print(f"   Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"   Predictions mean: {predictions.mean():.2f}")
    print(f"   Predictions std: {predictions.std():.2f}")
    
    print("\n✅ Test data generation completed successfully!")

if __name__ == "__main__":
    generate_test_data()