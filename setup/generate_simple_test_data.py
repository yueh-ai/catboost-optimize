#!/usr/bin/env python3

import json
import struct
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import seaborn as sns

def generate_simple_test_data():
    print("🎲 Generating simple test data (numeric encoding for categoricals)...")
    
    # Set random seed
    np.random.seed(42)
    
    # Load the diamonds dataset to understand value ranges
    diamonds = sns.load_dataset('diamonds')
    
    # Load saved model for ground truth predictions
    model = CatBoostRegressor()
    model.load_model('../models/baseline.cbm')
    
    n_samples = 1_000_000
    print(f"   Generating {n_samples:,} test vectors")
    
    # Generate features based on actual data distributions
    features = np.zeros((n_samples, 9), dtype=np.float32)
    
    # Numeric features
    features[:, 0] = np.random.uniform(diamonds['carat'].min(), diamonds['carat'].max(), n_samples)  # carat
    features[:, 1] = np.random.uniform(diamonds['depth'].min(), diamonds['depth'].max(), n_samples)  # depth
    features[:, 2] = np.random.uniform(diamonds['table'].min(), diamonds['table'].max(), n_samples)  # table
    features[:, 3] = np.random.uniform(diamonds['x'].min(), diamonds['x'].max(), n_samples)         # x
    features[:, 4] = np.random.uniform(diamonds['y'].min(), diamonds['y'].max(), n_samples)         # y
    features[:, 5] = np.random.uniform(diamonds['z'].min(), diamonds['z'].max(), n_samples)         # z
    
    # Categorical features (as numeric indices)
    cut_values = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_values = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_values = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    
    features[:, 6] = np.random.randint(0, len(cut_values), n_samples)     # cut index
    features[:, 7] = np.random.randint(0, len(color_values), n_samples)   # color index
    features[:, 8] = np.random.randint(0, len(clarity_values), n_samples) # clarity index
    
    print("🔮 Generating ground truth predictions...")
    
    # Create DataFrame for predictions with proper categorical values
    df_for_predictions = pd.DataFrame({
        'carat': features[:, 0],
        'depth': features[:, 1],
        'table': features[:, 2],
        'x': features[:, 3],
        'y': features[:, 4],
        'z': features[:, 5],
        'cut': [cut_values[int(i)] for i in features[:, 6]],
        'color': [color_values[int(i)] for i in features[:, 7]],
        'clarity': [clarity_values[int(i)] for i in features[:, 8]]
    })
    
    # Get predictions in batches
    batch_size = 100000
    predictions = np.zeros(n_samples, dtype=np.float32)
    
    # Create Pool with explicit categorical features
    from catboost import Pool
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = df_for_predictions.iloc[i:end]
        pool = Pool(batch, cat_features=['cut', 'color', 'clarity'])
        predictions[i:end] = model.predict(pool).astype(np.float32)
        print(f"   Progress: {i:,}/{n_samples:,}")
    
    print("💾 Saving test data...")
    
    # Write binary file with correct format
    binary_path = '../models/test_data.bin'
    with open(binary_path, 'wb') as f:
        # Write header (magic, version, nSamples, nFeatures)
        header = struct.pack('IIII', 0xCAFEBABE, 1, n_samples, 9)
        f.write(header)
        
        # Write features
        features.tofile(f)
        
        # Write ground truth predictions
        predictions.tofile(f)
    
    file_size = (16 + n_samples * 9 * 4 + n_samples * 4) / (1024 * 1024)
    print(f"   Saved binary data: {binary_path} ({file_size:.1f} MB)")
    
    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "n_features": 9,
        "feature_names": ["carat", "depth", "table", "x", "y", "z", "cut", "color", "clarity"],
        "feature_types": ["numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "categorical", "categorical", "categorical"],
        "categorical_mappings": {
            "cut": cut_values,
            "color": color_values,
            "clarity": clarity_values
        }
    }
    
    with open('../models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Test data generation completed!")

if __name__ == "__main__":
    generate_simple_test_data()