#!/usr/bin/env python3

import os
import pandas as pd
from catboost import CatBoostRegressor, Pool

def export_numeric_model():
    print("🔧 Re-exporting CatBoost model with all features as numeric...")
    
    # Load the saved model
    model = CatBoostRegressor()
    model.load_model('../models/baseline.cbm')
    print("✓ Loaded saved model")
    
    # Create a dummy dataset with all features as numeric
    # The model needs some data to export properly
    dummy_data = pd.DataFrame({
        'carat': [0.5],
        'cut': [0],      # Numeric encoding
        'color': [0],    # Numeric encoding  
        'clarity': [0],  # Numeric encoding
        'depth': [60.0],
        'table': [55.0],
        'x': [4.0],
        'y': [4.0],
        'z': [2.5]
    })
    
    # Create a pool without categorical features specified
    # This tells CatBoost to treat all features as numeric
    pool = Pool(dummy_data, cat_features=[])
    
    # Export to C++
    cpp_path = '../models/baseline_numeric.cpp'
    model.save_model(cpp_path, format='cpp', pool=pool)
    print(f"✓ Exported numeric C++ model to: {cpp_path}")
    
    # Also create a proper wrapper that adds the required export
    wrapper_content = '''#include "baseline_numeric.cpp"

// WebAssembly export wrapper
extern "C" float catboostPredict(const float* features, int featureCount) {
    // All features are numeric, matching our test data format
    std::vector<float> floatFeatures(features, features + featureCount);
    std::vector<std::string> catFeatures; // Empty for numeric model
    
    // Call the model function and convert double to float
    return static_cast<float>(ApplyCatboostModel(floatFeatures, catFeatures));
}
'''
    
    wrapper_path = '../models/baseline_numeric_wrapper.cpp'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    print(f"✓ Created wrapper at: {wrapper_path}")
    
    print("\n✅ Export completed successfully!")
    print("\nTo use the new model, run:")
    print("  ./framework/experiment.sh models/baseline_numeric_wrapper.cpp")

if __name__ == "__main__":
    export_numeric_model()