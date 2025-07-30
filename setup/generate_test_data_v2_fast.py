#!/usr/bin/env python3

import json
import struct
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def generate_test_data_v2():
    print("🎲 Generating test data V2 (with categorical handling)...")
    
    # Load metadata
    with open('../models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Set random seed
    np.random.seed(42)
    
    n_samples = 1_000_000
    print(f"   Generating {n_samples:,} test vectors")
    
    # For now, let's create a simpler wrapper approach
    # Instead of changing the entire data format, let's create a C++ wrapper
    # that properly converts the data
    
    print("\n✅ Actually, let's use a different approach...")
    print("   We'll create a proper C++ wrapper that handles the conversion")
    
    # Create the wrapper that matches the original signature
    wrapper_content = '''// Wrapper for baseline.cpp that provides the same interface
// but adds the WebAssembly export

#include "baseline.cpp"

// WebAssembly export that matches the original model signature
extern "C" {
    // We need to export functions to handle string arrays
    void* createStringArray(int size) {
        return new std::vector<std::string>(size);
    }
    
    void setStringArrayElement(void* arr, int index, const char* value) {
        auto* vec = static_cast<std::vector<std::string>*>(arr);
        if (index >= 0 && index < vec->size()) {
            (*vec)[index] = std::string(value);
        }
    }
    
    void deleteStringArray(void* arr) {
        delete static_cast<std::vector<std::string>*>(arr);
    }
    
    // Main prediction function that takes pre-separated features
    float catboostPredictSeparated(
        const float* floatFeatures, 
        int floatCount,
        void* catFeaturesPtr
    ) {
        std::vector<float> floats(floatFeatures, floatFeatures + floatCount);
        auto* catFeatures = static_cast<std::vector<std::string>*>(catFeaturesPtr);
        
        return static_cast<float>(ApplyCatboostModel(floats, *catFeatures));
    }
    
    // Convenience function that takes encoded categorical features
    float catboostPredict(const float* features, int featureCount) {
        // Expected format: 6 floats + 3 categorical indices
        if (featureCount != 9) return -1.0f;
        
        // Extract float features
        std::vector<float> floatFeatures;
        floatFeatures.push_back(features[0]); // carat
        floatFeatures.push_back(features[4]); // depth  
        floatFeatures.push_back(features[5]); // table
        floatFeatures.push_back(features[6]); // x
        floatFeatures.push_back(features[7]); // y
        floatFeatures.push_back(features[8]); // z
        
        // Decode categorical features from indices
        std::vector<std::string> catFeatures;
        
        // Cut (index 1)
        const char* cutValues[] = {"Ideal", "Premium", "Good", "Very Good", "Fair"};
        int cutIdx = static_cast<int>(features[1]);
        if (cutIdx >= 0 && cutIdx < 5) {
            catFeatures.push_back(cutValues[cutIdx]);
        } else {
            catFeatures.push_back("Ideal");
        }
        
        // Color (index 2)
        const char* colorValues[] = {"E", "I", "J", "H", "F", "G", "D"};
        int colorIdx = static_cast<int>(features[2]);
        if (colorIdx >= 0 && colorIdx < 7) {
            catFeatures.push_back(colorValues[colorIdx]);
        } else {
            catFeatures.push_back("E");
        }
        
        // Clarity (index 3)
        const char* clarityValues[] = {"SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"};
        int clarityIdx = static_cast<int>(features[3]);
        if (clarityIdx >= 0 && clarityIdx < 8) {
            catFeatures.push_back(clarityValues[clarityIdx]);
        } else {
            catFeatures.push_back("SI2");
        }
        
        return static_cast<float>(ApplyCatboostModel(floatFeatures, catFeatures));
    }
}
'''
    
    # Save the wrapper
    wrapper_path = '../models/baseline_with_export.cpp'
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    print(f"\n✓ Created wrapper: {wrapper_path}")
    
    print("\nThis wrapper:")
    print("  1. Includes the original baseline.cpp")
    print("  2. Adds WebAssembly exports")
    print("  3. Provides the same interface as the original model")
    print("  4. Handles conversion from numeric to string categoricals")
    
    print("\n✅ Done! Use: ./framework/experiment.sh models/baseline_with_export.cpp")

if __name__ == "__main__":
    generate_test_data_v2()