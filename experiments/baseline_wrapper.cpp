// Include the baseline model
#include "../models/baseline.cpp"

// Add the required extern "C" wrapper function
extern "C" {
    float catboostPredict(const float* features, int featureCount) {
        // The test data has 9 features total: 6 numeric + 3 categorical (encoded as floats)
        // Model expects: 6 floats + 3 categorical strings
        
        if (featureCount != 9) {
            return -1.0f; // Error: unexpected number of features
        }
        
        // First 6 features are numeric
        std::vector<float> floatFeatures(features, features + 6);
        
        // Last 3 features are categorical indices (encoded as floats)
        std::vector<std::string> catFeatures;
        
        // Define categorical mappings (must match the order in categorical_mappings.json)
        const std::vector<std::string> cutCategories = {"Ideal", "Premium", "Good", "Very Good", "Fair"};
        const std::vector<std::string> colorCategories = {"E", "I", "J", "H", "F", "G", "D"};
        const std::vector<std::string> clarityCategories = {"SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"};
        
        // Convert categorical indices to strings
        int cutIndex = static_cast<int>(features[6]);
        int colorIndex = static_cast<int>(features[7]);
        int clarityIndex = static_cast<int>(features[8]);
        
        // Bounds checking
        if (cutIndex >= 0 && cutIndex < cutCategories.size()) {
            catFeatures.push_back(cutCategories[cutIndex]);
        } else {
            catFeatures.push_back("Unknown");
        }
        
        if (colorIndex >= 0 && colorIndex < colorCategories.size()) {
            catFeatures.push_back(colorCategories[colorIndex]);
        } else {
            catFeatures.push_back("Unknown");
        }
        
        if (clarityIndex >= 0 && clarityIndex < clarityCategories.size()) {
            catFeatures.push_back(clarityCategories[clarityIndex]);
        } else {
            catFeatures.push_back("Unknown");
        }
        
        // Call the original model function
        return static_cast<float>(ApplyCatboostModel(floatFeatures, catFeatures));
    }
}