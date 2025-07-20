#include "../models/baseline.cpp"
#include <vector>
#include <string>

// Wrapper for WASM export
extern "C" {
    float catboostPredict(const float* features, int featureCount) {
        // Feature order from model metadata:
        // carat, cut, color, clarity, depth, table, x, y, z
        // Indices: 0=carat, 1=cut, 2=color, 3=clarity, 4=depth, 5=table, 6=x, 7=y, 8=z
        
        std::vector<float> floatFeatures;
        std::vector<std::string> catFeatures;
        
        // Add numeric features
        floatFeatures.push_back(features[0]); // carat
        floatFeatures.push_back(features[4]); // depth
        floatFeatures.push_back(features[5]); // table
        floatFeatures.push_back(features[6]); // x
        floatFeatures.push_back(features[7]); // y
        floatFeatures.push_back(features[8]); // z
        
        // Convert categorical indices to strings
        // Cut categories: Fair=0, Good=1, Very Good=2, Premium=3, Ideal=4
        const char* cutValues[] = {"Fair", "Good", "Very Good", "Premium", "Ideal"};
        int cutIdx = (int)features[1];
        catFeatures.push_back(cutValues[cutIdx]);
        
        // Color categories: J=0, I=1, H=2, G=3, F=4, E=5, D=6
        const char* colorValues[] = {"J", "I", "H", "G", "F", "E", "D"};
        int colorIdx = (int)features[2];
        catFeatures.push_back(colorValues[colorIdx]);
        
        // Clarity categories: I1=0, SI2=1, SI1=2, VS2=3, VS1=4, VVS2=5, VVS1=6, IF=7
        const char* clarityValues[] = {"I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"};
        int clarityIdx = (int)features[3];
        catFeatures.push_back(clarityValues[clarityIdx]);
        
        // Call the original model
        double prediction = ApplyCatboostModel(floatFeatures, catFeatures);
        
        return (float)prediction;
    }
}