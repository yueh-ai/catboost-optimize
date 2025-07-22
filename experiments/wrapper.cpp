#include <vector>
#include <string>
#include <cstring>
#include <emscripten.h>

// Include the baseline model
#include "../models/baseline.cpp"

// Categorical feature mappings
const char* CUT_MAPPING[] = {"Fair", "Good", "Very Good", "Premium", "Ideal"};
const char* COLOR_MAPPING[] = {"J", "I", "H", "G", "F", "E", "D"};
const char* CLARITY_MAPPING[] = {"I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"};

extern "C" {
    // Process all samples at once - this is the optimization target
    EMSCRIPTEN_KEEPALIVE
    void catboostPredictAll(
        const float* inputData,      // All input data (samples * features)
        double* predictions,         // Output predictions
        int numSamples,
        int numFloatFeatures,
        int numCatFeatures
    ) {
        // This is where optimizations should be implemented
        // Current implementation is naive - just loops through samples
        
        const int featuresPerSample = numFloatFeatures + numCatFeatures;
        
        // Pre-allocate vectors to avoid reallocation
        std::vector<float> floatFeatures(numFloatFeatures);
        std::vector<std::string> catFeatures(numCatFeatures);
        
        for (int i = 0; i < numSamples; i++) {
            const float* sampleData = inputData + (i * featuresPerSample);
            
            // Extract float features
            for (int j = 0; j < numFloatFeatures; j++) {
                floatFeatures[j] = sampleData[j];
            }
            
            // Extract and convert categorical features
            for (int j = 0; j < numCatFeatures; j++) {
                int catIdx = (int)sampleData[numFloatFeatures + j];
                
                // Map indices to strings based on feature position
                switch (j) {
                    case 0: // cut
                        catFeatures[j] = CUT_MAPPING[catIdx];
                        break;
                    case 1: // color
                        catFeatures[j] = COLOR_MAPPING[catIdx];
                        break;
                    case 2: // clarity
                        catFeatures[j] = CLARITY_MAPPING[catIdx];
                        break;
                }
            }
            
            // Make prediction
            predictions[i] = ApplyCatboostModel(floatFeatures, catFeatures);
        }
    }
    
    // Alternative single sample interface for compatibility
    EMSCRIPTEN_KEEPALIVE
    double catboostPredict(const float* features, int featureCount) {
        if (featureCount != 9) {
            return -1.0;
        }
        
        std::vector<float> floatFeatures = {
            features[0], features[1], features[2],
            features[3], features[4], features[5]
        };
        
        std::vector<std::string> catFeatures = {
            CUT_MAPPING[(int)features[6]],
            COLOR_MAPPING[(int)features[7]],
            CLARITY_MAPPING[(int)features[8]]
        };
        
        return ApplyCatboostModel(floatFeatures, catFeatures);
    }
}