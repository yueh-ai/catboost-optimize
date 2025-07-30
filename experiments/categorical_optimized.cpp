// Categorical-optimized version: Direct hash computation + memory optimizations
#include <cstring>

// Include baseline model structures
#include "../models/baseline.cpp"

// Pre-allocated buffers
static thread_local unsigned char binaryFeaturesBuffer[84];
static thread_local float ctrsBuffer[32];  // Max CTR features

// Pre-computed categorical hash lookup tables
static const int CUT_HASHES[] = {1754990671, -570237862, 1700310925, 1933222421, 610519841};
static const int COLOR_HASHES[] = {-1095458675, 1348280313, -472349076, -896563403, -1292729504, 1719715171, -204260682};
static const int CLARITY_HASHES[] = {-1581449724, 579192095, -1896862659, 2143106594, 88967919, 1708347785, 1353923139, -117150168};

// Pre-computed categorical feature packed indexes
static std::unordered_map<int, int> catFeaturePackedIndexes = []() {
    const struct CatboostModel& model = CatboostModelStatic;
    std::unordered_map<int, int> indexes;
    for (unsigned int i = 0; i < model.CatFeatureCount; ++i) {
        indexes[model.CatFeaturesIndex[i]] = i;
    }
    return indexes;
}();

extern "C" {
    float catboostPredict(const float* features, int featureCount) {
        if (featureCount != 9) {
            return -1.0f;
        }
        
        const struct CatboostModel& model = CatboostModelStatic;
        
        // Clear binary features buffer
        memset(binaryFeaturesBuffer, 0, model.BinaryFeatureCount);
        
        
        // Get categorical indices
        const int cutIndex = static_cast<int>(features[6]);
        const int colorIndex = static_cast<int>(features[7]);
        const int clarityIndex = static_cast<int>(features[8]);
        
        // Direct hash lookup with bounds checking
        const int hash0 = (cutIndex >= 0 && cutIndex < 5) ? CUT_HASHES[cutIndex] : 0x7fFFffFF;
        const int hash1 = (colorIndex >= 0 && colorIndex < 7) ? COLOR_HASHES[colorIndex] : 0x7fFFffFF;
        const int hash2 = (clarityIndex >= 0 && clarityIndex < 8) ? CLARITY_HASHES[clarityIndex] : 0x7fFFffFF;
        
        // Binarize float features - unroll for better performance
        unsigned int binFeatureIndex = 0;
        for (size_t i = 0; i < 6; ++i) {
            const auto& borders = model.FloatFeatureBorders[i];
            if (!borders.empty()) {
                const float floatFeature = features[i];
                const size_t numBorders = borders.size();
                
                // Process 4 borders at a time when possible
                size_t j = 0;
                for (; j + 3 < numBorders; j += 4) {
                    binaryFeaturesBuffer[binFeatureIndex] += (floatFeature > borders[j]);
                    binaryFeaturesBuffer[binFeatureIndex] += (floatFeature > borders[j+1]);
                    binaryFeaturesBuffer[binFeatureIndex] += (floatFeature > borders[j+2]);
                    binaryFeaturesBuffer[binFeatureIndex] += (floatFeature > borders[j+3]);
                }
                
                // Handle remaining borders
                for (; j < numBorders; ++j) {
                    binaryFeaturesBuffer[binFeatureIndex] += (floatFeature > borders[j]);
                }
                ++binFeatureIndex;
            }
        }
        
        // Binarize one-hot categorical features
        if (model.OneHotCatFeatureIndex.size() > 0) {
            const int transposedHash[3] = {hash0, hash1, hash2};
            
            for (unsigned int i = 0; i < model.OneHotCatFeatureIndex.size(); ++i) {
                const auto catIdx = catFeaturePackedIndexes.at(model.OneHotCatFeatureIndex[i]);
                const auto hash = transposedHash[catIdx];
                const auto& hashValues = model.OneHotHashValues[i];
                
                if (!hashValues.empty()) {
                    unsigned char result = 0;
                    for (unsigned int borderIdx = 0; borderIdx < hashValues.size(); ++borderIdx) {
                        result |= (hash == hashValues[borderIdx]) * (borderIdx + 1);
                    }
                    binaryFeaturesBuffer[binFeatureIndex++] = result;
                }
            }
        }
        
        // Handle CTR features if present
        if (model.modelCtrs.UsedModelCtrsCount > 0) {
            // Create temporary vectors for CalcCtrs (required by API)
            std::vector<unsigned char> binaryFeaturesVec(binaryFeaturesBuffer, binaryFeaturesBuffer + model.BinaryFeatureCount);
            std::vector<int> transposedHashVec = {hash0, hash1, hash2};
            std::vector<float> ctrs(model.modelCtrs.UsedModelCtrsCount);
            CalcCtrs(model.modelCtrs, binaryFeaturesVec, transposedHashVec, ctrs);
            
            // Binarize CTR features
            for (size_t i = 0; i < model.CtrFeatureBorders.size(); ++i) {
                const auto& borders = model.CtrFeatureBorders[i];
                const float ctrValue = ctrs[i];
                for (const float border : borders) {
                    binaryFeaturesBuffer[binFeatureIndex] += (ctrValue > border);
                }
                ++binFeatureIndex;
            }
        }
        
        // Tree evaluation with prefetching
        double result = 0.0;
        const auto* leafValuesPtr = model.LeafValues;
        size_t treeSplitsIdx = 0;
        
        for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
            const unsigned int currentTreeDepth = model.TreeDepth[treeId];
            unsigned int index = 0;
            
            
            // Tree traversal
            for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {
                const unsigned char borderVal = model.TreeSplitIdxs[treeSplitsIdx + depth];
                const unsigned int featureIndex = model.TreeSplitFeatureIndex[treeSplitsIdx + depth];
                const unsigned char xorMask = model.TreeSplitXorMask[treeSplitsIdx + depth];
                index |= ((binaryFeaturesBuffer[featureIndex] ^ xorMask) >= borderVal) << depth;
            }
            
            result += leafValuesPtr[index][0];
            leafValuesPtr += 1 << currentTreeDepth;
            treeSplitsIdx += currentTreeDepth;
        }
        
        return static_cast<float>(model.Scale * result + model.Biases[0]);
    }
}