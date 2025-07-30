// Memory-optimized version: Pre-allocate buffers and avoid dynamic allocations
#include "../models/baseline.cpp"

// Pre-allocated buffers to avoid repeated allocations
static thread_local unsigned char binaryFeaturesBuffer[84];
static thread_local float floatFeaturesBuffer[6];
static thread_local int transposedHashBuffer[3];

extern "C" {
    float catboostPredict(const float* features, int featureCount) {
        if (featureCount != 9) {
            return -1.0f;
        }
        
        const struct CatboostModel& model = CatboostModelStatic;
        
        // Copy numeric features to pre-allocated buffer
        for (int i = 0; i < 6; ++i) {
            floatFeaturesBuffer[i] = features[i];
        }
        
        // Convert categorical indices directly to hashes (avoid string creation)
        // Order must match categorical_mappings.json:
        // cut: ["Ideal", "Premium", "Good", "Very Good", "Fair"]
        static const int cutHashes[] = {1754990671, -570237862, 1700310925, 1933222421, 610519841};
        // color: ["E", "I", "J", "H", "F", "G", "D"]
        static const int colorHashes[] = {-1095458675, 1348280313, -472349076, -896563403, -1292729504, 1719715171, -204260682};
        // clarity: ["SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"]
        static const int clarityHashes[] = {-1581449724, 579192095, -1896862659, 2143106594, 88967919, 1708347785, 1353923139, -117150168};
        
        int cutIndex = static_cast<int>(features[6]);
        int colorIndex = static_cast<int>(features[7]);
        int clarityIndex = static_cast<int>(features[8]);
        
        // Direct hash lookup without string conversion
        transposedHashBuffer[0] = (cutIndex >= 0 && cutIndex < 5) ? cutHashes[cutIndex] : 0x7fFFffFF;
        transposedHashBuffer[1] = (colorIndex >= 0 && colorIndex < 7) ? colorHashes[colorIndex] : 0x7fFFffFF;
        transposedHashBuffer[2] = (clarityIndex >= 0 && clarityIndex < 8) ? clarityHashes[clarityIndex] : 0x7fFFffFF;
        
        // Binarize features using pre-allocated buffer
        memset(binaryFeaturesBuffer, 0, model.BinaryFeatureCount);
        unsigned int binFeatureIndex = 0;
        
        // Binarize float features
        for (size_t i = 0; i < model.FloatFeatureBorders.size(); ++i) {
            if (!model.FloatFeatureBorders[i].empty()) {
                const float floatFeature = floatFeaturesBuffer[i];
                for (const float border : model.FloatFeatureBorders[i]) {
                    binaryFeaturesBuffer[binFeatureIndex] += (unsigned char)(floatFeature > border);
                }
                ++binFeatureIndex;
            }
        }
        
        // Binarize one-hot categorical features
        if (model.OneHotCatFeatureIndex.size() > 0) {
            std::unordered_map<int, int> catFeaturePackedIndexes;
            for (unsigned int i = 0; i < model.CatFeatureCount; ++i) {
                catFeaturePackedIndexes[model.CatFeaturesIndex[i]] = i;
            }
            
            for (unsigned int i = 0; i < model.OneHotCatFeatureIndex.size(); ++i) {
                const auto catIdx = catFeaturePackedIndexes.at(model.OneHotCatFeatureIndex[i]);
                const auto hash = transposedHashBuffer[catIdx];
                if (!model.OneHotHashValues[i].empty()) {
                    for (unsigned int borderIdx = 0; borderIdx < model.OneHotHashValues[i].size(); ++borderIdx) {
                        binaryFeaturesBuffer[binFeatureIndex] |=
                            (unsigned char)(hash == model.OneHotHashValues[i][borderIdx]) * (borderIdx + 1);
                    }
                    ++binFeatureIndex;
                }
            }
        }
        
        // Handle CTR features if present
        if (model.modelCtrs.UsedModelCtrsCount > 0) {
            std::vector<float> ctrs(model.modelCtrs.UsedModelCtrsCount);
            std::vector<unsigned char> binaryFeaturesVec(binaryFeaturesBuffer, binaryFeaturesBuffer + model.BinaryFeatureCount);
            std::vector<int> transposedHashVec(transposedHashBuffer, transposedHashBuffer + 3);
            CalcCtrs(model.modelCtrs, binaryFeaturesVec, transposedHashVec, ctrs);
            
            for (size_t i = 0; i < model.CtrFeatureBorders.size(); ++i) {
                for (const float border : model.CtrFeatureBorders[i]) {
                    binaryFeaturesBuffer[binFeatureIndex] += (unsigned char)(ctrs[i] > border);
                }
                ++binFeatureIndex;
            }
        }
        
        // Tree evaluation
        double result = 0.0;
        const auto* leafValuesPtr = model.LeafValues;
        size_t treeSplitsIdx = 0;
        
        for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
            const unsigned int currentTreeDepth = model.TreeDepth[treeId];
            unsigned int index = 0;
            
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