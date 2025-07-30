// SIMD-optimized version: Use WebAssembly SIMD128 for vectorized operations
#include <cstring>
#include <wasm_simd128.h>

// Include baseline model structures
#include "../models/baseline.cpp"

// Pre-allocated buffers
static thread_local unsigned char binaryFeaturesBuffer[84] __attribute__((aligned(16)));
static thread_local float floatFeaturesBuffer[8] __attribute__((aligned(16)));  // Padded to 8 for SIMD

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
        
        // Clear binary features buffer using SIMD
        v128_t zero = wasm_i32x4_splat(0);
        for (int i = 0; i < 84; i += 16) {
            wasm_v128_store(&binaryFeaturesBuffer[i], zero);
        }
        
        // Copy float features to aligned buffer
        memcpy(floatFeaturesBuffer, features, 6 * sizeof(float));
        floatFeaturesBuffer[6] = 0.0f;  // Padding
        floatFeaturesBuffer[7] = 0.0f;  // Padding
        
        // Get categorical indices
        const int cutIndex = static_cast<int>(features[6]);
        const int colorIndex = static_cast<int>(features[7]);
        const int clarityIndex = static_cast<int>(features[8]);
        
        // Direct hash lookup
        const int hash0 = (cutIndex >= 0 && cutIndex < 5) ? CUT_HASHES[cutIndex] : 0x7fFFffFF;
        const int hash1 = (colorIndex >= 0 && colorIndex < 7) ? COLOR_HASHES[colorIndex] : 0x7fFFffFF;
        const int hash2 = (clarityIndex >= 0 && clarityIndex < 8) ? CLARITY_HASHES[clarityIndex] : 0x7fFFffFF;
        
        // Binarize float features using SIMD
        unsigned int binFeatureIndex = 0;
        for (size_t i = 0; i < 6; ++i) {
            const auto& borders = model.FloatFeatureBorders[i];
            if (!borders.empty()) {
                const float floatFeature = floatFeaturesBuffer[i];
                v128_t feature_vec = wasm_f32x4_splat(floatFeature);
                
                size_t j = 0;
                unsigned char accumulator = 0;
                
                // Process 4 borders at a time using SIMD
                for (; j + 3 < borders.size(); j += 4) {
                    v128_t borders_vec = wasm_f32x4_make(borders[j], borders[j+1], borders[j+2], borders[j+3]);
                    v128_t cmp = wasm_f32x4_gt(feature_vec, borders_vec);
                    
                    // Extract comparison results and add to accumulator
                    accumulator += (wasm_i32x4_extract_lane(cmp, 0) & 1);
                    accumulator += (wasm_i32x4_extract_lane(cmp, 1) & 1);
                    accumulator += (wasm_i32x4_extract_lane(cmp, 2) & 1);
                    accumulator += (wasm_i32x4_extract_lane(cmp, 3) & 1);
                }
                
                // Handle remaining borders
                for (; j < borders.size(); ++j) {
                    accumulator += (floatFeature > borders[j]);
                }
                
                binaryFeaturesBuffer[binFeatureIndex++] = accumulator;
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
            std::vector<unsigned char> binaryFeaturesVec(binaryFeaturesBuffer, binaryFeaturesBuffer + model.BinaryFeatureCount);
            std::vector<int> transposedHashVec = {hash0, hash1, hash2};
            std::vector<float> ctrs(model.modelCtrs.UsedModelCtrsCount);
            CalcCtrs(model.modelCtrs, binaryFeaturesVec, transposedHashVec, ctrs);
            
            // Binarize CTR features using SIMD where possible
            for (size_t i = 0; i < model.CtrFeatureBorders.size(); ++i) {
                const auto& borders = model.CtrFeatureBorders[i];
                const float ctrValue = ctrs[i];
                v128_t ctr_vec = wasm_f32x4_splat(ctrValue);
                
                size_t j = 0;
                unsigned char accumulator = 0;
                
                // Process 4 borders at a time
                for (; j + 3 < borders.size(); j += 4) {
                    v128_t borders_vec = wasm_f32x4_make(borders[j], borders[j+1], borders[j+2], borders[j+3]);
                    v128_t cmp = wasm_f32x4_gt(ctr_vec, borders_vec);
                    
                    accumulator += (wasm_i32x4_extract_lane(cmp, 0) & 1);
                    accumulator += (wasm_i32x4_extract_lane(cmp, 1) & 1);
                    accumulator += (wasm_i32x4_extract_lane(cmp, 2) & 1);
                    accumulator += (wasm_i32x4_extract_lane(cmp, 3) & 1);
                }
                
                // Handle remaining borders
                for (; j < borders.size(); ++j) {
                    accumulator += (ctrValue > borders[j]);
                }
                
                binaryFeaturesBuffer[binFeatureIndex++] = accumulator;
            }
        }
        
        // Tree evaluation with loop unrolling
        double result = 0.0;
        const auto* leafValuesPtr = model.LeafValues;
        size_t treeSplitsIdx = 0;
        
        // Most trees have depth 6, optimize for this common case
        for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
            const unsigned int currentTreeDepth = model.TreeDepth[treeId];
            unsigned int index = 0;
            
            if (currentTreeDepth == 6) {
                // Unrolled loop for depth 6 (most common)
                index |= ((binaryFeaturesBuffer[model.TreeSplitFeatureIndex[treeSplitsIdx]] ^ model.TreeSplitXorMask[treeSplitsIdx]) >= model.TreeSplitIdxs[treeSplitsIdx]) << 0;
                index |= ((binaryFeaturesBuffer[model.TreeSplitFeatureIndex[treeSplitsIdx+1]] ^ model.TreeSplitXorMask[treeSplitsIdx+1]) >= model.TreeSplitIdxs[treeSplitsIdx+1]) << 1;
                index |= ((binaryFeaturesBuffer[model.TreeSplitFeatureIndex[treeSplitsIdx+2]] ^ model.TreeSplitXorMask[treeSplitsIdx+2]) >= model.TreeSplitIdxs[treeSplitsIdx+2]) << 2;
                index |= ((binaryFeaturesBuffer[model.TreeSplitFeatureIndex[treeSplitsIdx+3]] ^ model.TreeSplitXorMask[treeSplitsIdx+3]) >= model.TreeSplitIdxs[treeSplitsIdx+3]) << 3;
                index |= ((binaryFeaturesBuffer[model.TreeSplitFeatureIndex[treeSplitsIdx+4]] ^ model.TreeSplitXorMask[treeSplitsIdx+4]) >= model.TreeSplitIdxs[treeSplitsIdx+4]) << 4;
                index |= ((binaryFeaturesBuffer[model.TreeSplitFeatureIndex[treeSplitsIdx+5]] ^ model.TreeSplitXorMask[treeSplitsIdx+5]) >= model.TreeSplitIdxs[treeSplitsIdx+5]) << 5;
            } else {
                // General case for other depths
                for (unsigned int depth = 0; depth < currentTreeDepth; ++depth) {
                    const unsigned char borderVal = model.TreeSplitIdxs[treeSplitsIdx + depth];
                    const unsigned int featureIndex = model.TreeSplitFeatureIndex[treeSplitsIdx + depth];
                    const unsigned char xorMask = model.TreeSplitXorMask[treeSplitsIdx + depth];
                    index |= ((binaryFeaturesBuffer[featureIndex] ^ xorMask) >= borderVal) << depth;
                }
            }
            
            result += leafValuesPtr[index][0];
            leafValuesPtr += 1 << currentTreeDepth;
            treeSplitsIdx += currentTreeDepth;
        }
        
        return static_cast<float>(model.Scale * result + model.Biases[0]);
    }
}