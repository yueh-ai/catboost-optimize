// Fully optimized version: Combines all optimizations
// - Memory pre-allocation
// - Direct categorical hash lookup
// - SIMD vectorization
// - Loop unrolling
// - Optimized tree traversal

#include <cstring>
#include <wasm_simd128.h>

// Include baseline model structures
#include "../models/baseline.cpp"

// Aligned buffers for better SIMD performance
static thread_local unsigned char binaryFeaturesBuffer[96] __attribute__((aligned(16)));  // Padded to 96 for alignment
static thread_local float tempFloatBuffer[8] __attribute__((aligned(16)));

// Pre-computed categorical hash lookup tables
static const int CUT_HASHES[] = {1754990671, -570237862, 1700310925, 1933222421, 610519841};
static const int COLOR_HASHES[] = {-1095458675, 1348280313, -472349076, -896563403, -1292729504, 1719715171, -204260682};
static const int CLARITY_HASHES[] = {-1581449724, 579192095, -1896862659, 2143106594, 88967919, 1708347785, 1353923139, -117150168};

// Pre-compute categorical feature indexes once
static const std::unordered_map<int, int> catFeaturePackedIndexes = []() {
    const struct CatboostModel& model = CatboostModelStatic;
    std::unordered_map<int, int> indexes;
    for (unsigned int i = 0; i < model.CatFeatureCount; ++i) {
        indexes[model.CatFeaturesIndex[i]] = i;
    }
    return indexes;
}();

// Inline helper for tree traversal
inline unsigned int traverseTree(const unsigned char* features, const unsigned short* splitFeatures, 
                               const unsigned char* splitIdxs, const unsigned char* xorMasks, 
                               unsigned int depth) {
    unsigned int index = 0;
    for (unsigned int d = 0; d < depth; ++d) {
        index |= ((features[splitFeatures[d]] ^ xorMasks[d]) >= splitIdxs[d]) << d;
    }
    return index;
}

extern "C" {
    float catboostPredict(const float* features, int featureCount) {
        if (featureCount != 9) {
            return -1.0f;
        }
        
        const struct CatboostModel& model = CatboostModelStatic;
        
        // Clear binary features buffer using SIMD (6 x 16 bytes = 96 bytes)
        v128_t zero = wasm_i32x4_splat(0);
        wasm_v128_store(binaryFeaturesBuffer, zero);
        wasm_v128_store(binaryFeaturesBuffer + 16, zero);
        wasm_v128_store(binaryFeaturesBuffer + 32, zero);
        wasm_v128_store(binaryFeaturesBuffer + 48, zero);
        wasm_v128_store(binaryFeaturesBuffer + 64, zero);
        wasm_v128_store(binaryFeaturesBuffer + 80, zero);
        
        // Get categorical hashes directly
        const int catIndices[3] = {
            static_cast<int>(features[6]),
            static_cast<int>(features[7]),
            static_cast<int>(features[8])
        };
        
        const int hashes[3] = {
            (catIndices[0] >= 0 && catIndices[0] < 5) ? CUT_HASHES[catIndices[0]] : 0x7fFFffFF,
            (catIndices[1] >= 0 && catIndices[1] < 7) ? COLOR_HASHES[catIndices[1]] : 0x7fFFffFF,
            (catIndices[2] >= 0 && catIndices[2] < 8) ? CLARITY_HASHES[catIndices[2]] : 0x7fFFffFF
        };
        
        // Binarize float features using SIMD with aggressive unrolling
        unsigned int binFeatureIndex = 0;
        
        // Process each float feature
        for (size_t i = 0; i < 6; ++i) {
            const auto& borders = model.FloatFeatureBorders[i];
            if (borders.empty()) continue;
            
            const float floatFeature = features[i];
            const v128_t feature_vec = wasm_f32x4_splat(floatFeature);
            const size_t numBorders = borders.size();
            
            unsigned char accumulator = 0;
            size_t j = 0;
            
            // Process 8 borders at a time (2 SIMD operations)
            for (; j + 7 < numBorders; j += 8) {
                // First 4 borders
                v128_t borders_vec1 = wasm_f32x4_make(borders[j], borders[j+1], borders[j+2], borders[j+3]);
                v128_t cmp1 = wasm_f32x4_gt(feature_vec, borders_vec1);
                
                // Next 4 borders
                v128_t borders_vec2 = wasm_f32x4_make(borders[j+4], borders[j+5], borders[j+6], borders[j+7]);
                v128_t cmp2 = wasm_f32x4_gt(feature_vec, borders_vec2);
                
                // Sum up comparisons
                accumulator += __builtin_popcount(wasm_i32x4_bitmask(cmp1));
                accumulator += __builtin_popcount(wasm_i32x4_bitmask(cmp2));
            }
            
            // Process remaining 4 borders if any
            if (j + 3 < numBorders) {
                v128_t borders_vec = wasm_f32x4_make(
                    borders[j], 
                    (j+1 < numBorders) ? borders[j+1] : 0.0f,
                    (j+2 < numBorders) ? borders[j+2] : 0.0f,
                    (j+3 < numBorders) ? borders[j+3] : 0.0f
                );
                v128_t cmp = wasm_f32x4_gt(feature_vec, borders_vec);
                accumulator += __builtin_popcount(wasm_i32x4_bitmask(cmp) & ((1 << (numBorders - j)) - 1));
                j += 4;
            }
            
            // Handle remaining borders
            for (; j < numBorders; ++j) {
                accumulator += (floatFeature > borders[j]);
            }
            
            binaryFeaturesBuffer[binFeatureIndex++] = accumulator;
        }
        
        // Binarize one-hot categorical features (optimized)
        if (model.OneHotCatFeatureIndex.size() > 0) {
            for (unsigned int i = 0; i < model.OneHotCatFeatureIndex.size(); ++i) {
                const auto catIdx = catFeaturePackedIndexes.at(model.OneHotCatFeatureIndex[i]);
                const auto hash = hashes[catIdx];
                const auto& hashValues = model.OneHotHashValues[i];
                
                if (!hashValues.empty()) {
                    unsigned char result = 0;
                    // Unroll common case (usually 2-4 hash values)
                    const size_t numValues = hashValues.size();
                    if (numValues == 2) {
                        result = (hash == hashValues[0]) * 1 + (hash == hashValues[1]) * 2;
                    } else if (numValues == 3) {
                        result = (hash == hashValues[0]) * 1 + (hash == hashValues[1]) * 2 + (hash == hashValues[2]) * 3;
                    } else {
                        for (unsigned int idx = 0; idx < numValues; ++idx) {
                            result |= (hash == hashValues[idx]) * (idx + 1);
                        }
                    }
                    binaryFeaturesBuffer[binFeatureIndex++] = result;
                }
            }
        }
        
        // Handle CTR features if present
        if (model.modelCtrs.UsedModelCtrsCount > 0) {
            std::vector<unsigned char> binaryFeaturesVec(binaryFeaturesBuffer, binaryFeaturesBuffer + model.BinaryFeatureCount);
            std::vector<int> transposedHashVec(hashes, hashes + 3);
            std::vector<float> ctrs(model.modelCtrs.UsedModelCtrsCount);
            CalcCtrs(model.modelCtrs, binaryFeaturesVec, transposedHashVec, ctrs);
            
            // Binarize CTR features
            for (size_t i = 0; i < model.CtrFeatureBorders.size(); ++i) {
                const auto& borders = model.CtrFeatureBorders[i];
                unsigned char acc = 0;
                for (const float border : borders) {
                    acc += (ctrs[i] > border);
                }
                binaryFeaturesBuffer[binFeatureIndex++] = acc;
            }
        }
        
        // Optimized tree evaluation
        double result = 0.0;
        const auto* leafValuesPtr = model.LeafValues;
        size_t treeSplitsIdx = 0;
        
        // Process trees - most are depth 6
        for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
            const unsigned int depth = model.TreeDepth[treeId];
            unsigned int index = 0;
            
            // Specialized handling for depth 6 (most common)
            if (depth == 6) {
                // Fully unrolled for depth 6
                const auto* splits = &model.TreeSplitIdxs[treeSplitsIdx];
                const auto* features = &model.TreeSplitFeatureIndex[treeSplitsIdx];
                const auto* masks = &model.TreeSplitXorMask[treeSplitsIdx];
                
                index = ((binaryFeaturesBuffer[features[0]] ^ masks[0]) >= splits[0]) |
                       (((binaryFeaturesBuffer[features[1]] ^ masks[1]) >= splits[1]) << 1) |
                       (((binaryFeaturesBuffer[features[2]] ^ masks[2]) >= splits[2]) << 2) |
                       (((binaryFeaturesBuffer[features[3]] ^ masks[3]) >= splits[3]) << 3) |
                       (((binaryFeaturesBuffer[features[4]] ^ masks[4]) >= splits[4]) << 4) |
                       (((binaryFeaturesBuffer[features[5]] ^ masks[5]) >= splits[5]) << 5);
            } else if (depth == 5) {
                // Unrolled for depth 5
                const auto* splits = &model.TreeSplitIdxs[treeSplitsIdx];
                const auto* features = &model.TreeSplitFeatureIndex[treeSplitsIdx];
                const auto* masks = &model.TreeSplitXorMask[treeSplitsIdx];
                
                index = ((binaryFeaturesBuffer[features[0]] ^ masks[0]) >= splits[0]) |
                       (((binaryFeaturesBuffer[features[1]] ^ masks[1]) >= splits[1]) << 1) |
                       (((binaryFeaturesBuffer[features[2]] ^ masks[2]) >= splits[2]) << 2) |
                       (((binaryFeaturesBuffer[features[3]] ^ masks[3]) >= splits[3]) << 3) |
                       (((binaryFeaturesBuffer[features[4]] ^ masks[4]) >= splits[4]) << 4);
            } else {
                // General case for other depths
                index = traverseTree(binaryFeaturesBuffer, 
                                   &model.TreeSplitFeatureIndex[treeSplitsIdx],
                                   &model.TreeSplitIdxs[treeSplitsIdx],
                                   &model.TreeSplitXorMask[treeSplitsIdx],
                                   depth);
            }
            
            result += leafValuesPtr[index][0];
            leafValuesPtr += 1 << depth;
            treeSplitsIdx += depth;
        }
        
        return static_cast<float>(model.Scale * result + model.Biases[0]);
    }
}