# Developer Guide

This guide provides detailed information for developing and optimizing CatBoost WASM implementations.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Creating Custom Wrappers](#creating-custom-wrappers)
3. [Optimization Strategies](#optimization-strategies)
4. [Running Experiments](#running-experiments)
5. [Framework Internals](#framework-internals)
6. [Troubleshooting](#troubleshooting)

## Architecture Overview

The framework consists of three main components:

### 1. C++ Model and Wrapper Layer
- **models/baseline.cpp**: Auto-generated CatBoost model exported to C++
- **experiments/batch_wrapper.cpp**: Wrapper providing batch processing APIs

### 2. Experiment Runner
- **experiment_runner.js**: Node.js application that:
  - Compiles C++ to WASM using Emscripten
  - Spawns Web Workers for parallel testing
  - Measures performance and accuracy
  - Generates comprehensive reports

### 3. Test Infrastructure
- **test_data/**: Binary test data with ground truth
- **run_experiment.sh**: User-friendly CLI interface

## Creating Custom Wrappers

### Basic Wrapper Structure

```cpp
#include "../models/baseline.cpp"

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    double catboostPredict(const float* features, int featureCount) {
        // Convert features and call model
        return ApplyCatboostModel(floatFeatures, catFeatures);
    }
    
    EMSCRIPTEN_KEEPALIVE
    void catboostPredictBatch(const float* features, double* predictions, 
                              int batchSize, int featuresPerSample) {
        // Process batch of samples
    }
}
```

### Key Optimization Opportunities

1. **Batch Processing**: Process multiple samples in single call
2. **Memory Layout**: Optimize for cache locality
3. **Feature Preprocessing**: Minimize categorical conversions
4. **SIMD Operations**: Use vector instructions for parallel processing

## Optimization Strategies

### 1. Compiler Optimizations

```bash
# Basic optimization
./run_experiment.sh --emflags "-O3"

# Link-time optimization
./run_experiment.sh --emflags "-O3 -flto"

# SIMD support
./run_experiment.sh --simd --emflags "-O3 -msimd128"

# Threading (experimental)
./run_experiment.sh --threads --emflags "-O3 -pthread"
```

### 2. Batch Size Tuning

Test different batch sizes to find optimal memory/performance balance:

```bash
./run_experiment.sh --batch-sizes "1 10 100 1000 10000"
```

### 3. Custom Wrapper Optimizations

Example optimizations for batch_wrapper.cpp:

```cpp
// Pre-allocate vectors to avoid reallocation
std::vector<float> floatFeatures(6);
std::vector<std::string> catFeatures(3);

// Process entire batch with minimal allocations
for (int i = 0; i < batchSize; i++) {
    // Reuse vectors, update values only
    UpdateFeatures(features + i * featuresPerSample);
    predictions[i] = ApplyCatboostModel(floatFeatures, catFeatures);
}
```

## Running Experiments

### Command Line Options

```bash
./run_experiment.sh [OPTIONS]

Options:
  -h, --help                Show help message
  -n, --name NAME          Experiment name (default: auto-generated)
  -w, --wrapper FILE       Path to wrapper C++ file
  -e, --emflags FLAGS      Emscripten compiler flags
  -b, --batch-sizes SIZES  Space-separated batch sizes
  -d, --data FILE          Test data file
  -o, --output DIR         Output directory
  --simd                   Enable SIMD optimizations
  --threads                Enable threading support
  --no-batch-api          Disable batch API usage
```

### Analyzing Results

Results JSON structure:
```json
{
  "experimentName": "test_final",
  "config": {...},
  "batchResults": [
    {
      "batchSize": 1000,
      "totalTime": 9.94,
      "predictionsPerSecond": 100634,
      "accuracy": 0.95
    }
  ],
  "optimal": {
    "batchSize": 1000,
    "predictionsPerSecond": 100634,
    "speedupVsBaseline": 1.25
  }
}
```

## Framework Internals

### Emscripten Configuration

The framework automatically configures Emscripten at `/Users/yuehu/opensources/emsdk`.

Key compilation settings:
- `EXPORTED_FUNCTIONS`: C++ functions accessible from JavaScript
- `EXPORTED_RUNTIME_METHODS`: WASM memory access functions
- `ALLOW_MEMORY_GROWTH`: Dynamic memory allocation
- `MODULARIZE`: Creates importable ES6 module

### Worker Communication

Workers communicate via message passing:
1. `init`: Load WASM module and test data
2. `run`: Execute predictions
3. `progress`: Report completion status
4. `complete`: Return results

### Memory Management

- Features are copied to WASM heap using typed arrays
- Batch processing minimizes allocation overhead
- Memory is freed after each batch

## Troubleshooting

### Common Issues

1. **Emscripten not found**
   - Ensure Emscripten is installed at `/Users/yuehu/opensources/emsdk`
   - Run `source /Users/yuehu/opensources/emsdk/emsdk_env.sh`

2. **Low accuracy (0%)**
   - Check float precision differences between Python and C++
   - Verify categorical encoding matches training data
   - Compare with baseline model predictions

3. **Performance issues**
   - Test different batch sizes
   - Enable compiler optimizations
   - Profile memory allocation patterns

### Debugging Tips

1. Add console logging to wrapper:
   ```cpp
   EM_ASM({
       console.log('Processing batch:', $0);
   }, batchSize);
   ```

2. Check WASM module properties:
   ```javascript
   console.log('HEAP size:', moduleInstance.HEAP8.length);
   console.log('Functions:', Object.keys(moduleInstance));
   ```

3. Monitor memory usage:
   ```bash
   # In experiment_runner.js
   console.log('Memory:', process.memoryUsage());
   ```

## Advanced Topics

### Model Export Settings

When exporting from CatBoost:
```python
model.save_model('model.cbm', format='cbm')
model.save_model('model.cpp', format='cpp', 
                 export_parameters={'cat_features_count': 3})
```

### Custom Memory Allocators

For better performance with large batches:
```cpp
// Pre-allocate memory pool
static float* featurePool = nullptr;
static size_t poolSize = 0;

void initializePool(size_t maxBatchSize) {
    poolSize = maxBatchSize * featuresPerSample;
    featurePool = (float*)malloc(poolSize * sizeof(float));
}
```

### SIMD Optimizations

Example using WebAssembly SIMD:
```cpp
#include <wasm_simd128.h>

void processFeaturesSIMD(const float* input, float* output, int count) {
    for (int i = 0; i < count; i += 4) {
        v128_t vec = wasm_v128_load(&input[i]);
        // SIMD operations
        wasm_v128_store(&output[i], vec);
    }
}
```

## Contributing

1. Create a new wrapper in `experiments/`
2. Test with various configurations
3. Document optimization techniques
4. Submit results and analysis

For questions or issues, please open a GitHub issue.