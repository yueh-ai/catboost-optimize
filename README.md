# CatBoost WASM Optimization Framework

A streamlined framework for optimizing CatBoost model inference in WebAssembly. Process 1M samples at once and measure throughput.

## Quick Start

```bash
# Run with default settings
./run_experiment.sh

# Run with custom wrapper
./run_experiment.sh --wrapper ./my_wrapper.cpp --name my_test

# Enable SIMD optimizations
./run_experiment.sh --simd --name simd_test

# Use aggressive optimization flags
./run_experiment.sh --emflags "-O3 -flto" --name aggressive_opt
```

## Current Performance

- **Baseline**: ~100,000 samples/second
- **Test Data**: 1,000,000 samples
- **Features**: 9 total (6 float + 3 categorical)
- **Model**: 999 decision trees

## Architecture Overview

This project uses a simplified architecture focused purely on throughput optimization:

1. **Node.js Coordinator**: `experiment_runner.js` manages the experiment lifecycle
2. **Worker Thread**: `experiments/worker.js` runs the WASM module in isolation
3. **WASM Module**: Compiled from `wrapper.cpp` to process all samples
4. **Binary Test Data**: Pre-generated 1M samples in binary format

## Project Structure

```
catboost-optimize/
├── models/
│   ├── baseline.cpp      # The CatBoost model (999 trees)
│   ├── baseline.cbm      # Original CatBoost model file
│   └── test_data.bin     # Binary test data (1M samples)
├── test_data/
│   └── test_data_1M.bin  # Binary format test data (44MB)
├── experiments/
│   ├── wrapper.cpp       # ⭐ OPTIMIZATION TARGET - modify this!
│   └── worker.js         # Node.js worker for running experiments
├── experiment_results/   # Output directory for results
├── run_experiment.sh     # Main experiment runner script
├── experiment_runner.js  # Node.js experiment coordinator
├── package.json         # Node.js dependencies
└── pyproject.toml       # Python dependencies for data generation
```

## How It Works

1. **Compilation**: The wrapper.cpp is compiled to WASM using Emscripten with specified optimization flags
2. **Data Loading**: Binary test data (1M samples) is loaded into WASM memory using a custom format
3. **Prediction**: The `catboostPredictAll` function processes all samples in a single batch
4. **Measurement**: Worker thread measures execution time and calculates throughput

### Binary Data Format

The test data uses a custom binary format:
- Magic number: `0xCAFEBABE`
- Version and metadata in header
- All features stored as float32 values
- Sequential layout for efficient memory access

## Optimization Target

The main optimization target is the `catboostPredictAll` function in `experiments/wrapper.cpp`:

```cpp
void catboostPredictAll(
    const float* inputData,   // All input data (flattened)
    double* predictions,      // Output predictions
    int numSamples,          // Number of samples (1M)
    int numFloatFeatures,    // Number of float features (6)
    int numCatFeatures       // Number of categorical features (3)
)
```

### Input Data Format

Features are stored sequentially for each sample:
- Features 0-5: Float features (carat, depth, table, x, y, z)
- Features 6-8: Categorical features encoded as floats (cut, color, clarity)

### Categorical Mappings

- **cut**: 5 values (Fair=0, Good=1, Very Good=2, Premium=3, Ideal=4)
- **color**: 7 values (J=0, I=1, H=2, G=3, F=4, E=5, D=6)
- **clarity**: 8 values (I1=0, SI2=1, SI1=2, VS2=3, VS1=4, VVS2=5, VVS1=6, IF=7)

## Optimization Ideas

1. **Minimize String Operations**
   - Pre-compute categorical mappings
   - Use integer lookups instead of string comparisons

2. **Vectorize Tree Traversal**
   - Process multiple samples through each tree simultaneously
   - Use SIMD for comparisons and accumulation

3. **Memory Layout Optimization**
   - Reorganize data for better cache usage
   - Consider structure-of-arrays instead of array-of-structures

4. **Tree Structure Optimization**
   - Flatten tree nodes for sequential access
   - Pre-compute common paths

5. **WASM-Specific Optimizations**
   - Use SIMD128 instructions (`--simd` flag)
   - Minimize function calls
   - Optimize for WASM's execution model

## Running Experiments

```bash
# Baseline performance
./run_experiment.sh --name baseline

# With SIMD enabled
./run_experiment.sh --name simd_test --simd

# Custom optimization
./run_experiment.sh --name my_opt --wrapper ./my_optimized_wrapper.cpp

# View results
cat experiment_results/my_opt/results.json
```

### Result Format

Experiments output a simple JSON with throughput metrics:

```json
{
  "totalTime": 9.932,
  "throughput": 100695.37,
  "numSamples": 1000000,
  "avgTimePerSample": 0.0000099
}
```

## Prerequisites

- **Node.js** (v14+)
- **Emscripten SDK** (expected at `/Users/yuehu/opensources/emsdk`)
- **Python with uv** package manager (for data generation)

### Node.js Dependencies

- `yargs` - Command-line argument parsing

### Python Dependencies (via uv)

- `catboost` - For model training and conversion
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - ML utilities
- `matplotlib`, `seaborn` - Visualization (if needed)

## Recent Experiments

Recent test runs have achieved consistent baseline performance:
- `test_final2`: ~100,695 samples/second
- Various optimization attempts in `test_run` through `test_run7`

## Tips

- Start by profiling where time is spent (tree traversal, feature extraction, etc.)
- The model has 999 trees - consider batching tree evaluations
- Categorical features are currently converted to strings on every prediction
- The baseline implementation is intentionally naive to leave room for optimization
- Use Node.js worker threads to ensure clean isolation between runs

Good luck optimizing! 🚀