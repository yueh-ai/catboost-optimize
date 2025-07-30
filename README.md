# CatBoost WebWorker Experimentation Framework

A streamlined framework for optimizing CatBoost models running in WebAssembly (WASM) within web workers. Test different C++ optimization strategies and measure their impact on performance and accuracy.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Train model and generate test data (one-time setup)
cd setup
python train_model.py
python generate_test_data.py
cd ..

# 3. Run an experiment
./framework/experiment.sh experiments/baseline_wrapper.cpp
```

## Features

- **One-command testing**: Modify C++ code, run one command, get comprehensive results
- **Performance metrics**: Predictions per second, batch size optimization, memory usage
- **Accuracy validation**: Compare against ground truth with detailed error analysis
- **Visual reports**: Automatic generation of performance plots and summaries
- **WebWorker ready**: Test real-world WASM performance in browser environments

## Project Structure

```
catboost-optimize/
├── setup/                      # Model training and test data generation
│   ├── train_model.py         # Train CatBoost on Diamonds dataset
│   └── generate_test_data.py  # Generate 1M test vectors
├── models/                     # Model files and test data
│   ├── baseline.cpp           # CatBoost C++ export (raw model)
│   ├── baseline.cbm           # Native CatBoost model (ground truth)
│   └── test_data.bin          # 1M test vectors (binary)
├── experiments/               # Your optimized models go here
│   ├── baseline_wrapper.cpp   # Wrapper for baseline model with catboostPredict
│   └── example_simd.cpp      # Example optimization
├── framework/                  # Core experimentation tools
│   ├── experiment.sh          # Main experiment runner
│   ├── compile_wasm.js        # Emscripten wrapper
│   ├── run_predictions.js     # WASM execution engine
│   └── accuracy_checker.py    # Accuracy validation
└── results/                   # Experiment outputs
    └── experiment_*.json      # Detailed results
```

## Usage Examples

### Basic Experiment

```bash
# Test your optimized model (make sure it exports catboostPredict function)
./framework/experiment.sh experiments/my_optimized_model.cpp
```

### Advanced Options

```bash
# Test different batch sizes
./framework/experiment.sh my_model.cpp --batch-sizes=1,10,100,1000,10000

# Custom compilation flags
./framework/experiment.sh my_model.cpp --em-flags="-O3 -msimd128"

# Named experiment
./framework/experiment.sh my_model.cpp --name=simd_v2_test
```

### Visualize Results

```bash
# Generate performance plots
python framework/visualize_results.py results/experiment_20240120_153045_report.json
```

## Writing Optimizations

1. Start with the baseline wrapper in `experiments/baseline_wrapper.cpp`
2. Create your optimized version in `experiments/`
3. Ensure your model exports: `extern "C" float catboostPredict(const float* features, int featureCount)`
4. Common optimizations to try:
   - SIMD vectorization
   - Loop unrolling
   - Memory access patterns
   - Batch processing
   - Precision adjustments

Example structure:
```cpp
extern "C" {
    float catboostPredict(const float* features, int featureCount) {
        // Your optimized prediction logic here
    }
}
```

## Understanding Results

### Performance Metrics
- **Predictions/second**: Primary performance metric
- **Best batch size**: Optimal batch size for your implementation
- **Speedup vs baseline**: Performance improvement factor

### Accuracy Metrics
- **Max absolute error**: Largest prediction difference
- **Mean absolute error**: Average prediction difference
- **RMSE**: Root mean square error
- **Exact matches**: Percentage of bit-exact predictions

### Example Report
```json
{
  "performance": {
    "predictions_per_second": 1176470,
    "speedup_vs_baseline": 3.2
  },
  "accuracy": {
    "max_absolute_error": 0.0001,
    "exact_matches_ratio": 0.95
  }
}
```

## Environment Requirements

- **Python**: 3.10+ with uv package manager
- **Node.js**: 20.x LTS (minimum 18.3)
- **Emscripten**: 4.0.9+ (configured at `/Users/yuehu/opensources/emsdk`)

## Troubleshooting

### Emscripten not found
Update the path in `framework/experiment.sh`:
```bash
source /path/to/your/emsdk/emsdk_env.sh
```

### Test data not generated
```bash
cd setup
python train_model.py
python generate_test_data.py
```

### Compilation errors
Check your C++ syntax and ensure the exported function signature matches:
```cpp
extern "C" float catboostPredict(const float* features, int featureCount)
```

## Contributing

1. Test your optimizations thoroughly
2. Document any new compilation flags or techniques
3. Share successful optimization strategies

## License

This experimentation framework is provided as-is for research and optimization purposes.