# CatBoost WASM Optimization Framework - User Guide

## Overview

This framework helps you benchmark and optimize CatBoost models compiled to WebAssembly (WASM). It measures performance, memory usage, and accuracy across different batch sizes and compiler optimizations.

## Quick Start

### 1. Prerequisites

- **Node.js 14+**: Required for running the JavaScript components
- **Python 3.8+**: Required for model training and analysis
- **Emscripten SDK**: Required for compiling C++ to WASM
  - Default location: `/Users/yuehu/opensources/emsdk`
  - To use a different location: `export EMSDK_PATH=/your/path/to/emsdk`

### 2. Initial Setup

If this is your first time using the framework, generate the test data:

```bash
cd setup
python train_model.py        # Trains a baseline CatBoost model
python generate_test_data.py # Creates 1M test samples
cd ..
```

This creates:
- `models/baseline.cpp`: The C++ model file
- `models/test_data.bin`: Binary test data (1M samples)
- `models/model_metadata.json`: Model information

### 3. Running Experiments

Basic usage:
```bash
./framework/experiment.sh models/baseline.cpp
```

With custom options:
```bash
# Test different batch sizes
./framework/experiment.sh models/baseline.cpp --batch-sizes=1,50,500,5000

# Enable SIMD optimizations
./framework/experiment.sh models/baseline.cpp --em-flags="-O3 -msimd128"

# Custom experiment name
./framework/experiment.sh models/baseline.cpp --name=my_experiment

# All options combined
./framework/experiment.sh models/baseline.cpp \
  --batch-sizes=1,100,1000 \
  --em-flags="-O3 -msimd128 -mrelaxed-simd" \
  --name=simd_optimization_test
```

### 4. Understanding the Output

The experiment runs through 5 steps:

1. **Compilation**: Converts C++ model to WASM
   - Shows JS and WASM file sizes
   
2. **Loading Test Data**: Loads 1M samples into memory

3. **Running Predictions**: Tests each batch size
   - Shows progress and predictions/second
   
4. **Checking Accuracy**: Compares WASM output to ground truth
   - Reports max error and mean relative error
   
5. **Generating Report**: Creates comprehensive JSON report

### 5. Analyzing Results

Results are saved in the `results/` directory:
- `<name>.js/.wasm`: Compiled WebAssembly files
- `<name>_predictions.json`: Raw prediction timings
- `<name>_accuracy.json`: Accuracy metrics
- `<name>_report.json`: Final report with all metrics

Key metrics to look for:
- **Speed**: Predictions per second for each batch size
- **Memory**: Peak memory usage during execution
- **Accuracy**: Maximum absolute error (should be < 1e-5 for lossless conversion)
- **File Size**: WASM binary size

## Common Optimization Strategies

### 1. Compiler Flags

```bash
# Maximum performance (default)
--em-flags="-O3"

# Enable SIMD (if your model supports it)
--em-flags="-O3 -msimd128"

# Relaxed SIMD for more optimizations
--em-flags="-O3 -msimd128 -mrelaxed-simd"

# Optimize for size
--em-flags="-Os"

# Balance size and speed
--em-flags="-O2"
```

### 2. Batch Size Selection

- **Batch size 1**: For real-time, single predictions
- **Batch size 10-100**: For small batches with low latency
- **Batch size 1000+**: For bulk processing with high throughput

### 3. Memory Optimization

The framework automatically tracks:
- Peak memory usage
- Memory per prediction
- WASM heap utilization

## Troubleshooting

### "Emscripten not found"
```bash
# Set the correct path
export EMSDK_PATH=/path/to/your/emsdk

# Or install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
```

### "Test data not found"
The framework will automatically generate test data if missing. You can also manually regenerate:
```bash
cd setup
python generate_test_data.py
```

### High accuracy errors
If you see accuracy errors > 1e-5:
1. Check if your model uses features not supported in the C++ export
2. Try different compiler flags (avoid aggressive optimizations)
3. Verify the test data matches your model's training distribution

## Advanced Usage

### Custom Models

To test your own CatBoost model:

1. Train your model in Python:
```python
import catboost

model = catboost.CatBoostRegressor()
model.fit(X_train, y_train)
model.save_model('my_model.cbm')
```

2. Export to C++:
```python
model.save_model('my_model.cpp', format='cpp')
```

3. Run experiments:
```bash
./framework/experiment.sh my_model.cpp --name=my_model_test
```

### Comparing Multiple Configurations

Run experiments with different settings and compare results:

```bash
# Baseline
./framework/experiment.sh models/baseline.cpp --name=baseline

# With SIMD
./framework/experiment.sh models/baseline.cpp \
  --em-flags="-O3 -msimd128" \
  --name=with_simd

# Size optimized
./framework/experiment.sh models/baseline.cpp \
  --em-flags="-Os" \
  --name=size_optimized
```

Then compare the JSON reports in the `results/` directory.

### Environment Variables

- `EMSDK_PATH`: Path to Emscripten SDK (default: `/Users/yuehu/opensources/emsdk`)

## Help

For detailed help on all options:
```bash
./framework/experiment.sh --help
```

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up environment (if needed)
export EMSDK_PATH=/path/to/emsdk

# 2. Generate test data (first time only)
cd setup
python train_model.py
python generate_test_data.py
cd ..

# 3. Run baseline experiment
./framework/experiment.sh models/baseline.cpp --name=baseline

# 4. Try SIMD optimization
./framework/experiment.sh models/baseline.cpp \
  --em-flags="-O3 -msimd128" \
  --name=simd_test

# 5. Test different batch sizes for production
./framework/experiment.sh models/baseline.cpp \
  --batch-sizes=1,10,100,1000,10000 \
  --name=batch_size_analysis

# 6. Compare results
ls -la results/*.json
```

## Performance Tips

1. **Start with default settings** to establish a baseline
2. **Test one optimization at a time** to understand its impact
3. **Consider your use case** when selecting batch sizes
4. **Monitor memory usage** for resource-constrained environments
5. **Verify accuracy** after each optimization

## Next Steps

- Experiment with different compiler flags
- Test various batch sizes for your use case
- Try the framework with your own CatBoost models
- Analyze the JSON reports to find the optimal configuration

Happy optimizing!