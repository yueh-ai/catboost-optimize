# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CatBoost WebWorker Experimentation Framework - A system for optimizing CatBoost models running in WebAssembly within web workers. The framework automates testing of C++ optimization strategies and measures their impact on performance and accuracy.

The project uses the Diamonds dataset (53,940 samples, 10 features) to train a CatBoost regression model predicting diamond prices. The framework generates 1M test vectors for consistent benchmarking across experiments.

## Key Commands

### Setup and Dependencies
```bash
# Install dependencies using uv (Python 3.12+ required)
uv sync

# Initial setup: train model and generate test data
cd setup
python train_model.py
python generate_test_data.py
cd ..
```

### Running Experiments

```bash
# Basic experiment run - processes all 1M inputs at once
./framework/experiment.sh experiments/baseline_wrapper.cpp

# With custom compilation flags
./framework/experiment.sh experiments/my_model.cpp --em-flags="-O3 -msimd128"

# Named experiment
./framework/experiment.sh experiments/my_model.cpp --name=optimized_v1
```

### Visualizing Results
```bash
python framework/visualize_results.py results/experiment_*_report.json
```

## Architecture Overview

The codebase follows a clear separation of concerns:

1. **Setup Phase** (`setup/`): One-time model training and test data generation
   - `train_model.py` trains CatBoost on Diamonds dataset from seaborn
   - `generate_test_data.py` creates 1M test vectors in binary format with fixed seed

2. **Experimentation Pipeline** (`framework/`):
   - `experiment.sh` orchestrates the entire workflow (compile → run → measure → report)
   - `run.js` loads and processes all 1M samples at once
   - `worker.js` executes predictions in WebWorker environment
   - `report.py` generates performance report with accuracy metrics
   - `compile_wasm.js` handles Emscripten compilation
   - `visualize_results.py` generates performance plots

3. **Model Storage** (`models/`):
   - `baseline.cpp` - Original CatBoost C++ export (raw model without wrapper)
   - `baseline.cbm` - Native CatBoost model (ground truth)
   - `test_data.bin` - 1M test vectors in binary format
   - `model_metadata.json` - Feature information and ranges

4. **Experiments** (`experiments/`):
   - `baseline_wrapper.cpp` - Wrapper for baseline.cpp that exports required catboostPredict function
   - Your optimized models go here

5. **Results** (`results/`):
   - JSON reports with performance metrics, accuracy analysis, and memory usage
   - Baseline results stored for automatic speedup comparison

## Important Implementation Details

- **C++ Function Signature**: All models must export:
  ```cpp
  extern "C" float catboostPredict(const float* features, int featureCount)
  ```

- **Emscripten Path**: Hardcoded at `/Users/yuehu/opensources/emsdk` in `framework/experiment.sh:74`

- **Test Data Format**: Binary format with 1M samples, 9 features each (float32) - 6 numeric + 3 categorical encoded as indices

- **Performance Metrics**: Primary metric is predictions/second, with automatic speedup calculation vs baseline

- **Accuracy Validation**: Validates first 1000 predictions with metrics (max error, mean error, RMSE)

- **WebWorker Execution**: Tests run in actual WebWorker environment using Node.js worker_threads

- **Framework Mode**: Processes all 1M inputs in a single batch for focused optimization

## Development Workflow

When implementing optimizations:
1. Start with `experiments/baseline_wrapper.cpp` as reference (NOT `models/baseline.cpp` directly)
2. Create optimized version in `experiments/` directory
3. Ensure your model exports the required function: `extern "C" float catboostPredict(const float* features, int featureCount)`
4. Common optimizations: SIMD vectorization, loop unrolling, memory access patterns, batch processing
5. Run experiment using `./framework/experiment.sh experiments/your_model.cpp`
6. Results include predictions/second, accuracy metrics, and memory usage
7. Use `visualize_results.py` to generate performance plots

## Environment Requirements

- **Python**: 3.10+ (tested with 3.12)
- **Node.js**: 20.x LTS (minimum 18.3 for Emscripten 4.0)
- **Emscripten**: 4.0.9+ 
- **Python packages**: catboost>=1.2.8, numpy>=2.3.1, pandas>=2.3.1, matplotlib>=3.10.3, scikit-learn>=1.7.1, seaborn>=0.13.2