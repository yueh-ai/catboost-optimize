# CatBoost WebWorker Experimentation Framework

## Project Overview

This project provides a streamlined experimentation framework for optimizing CatBoost models running in WebAssembly (WASM) within web workers. The framework enables researchers and developers to quickly test different C++ optimization strategies and measure their impact on performance and accuracy.

### Key Goals

- **Rapid experimentation**: Modify C++ code, run one command, get comprehensive results
- **Performance optimization**: Find the optimal balance between speed and accuracy
- **WebWorker deployment**: Test real-world WASM performance in browser environments
- **Reproducible benchmarks**: Fixed test data ensures fair comparisons across experiments

## Technical Architecture

### Pipeline Overview

```
User modifies CPP → Run experiment.sh → Auto-compile to WASM → Run in WebWorker → Compare results → Generate report
```

### Core Components

1. **Model Training Pipeline**

   - Dataset: Diamonds dataset from seaborn/ggplot2 (53,940 samples, 10 features)
     - Available via: `seaborn.load_dataset('diamonds')` or Kaggle
     - Features: carat, cut, color, clarity, depth, table, price, x, y, z
     - Categorical: cut (Fair/Good/Very Good/Premium/Ideal), color (D-J), clarity (I1/SI2/SI1/VS2/VS1/VVS2/VVS1/IF)
   - Model: CatBoost regression predicting diamond prices
   - Outputs: `.cbm` file (ground truth) and `.cpp` file (baseline)

2. **Test Data Generation**

   - 1 million random test vectors
   - Valid ranges from training data
   - Binary format for fast loading
   - Fixed seed for reproducibility

3. **Experimentation Framework**

   - Automated WASM compilation using Emscripten
   - Web worker harness for isolated execution
   - Accuracy comparison against CBM ground truth
   - Performance metrics collection

4. **Reporting System**
   - JSON output with performance metrics
   - Error distribution analysis
   - Visualization of results
   - Regression detection

## Environment Configuration

### Dependencies and Versions

#### Python Package Management

We use **uv** as our Python package manager for its speed and reliability. To install dependencies:

```bash
# Install a package
uv add catboost

# Install all dependencies
uv sync
```

#### Python Dependencies

```
catboost==1.2.8          # Latest stable release (Apr 2025)
numpy==2.3.1             # For numerical operations (Jun 2025)
pandas==2.3.1            # Data manipulation (Jul 2025)
seaborn==0.13.2          # Diamonds dataset source
matplotlib==3.10.3       # Visualization (May 2025)
scikit-learn==1.7.1      # Train/test split utilities (Jul 2025)
```

#### System Dependencies

- **Python**: 3.10+ (tested with 3.12)
- **Node.js**: 20.x LTS (for running WASM tests, minimum 18.3 for Emscripten 4.0)
- **Emscripten**: 4.0.9+ (latest stable series)

### Emscripten Setup

The framework uses Emscripten installed at:

```
/Users/yuehu/opensources/emsdk
```

The build scripts will automatically configure the environment using:

```bash
source /Users/yuehu/opensources/emsdk/emsdk_env.sh
```

## Usage Guide

### Basic Workflow

1. **Initial Setup** (one-time)

   ```bash
   # Install dependencies using uv
   uv sync
   
   # Train model and generate test data
   python setup/train_model.py
   python setup/generate_test_data.py
   ```

2. **Run Experiment**
   ```bash
   # Test your optimized model (automatically compares to baseline)
   ./experiment.sh experiments/my_optimized_model.cpp
   ```

### Advanced Features

- **Batch Size Testing**

  ```bash
  ./experiment.sh my_model.cpp --batch-sizes=1,10,100,1000
  ```

- **Custom Compilation Flags**
  ```bash
  ./experiment.sh my_model.cpp --em-flags="-O3 -msimd128"
  ```

## Implementation Phases

### Phase 1: Foundation (Week 1)

- [ ] Set up project structure
- [ ] Train CatBoost model on Diamonds dataset
- [ ] Export model to CBM and CPP formats
- [ ] Generate 1M test vectors
- [ ] Verify baseline CPP matches CBM predictions

### Phase 2: Build Pipeline (Week 2)

- [ ] Create Emscripten compilation wrapper
- [ ] Implement web worker harness
- [ ] Build accuracy comparison tools
- [ ] Create basic experiment.sh script
- [ ] Test end-to-end pipeline with baseline model

### Phase 3: Basic Features (Week 3)

- [ ] Add error handling for common failures
- [ ] Create simple visualization of results
- [ ] Write basic documentation

### Phase 4: Polish (Week 4)

- [ ] Test framework end-to-end
- [ ] Fix bugs

## Project Structure

```
catboost-optimize/
├── plan.md                     # This file
├── README.md                   # User-facing documentation
├── setup/
│   ├── train_model.py          # Train CatBoost model
│   ├── generate_test_data.py   # Generate 1M test vectors
│   └── requirements.txt        # Python dependencies
├── models/
│   ├── baseline.cpp            # Original CatBoost C++ export
│   ├── baseline.cbm            # CatBoost model file (ground truth)
│   ├── model_metadata.json     # Feature info, ranges, etc.
│   └── test_data.bin           # 1M test vectors (binary format)
├── framework/
│   ├── experiment.sh           # Main experiment runner
│   ├── compile_wasm.js         # Emscripten compilation wrapper
│   ├── worker_template.js      # Web worker implementation
│   ├── run_predictions.js      # Node.js runner for WASM
│   ├── accuracy_checker.py     # Compare outputs vs ground truth
│   ├── report_generator.py     # Generate JSON/HTML reports
│   └── visualize_results.py    # Create performance plots
├── experiments/                # User's optimized CPP files
│   ├── .gitignore             # Ignore user experiments
│   └── example_simd.cpp       # Example optimization
├── results/                    # Experiment outputs
│   ├── baseline_results.json   # Baseline performance
│   └── experiment_*.json       # Timestamped results
└── web/                        # Browser-based testing
    ├── index.html              # Test interface
    ├── worker.js               # Production web worker
    └── benchmark.html          # Performance testing UI
```

## Example Output

### Experiment Report (JSON)

```json
{
  "experiment_id": "2024-01-20_15-30-45",
  "model": {
    "name": "my_optimized_v3.cpp",
    "wasm_size_kb": 245,
    "compilation_flags": "-O3 -msimd128"
  },
  "performance": {
    "total_predictions": 1000000,
    "total_time_ms": 850,
    "predictions_per_second": 1176470,
    "mean_prediction_time_us": 0.85,
    "p95_prediction_time_us": 1.2,
    "p99_prediction_time_us": 2.1,
    "speedup_vs_baseline": 3.2
  },
  "accuracy": {
    "comparison_against": "baseline.cbm",
    "exact_matches_ratio": 0.95,
    "max_absolute_error": 0.0001,
    "mean_absolute_error": 0.00002,
    "rmse": 0.00003,
    "error_percentiles": {
      "p50": 0.00001,
      "p95": 0.00005,
      "p99": 0.00008
    },
    "regression_detected": false
  },
  "memory": {
    "wasm_module_size_kb": 245,
    "heap_size_mb": 16,
    "peak_memory_mb": 42
  },
  "environment": {
    "node_version": "18.17.0",
    "emscripten_version": "3.1.47",
    "platform": "darwin x64"
  }
}
```

### Console Output

```
$ ./experiment.sh experiments/my_optimized_v3.cpp

🚀 CatBoost WASM Experiment Runner
================================
📁 Model: experiments/my_optimized_v3.cpp
📊 Test data: 1,000,000 samples

[1/5] Compiling C++ to WASM...
      ✓ Compilation successful (245 KB)

[2/5] Loading test data...
      ✓ Loaded 1M samples (76.3 MB)

[3/5] Running predictions...
      ⠦ Progress: 650,000/1,000,000 (65%)
      ✓ Completed in 850ms (1,176,470 pred/s)

[4/5] Checking accuracy...
      ✓ Max error: 0.0001 (within threshold)
      ✓ 95% of predictions exact match

[5/5] Generating report...
      ✓ Report saved: results/experiment_2024-01-20_15-30-45.json

📈 Summary:
   • Speed: 3.2x faster than baseline
   • Accuracy: Within acceptable bounds
   • Memory: 42 MB peak usage

✅ Experiment completed successfully!
```

## Success Metrics

### Minimum Viable Framework

- **It works**: Can compile CPP to WASM and run predictions
- **Accurate comparison**: Results match between different runs
- **Simple workflow**: One command to test an optimization
- **Clear results**: JSON report with speed and accuracy metrics
