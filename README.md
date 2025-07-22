# CatBoost WebAssembly Optimization Framework

A streamlined framework for optimizing CatBoost models running in WebAssembly. Test different C++ optimization strategies and measure their impact on performance.

## Quick Start

### Prerequisites
- Node.js (v14+)
- Python with uv package manager
- Emscripten SDK installed at `/Users/yuehu/opensources/emsdk`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/catboost-optimize.git
cd catboost-optimize

# Install Node.js dependencies
npm install

# Install Python dependencies (using uv)
uv pip install -r pyproject.toml
```

### Basic Usage

```bash
# Run experiment with default settings
./run_experiment.sh

# Test with custom batch sizes
./run_experiment.sh --name my_test --batch-sizes "100 1000 10000"

# Enable SIMD optimizations
./run_experiment.sh --name simd_test --simd

# Use aggressive compiler optimizations
./run_experiment.sh --name fast --emflags "-O3 -flto"
```

### Results

Experiments output:
- Performance metrics (predictions/second)
- Accuracy comparison
- Optimal batch size recommendations

Results are saved to `experiment_results/<experiment_name>/results.json`

## Project Structure

```
catboost-optimize/
├── models/                 # CatBoost model files
│   ├── baseline.cpp       # Exported C++ model
│   └── baseline.cbm       # Original CatBoost model
├── experiments/           # C++ wrapper implementations
│   └── batch_wrapper.cpp  # Optimized batch processing wrapper
├── test_data/            # Test datasets
│   └── test_data_1M.bin  # 1M sample test data
├── setup/                # Model training and data generation
├── run_experiment.sh     # Main experiment runner
└── experiment_runner.js  # Core experiment engine
```

## Key Features

- **Flexible Wrapper System**: Test different C++ implementations
- **Batch Processing**: Optimize prediction throughput
- **Compiler Optimizations**: Support for SIMD, threading, and custom flags
- **Automated Testing**: Compare performance across configurations
- **Detailed Metrics**: Track speed, accuracy, and memory usage

## Documentation

For detailed usage and development information, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

## License

MIT License - see LICENSE file for details