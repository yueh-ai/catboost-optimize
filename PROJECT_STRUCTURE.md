# Project Structure

## Core Files

### Execution
- `run_experiment.sh` - Main CLI interface for running experiments
- `experiment_runner.js` - Core experiment engine (Node.js)

### C++ Components
- `models/baseline.cpp` - Auto-generated CatBoost model (1.5MB)
- `experiments/batch_wrapper.cpp` - Optimized batch processing wrapper

### Setup & Data
- `setup/train_model.py` - Train CatBoost model
- `setup/generate_test_data.py` - Generate test datasets
- `test_data/test_data_1M.bin` - 1M sample test data (44MB)

### Documentation
- `README.md` - Quick start guide
- `DEVELOPER_GUIDE.md` - Detailed development documentation

### Configuration
- `package.json` - Node.js dependencies
- `pyproject.toml` - Python dependencies
- `.gitignore` - Version control exclusions

## Directory Structure

```
catboost-optimize/
├── models/              # Model files
│   ├── baseline.cbm    # Original CatBoost model
│   ├── baseline.cpp    # Exported C++ model
│   └── encoding_map.json
├── experiments/         # C++ wrappers
│   └── batch_wrapper.cpp
├── test_data/          # Test datasets
│   └── test_data_1M.bin
├── setup/              # Training scripts
│   ├── train_model.py
│   └── generate_test_data.py
├── run_experiment.sh   # Main CLI
├── experiment_runner.js # Core engine
└── docs/               # Documentation
    ├── README.md
    └── DEVELOPER_GUIDE.md
```

## Workflow

1. **Model Training**: `setup/train_model.py` → `models/baseline.cbm`
2. **C++ Export**: CatBoost → `models/baseline.cpp`
3. **Test Data**: `setup/generate_test_data.py` → `test_data/test_data_1M.bin`
4. **Experiments**: `run_experiment.sh` → compile & test → results

## Results

Experiments create:
- `experiment_results/<name>/model.js` - Compiled WASM module
- `experiment_results/<name>/model.wasm` - WebAssembly binary
- `experiment_results/<name>/results.json` - Performance metrics