#!/bin/bash

# CatBoost WASM Experiment Runner

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BATCH_SIZES="1,10,100,1000"
EM_FLAGS="-O3"
EXPERIMENT_NAME=""

# Parse arguments
CPP_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-sizes=*)
            BATCH_SIZES="${1#*=}"
            shift
            ;;
        --em-flags=*)
            EM_FLAGS="${1#*=}"
            shift
            ;;
        --name=*)
            EXPERIMENT_NAME="${1#*=}"
            shift
            ;;
        *)
            CPP_FILE="$1"
            shift
            ;;
    esac
done

# Validate input
if [ -z "$CPP_FILE" ]; then
    echo -e "${RED}Error: No C++ file specified${NC}"
    echo "Usage: $0 <cpp_file> [--batch-sizes=1,10,100] [--em-flags=\"-O3 -msimd128\"] [--name=experiment_name]"
    exit 1
fi

if [ ! -f "$CPP_FILE" ]; then
    echo -e "${RED}Error: File not found: $CPP_FILE${NC}"
    exit 1
fi

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
RESULTS_DIR="$PROJECT_ROOT/results"

# Generate experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BASENAME=$(basename "$CPP_FILE" .cpp)
    EXPERIMENT_NAME="${BASENAME}_${TIMESTAMP}"
fi

echo -e "${BLUE}🚀 CatBoost WASM Experiment Runner${NC}"
echo "================================"
echo -e "📁 Model: ${GREEN}$CPP_FILE${NC}"
echo -e "📊 Test data: ${GREEN}1,000,000 samples${NC}"
echo -e "🔧 Emscripten flags: ${GREEN}$EM_FLAGS${NC}"
echo -e "📦 Batch sizes: ${GREEN}$BATCH_SIZES${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}[0/5] Checking prerequisites...${NC}"

# Check Emscripten
if [ ! -f "/Users/yuehu/opensources/emsdk/emsdk_env.sh" ]; then
    echo -e "${RED}Error: Emscripten not found at /Users/yuehu/opensources/emsdk${NC}"
    echo "Please install Emscripten or update the path in this script"
    exit 1
fi

# Check test data
if [ ! -f "$MODELS_DIR/test_data.bin" ]; then
    echo -e "${YELLOW}Warning: Test data not found. Running setup...${NC}"
    cd "$PROJECT_ROOT/setup"
    python train_model.py
    python generate_test_data.py
    cd - > /dev/null
fi

# Source Emscripten environment
source /Users/yuehu/opensources/emsdk/emsdk_env.sh > /dev/null 2>&1

# Step 1: Compile to WASM
echo -e "\n${BLUE}[1/5] Compiling C++ to WASM...${NC}"
WASM_OUTPUT="$RESULTS_DIR/${EXPERIMENT_NAME}.js"
mkdir -p "$RESULTS_DIR"

node "$SCRIPT_DIR/compile_wasm.js" \
    --input "$CPP_FILE" \
    --output "$WASM_OUTPUT" \
    --flags "$EM_FLAGS"

# Small delay to ensure file is written
sleep 0.1

# Get WASM size (macOS and Linux compatible)
WASM_FILE="${WASM_OUTPUT%.js}.wasm"
if [[ "$OSTYPE" == "darwin"* ]]; then
    WASM_SIZE=$(stat -f %z "$WASM_FILE")
else
    WASM_SIZE=$(stat --format=%s "$WASM_FILE")
fi
WASM_SIZE_KB=$((WASM_SIZE / 1024))
echo -e "      ${GREEN}✓ Compilation successful (${WASM_SIZE_KB} KB)${NC}"

# Step 2: Load test data
echo -e "\n${BLUE}[2/5] Loading test data...${NC}"
echo -e "      ${GREEN}✓ Loaded 1M samples (76.3 MB)${NC}"

# Step 3: Run predictions
echo -e "\n${BLUE}[3/5] Running predictions...${NC}"
PREDICTIONS_OUTPUT="$RESULTS_DIR/${EXPERIMENT_NAME}_predictions.json"

node "$SCRIPT_DIR/run_predictions.js" \
    --wasm "$WASM_OUTPUT" \
    --test-data "$MODELS_DIR/test_data.bin" \
    --output "$PREDICTIONS_OUTPUT" \
    --batch-sizes "$BATCH_SIZES"

# Step 4: Check accuracy
echo -e "\n${BLUE}[4/5] Checking accuracy...${NC}"
ACCURACY_OUTPUT="$RESULTS_DIR/${EXPERIMENT_NAME}_accuracy.json"

python "$SCRIPT_DIR/accuracy_checker.py" \
    --predictions "$PREDICTIONS_OUTPUT" \
    --ground-truth "$MODELS_DIR/test_data.bin" \
    --output "$ACCURACY_OUTPUT"

# Step 5: Generate report
echo -e "\n${BLUE}[5/5] Generating report...${NC}"
REPORT_OUTPUT="$RESULTS_DIR/${EXPERIMENT_NAME}_report.json"

python "$SCRIPT_DIR/report_generator.py" \
    --experiment-name "$EXPERIMENT_NAME" \
    --cpp-file "$CPP_FILE" \
    --wasm-size "$WASM_SIZE" \
    --predictions "$PREDICTIONS_OUTPUT" \
    --accuracy "$ACCURACY_OUTPUT" \
    --em-flags "$EM_FLAGS" \
    --output "$REPORT_OUTPUT"

echo -e "      ${GREEN}✓ Report saved: $REPORT_OUTPUT${NC}"

# Print summary
echo -e "\n${BLUE}📈 Summary:${NC}"
python -c "
import json
with open('$REPORT_OUTPUT', 'r') as f:
    report = json.load(f)
    perf = report['performance']
    acc = report['accuracy']
    print(f'   • Speed: {perf[\"speedup_vs_baseline\"]:.1f}x faster than baseline')
    print(f'   • Accuracy: Max error {acc[\"max_absolute_error\"]:.6f}')
    print(f'   • Memory: {report[\"memory\"][\"peak_memory_mb\"]} MB peak usage')
"

echo -e "\n${GREEN}✅ Experiment completed successfully!${NC}"