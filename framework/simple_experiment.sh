#!/bin/bash

# Simplified CatBoost WASM Experiment Runner
# Tests 1 million inputs in a single batch

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
EM_FLAGS="-O3"
EXPERIMENT_NAME=""

# Parse arguments
CPP_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
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
    echo "Usage: $0 <cpp_file> [--em-flags=\"-O3 -msimd128\"] [--name=experiment_name]"
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

echo -e "${BLUE}🚀 Simplified CatBoost WASM Tester${NC}"
echo "================================"
echo -e "📁 Model: ${GREEN}$CPP_FILE${NC}"
echo -e "📊 Test data: ${GREEN}1,000,000 samples${NC}"
echo -e "🔧 Emscripten flags: ${GREEN}$EM_FLAGS${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}[1/4] Checking prerequisites...${NC}"

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
echo -e "\n${BLUE}[2/4] Compiling C++ to WASM...${NC}"
WASM_OUTPUT="$RESULTS_DIR/${EXPERIMENT_NAME}.js"
mkdir -p "$RESULTS_DIR"

node "$SCRIPT_DIR/simple_compile_wasm.js" \
    --input "$CPP_FILE" \
    --output "$WASM_OUTPUT" \
    --flags "$EM_FLAGS"

# Get WASM size
WASM_FILE="${WASM_OUTPUT%.js}.wasm"
if [[ "$OSTYPE" == "darwin"* ]]; then
    WASM_SIZE=$(stat -f %z "$WASM_FILE")
else
    WASM_SIZE=$(stat --format=%s "$WASM_FILE")
fi
WASM_SIZE_KB=$((WASM_SIZE / 1024))
echo -e "      ${GREEN}✓ Compilation successful (${WASM_SIZE_KB} KB)${NC}"

# Step 2: Run predictions
echo -e "\n${BLUE}[3/4] Running 1M predictions...${NC}"
RESULTS_OUTPUT="$RESULTS_DIR/${EXPERIMENT_NAME}_results.json"

node "$SCRIPT_DIR/simple_run.js" \
    --wasm "$WASM_OUTPUT" \
    --test-data "$MODELS_DIR/test_data.bin" \
    --output "$RESULTS_OUTPUT"

# Step 3: Generate report
echo -e "\n${BLUE}[4/4] Generating report...${NC}"
REPORT_OUTPUT="$RESULTS_DIR/${EXPERIMENT_NAME}_report.json"

python "$SCRIPT_DIR/simple_report.py" \
    --experiment-name "$EXPERIMENT_NAME" \
    --cpp-file "$CPP_FILE" \
    --wasm-size "$WASM_SIZE" \
    --results "$RESULTS_OUTPUT" \
    --em-flags="$EM_FLAGS" \
    --output "$REPORT_OUTPUT"

echo -e "      ${GREEN}✓ Report saved: $REPORT_OUTPUT${NC}"

# Print summary
echo -e "\n${BLUE}📈 Results:${NC}"
python -c "
import json
with open('$REPORT_OUTPUT', 'r') as f:
    report = json.load(f)
    print(f'   • Total time: {report[\"total_time_ms\"]:.0f} ms')
    print(f'   • Speed: {report[\"predictions_per_second\"]:,.0f} predictions/second')
    print(f'   • Accuracy: Max error {report[\"accuracy\"][\"max_error\"]:.6f}')
"

echo -e "\n${GREEN}✅ Test completed successfully!${NC}"