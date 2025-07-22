#!/bin/bash

# Simplified CatBoost WASM Optimization Experiment Runner
# This script compiles and runs the optimization experiment without batching

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
WRAPPER="./experiments/wrapper.cpp"
EMFLAGS="-O3"
TEST_DATA="./models/test_data.bin"
OUTPUT_DIR="./experiment_results"

# Function to display usage
usage() {
    echo "CatBoost WASM Optimization Experiment Runner (Simplified)"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -n, --name NAME          Experiment name (default: auto-generated)"
    echo "  -w, --wrapper FILE       Path to wrapper C++ file (default: $WRAPPER)"
    echo "  -e, --emflags FLAGS      Emscripten compiler flags (default: $EMFLAGS)"
    echo "  -d, --data FILE          Test data file (default: $TEST_DATA)"
    echo "  -o, --output DIR         Output directory (default: $OUTPUT_DIR)"
    echo "  --simd                   Enable SIMD optimizations"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings"
    echo "  $0"
    echo ""
    echo "  # Run with custom wrapper and SIMD"
    echo "  $0 --wrapper ./my_optimized_wrapper.cpp --simd"
    echo ""
    echo "  # Run with aggressive optimization"
    echo "  $0 --emflags \"-O3 -flto --closure 1\" --name aggressive_opt"
}

# Parse command line arguments
EXPERIMENT_NAME=""
SIMD_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -w|--wrapper)
            WRAPPER="$2"
            shift 2
            ;;
        -e|--emflags)
            EMFLAGS="$2"
            shift 2
            ;;
        -d|--data)
            TEST_DATA="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --simd)
            SIMD_FLAG="-msimd128"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Generate experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="exp_$(date +%s)"
fi

# Check if required files exist
if [ ! -f "$WRAPPER" ]; then
    echo -e "${RED}Error: Wrapper file not found: $WRAPPER${NC}"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo -e "${RED}Error: Test data file not found: $TEST_DATA${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

# Source Emscripten environment if available
EMSDK_PATH="/Users/yuehu/opensources/emsdk"
if [ -f "$EMSDK_PATH/emsdk_env.sh" ]; then
    echo -e "${GREEN}Sourcing Emscripten environment...${NC}"
    source "$EMSDK_PATH/emsdk_env.sh" > /dev/null 2>&1
fi

# Check if Emscripten is installed
if ! command -v emcc &> /dev/null; then
    echo -e "${RED}Error: Emscripten is not installed or not in PATH${NC}"
    echo "Expected location: $EMSDK_PATH"
    echo "Please install Emscripten: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

# Create experiment directory
EXPERIMENT_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"

# Display experiment configuration
echo -e "${GREEN}Starting CatBoost WASM Optimization Experiment (Simplified)${NC}"
echo -e "Experiment name: ${YELLOW}$EXPERIMENT_NAME${NC}"
echo -e "Wrapper: ${YELLOW}$WRAPPER${NC}"
echo -e "Emscripten flags: ${YELLOW}$EMFLAGS $SIMD_FLAG${NC}"
echo -e "Test data: ${YELLOW}$TEST_DATA${NC}"
echo -e "Output directory: ${YELLOW}$EXPERIMENT_DIR${NC}"
echo ""

# Compile the WASM module
echo -e "${GREEN}Compiling WASM module...${NC}"
emcc $WRAPPER \
    -o "$EXPERIMENT_DIR/model.js" \
    $EMFLAGS \
    $SIMD_FLAG \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_catboostPredictAll", "_catboostPredict", "_malloc", "_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap", "HEAP8", "HEAP16", "HEAP32", "HEAPU8", "HEAPU16", "HEAPU32", "HEAPF32", "HEAPF64"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="Module" \
    -s ENVIRONMENT='node'

if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful!${NC}"
echo ""

# Run the experiment
echo -e "${GREEN}Running experiment...${NC}"
node experiment_runner.js "$EXPERIMENT_NAME"

# Check if experiment was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Experiment completed successfully!${NC}"
    echo -e "Results saved to: ${YELLOW}$EXPERIMENT_DIR/results.json${NC}"
    
    # Display results
    if [ -f "$EXPERIMENT_DIR/results.json" ]; then
        echo ""
        echo -e "${GREEN}Results:${NC}"
        cat "$EXPERIMENT_DIR/results.json"
    fi
else
    echo ""
    echo -e "${RED}Experiment failed!${NC}"
fi