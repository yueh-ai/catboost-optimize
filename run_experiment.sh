#!/bin/bash

# CatBoost WASM Optimization Experiment Runner
# This script provides an easy interface to run optimization experiments

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
WRAPPER="./experiments/batch_wrapper.cpp"
EMFLAGS="-O3"
BATCH_SIZES="1 10 100 1000 5000 10000"
TEST_DATA="./test_data/test_data_1M.bin"
OUTPUT_DIR="./experiment_results"

# Function to display usage
usage() {
    echo "CatBoost WASM Optimization Experiment Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -n, --name NAME          Experiment name (default: auto-generated)"
    echo "  -w, --wrapper FILE       Path to wrapper C++ file (default: $WRAPPER)"
    echo "  -e, --emflags FLAGS      Emscripten compiler flags (default: $EMFLAGS)"
    echo "  -b, --batch-sizes SIZES  Space-separated batch sizes (default: $BATCH_SIZES)"
    echo "  -d, --data FILE          Test data file (default: $TEST_DATA)"
    echo "  -o, --output DIR         Output directory (default: $OUTPUT_DIR)"
    echo "  --simd                   Enable SIMD optimizations"
    echo "  --threads                Enable threading support"
    echo "  --no-batch-api          Disable batch API usage"
    echo ""
    echo "Examples:"
    echo "  # Run with default settings"
    echo "  $0"
    echo ""
    echo "  # Run with custom name and SIMD enabled"
    echo "  $0 --name simd_test --simd"
    echo ""
    echo "  # Run with aggressive optimization flags"
    echo "  $0 --emflags \"-O3 -flto\" --name aggressive_opt"
    echo ""
    echo "  # Test custom wrapper with specific batch sizes"
    echo "  $0 --wrapper ./my_wrapper.cpp --batch-sizes \"100 1000 10000\""
}

# Parse command line arguments
EXPERIMENT_NAME=""
SIMD_FLAG=""
THREADS_FLAG=""
USE_BATCH_API="--use-batch-api"

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
        -b|--batch-sizes)
            BATCH_SIZES="$2"
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
            SIMD_FLAG="--simd"
            shift
            ;;
        --threads)
            THREADS_FLAG="--threads"
            shift
            ;;
        --no-batch-api)
            USE_BATCH_API=""
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
    EXPERIMENT_NAME="exp_$(date +%Y%m%d_%H%M%S)"
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

# Display experiment configuration
echo -e "${GREEN}Starting CatBoost WASM Optimization Experiment${NC}"
echo -e "Experiment name: ${YELLOW}$EXPERIMENT_NAME${NC}"
echo -e "Wrapper: ${YELLOW}$WRAPPER${NC}"
echo -e "Emscripten flags: ${YELLOW}$EMFLAGS${NC}"
echo -e "Batch sizes: ${YELLOW}$BATCH_SIZES${NC}"
echo -e "Test data: ${YELLOW}$TEST_DATA${NC}"
echo -e "Output directory: ${YELLOW}$OUTPUT_DIR${NC}"

if [ -n "$SIMD_FLAG" ]; then
    echo -e "SIMD: ${YELLOW}Enabled${NC}"
fi

if [ -n "$THREADS_FLAG" ]; then
    echo -e "Threading: ${YELLOW}Enabled${NC}"
fi

echo ""

# Convert batch sizes to array format for Node.js
BATCH_ARRAY=""
for size in $BATCH_SIZES; do
    if [ -z "$BATCH_ARRAY" ]; then
        BATCH_ARRAY="$size"
    else
        BATCH_ARRAY="$BATCH_ARRAY $size"
    fi
done

# Run the experiment
node experiment_runner.js \
    --name "$EXPERIMENT_NAME" \
    --wrapper "$WRAPPER" \
    --emflags "$EMFLAGS" \
    --batch-sizes $BATCH_ARRAY \
    --test-data "$TEST_DATA" \
    --output-dir "$OUTPUT_DIR" \
    $SIMD_FLAG \
    $THREADS_FLAG \
    $USE_BATCH_API

# Check if experiment was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Experiment completed successfully!${NC}"
    echo -e "Results saved to: ${YELLOW}$OUTPUT_DIR/$EXPERIMENT_NAME/results.json${NC}"
else
    echo ""
    echo -e "${RED}Experiment failed!${NC}"
    echo -e "Check error log at: ${YELLOW}$OUTPUT_DIR/$EXPERIMENT_NAME/error.json${NC}"
fi