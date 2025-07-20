#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
    input: '',
    output: '',
    flags: '-O3'
};

for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];
    if (key in options) {
        options[key] = value;
    }
}

if (!options.input || !options.output) {
    console.error('Usage: node compile_wasm.js --input <cpp_file> --output <js_output> [--flags "-O3"]');
    process.exit(1);
}

console.log('🔨 Compiling C++ to WASM...');
console.log(`   Input: ${options.input}`);
console.log(`   Output: ${options.output}`);
console.log(`   Flags: ${options.flags}`);

// Prepare output directory
const outputDir = path.dirname(options.output);
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Create temporary wrapper file
const tempDir = '/tmp';
const wrapperPath = path.join(tempDir, `catboost_wrapper_${Date.now()}.cpp`);
const wrapperContent = `#include <vector>
#include <string>
#include <cmath>

// Include the generated model file
#include "${path.resolve(options.input)}"

// Mapping from encoded values to string values
const char* cutValues[] = {"Ideal", "Premium", "Good", "Very Good", "Fair"};
const char* colorValues[] = {"E", "I", "J", "H", "F", "G", "D"};
const char* clarityValues[] = {"SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"};

// External C interface for WASM
extern "C" {
    double catboostPredict(float* allFeatures, size_t totalFeatureCount) {
        // The model expects 6 float features and 3 categorical features
        const size_t floatFeatureCount = 6;
        
        // Extract float features (first 6)
        std::vector<float> floatFeatures(allFeatures, allFeatures + floatFeatureCount);
        
        // Extract and decode categorical features (last 3)
        std::vector<std::string> catFeatures;
        
        // Feature 6: cut
        int cutIdx = (int)std::round(allFeatures[6]);
        if (cutIdx >= 0 && cutIdx < 5) {
            catFeatures.push_back(cutValues[cutIdx]);
        } else {
            catFeatures.push_back("Good"); // default
        }
        
        // Feature 7: color
        int colorIdx = (int)std::round(allFeatures[7]);
        if (colorIdx >= 0 && colorIdx < 7) {
            catFeatures.push_back(colorValues[colorIdx]);
        } else {
            catFeatures.push_back("G"); // default
        }
        
        // Feature 8: clarity
        int clarityIdx = (int)std::round(allFeatures[8]);
        if (clarityIdx >= 0 && clarityIdx < 8) {
            catFeatures.push_back(clarityValues[clarityIdx]);
        } else {
            catFeatures.push_back("SI1"); // default
        }
        
        return ApplyCatboostModel(floatFeatures, catFeatures);
    }
}`;

fs.writeFileSync(wrapperPath, wrapperContent);

// Build emcc command
const emccCommand = `emcc ${wrapperPath} \
    -o ${options.output} \
    ${options.flags} \
    -std=c++20 \
    -s EXPORTED_FUNCTIONS='["_catboostPredict", "_malloc", "_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap", "setValue", "getValue", "HEAPF32"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="CatBoostModule" \
    -s ENVIRONMENT='web,worker,node' \
    -s WASM=1 \
    -s NO_EXIT_RUNTIME=1`;

try {
    // Execute compilation
    console.log('   Running emcc...');
    execSync(emccCommand, { stdio: 'inherit' });
    
    // Verify output files exist
    const jsFile = options.output;
    const wasmFile = options.output.replace('.js', '.wasm');
    
    if (!fs.existsSync(jsFile)) {
        throw new Error(`JS output file not created: ${jsFile}`);
    }
    
    if (!fs.existsSync(wasmFile)) {
        throw new Error(`WASM output file not created: ${wasmFile}`);
    }
    
    // Get file sizes
    const jsSize = fs.statSync(jsFile).size;
    const wasmSize = fs.statSync(wasmFile).size;
    
    console.log(`   ✓ JS file: ${(jsSize / 1024).toFixed(1)} KB`);
    console.log(`   ✓ WASM file: ${(wasmSize / 1024).toFixed(1)} KB`);
    console.log('   ✓ Compilation successful!');
    
} catch (error) {
    console.error('❌ Compilation failed:');
    console.error(error.message);
    // Clean up temporary wrapper file
    if (fs.existsSync(wrapperPath)) {
        fs.unlinkSync(wrapperPath);
    }
    process.exit(1);
} finally {
    // Clean up temporary wrapper file
    if (fs.existsSync(wrapperPath)) {
        fs.unlinkSync(wrapperPath);
    }
}