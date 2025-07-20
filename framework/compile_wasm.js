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

// Build emcc command
const emccCommand = `emcc ${options.input} \
    -o ${options.output} \
    ${options.flags} \
    -std=c++11 \
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
    process.exit(1);
}