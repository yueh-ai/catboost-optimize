#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
    wasm: '',
    'test-data': '',
    output: ''
};

for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];
    if (key in options) {
        options[key] = value;
    }
}

// Validate inputs
if (!options.wasm || !options['test-data'] || !options.output) {
    console.error('Usage: node simple_run.js --wasm <module.js> --test-data <data.bin> --output <results.json>');
    process.exit(1);
}

// Load test data
function loadTestData(filePath) {
    const buffer = fs.readFileSync(filePath);
    let offset = 0;
    
    // Read header
    const magic = buffer.readUInt32LE(offset); offset += 4;
    const version = buffer.readUInt32LE(offset); offset += 4;
    const nSamples = buffer.readUInt32LE(offset); offset += 4;
    const nFloatFeatures = buffer.readUInt32LE(offset); offset += 4;
    
    if (magic !== 0xCAFEBABE) {
        throw new Error('Invalid test data file format');
    }
    
    if (version === 2) {
        // Version 2 format: has additional header field
        const nCatFeatures = buffer.readUInt32LE(offset); offset += 4;
        const totalFeatures = nFloatFeatures + nCatFeatures;
        
        console.log(`      Loading ${nSamples.toLocaleString()} samples, ${totalFeatures} features (${nFloatFeatures} numeric, ${nCatFeatures} categorical)`);
        
        // Read features - convert to format expected by C++ (all floats)
        const features = new Float32Array(nSamples * totalFeatures);
        const bytesPerSample = nFloatFeatures * 4 + nCatFeatures + 1; // 6 floats + 3 uint8 + 1 padding
        
        for (let i = 0; i < nSamples; i++) {
            const sampleOffset = offset + i * bytesPerSample;
            const featureOffset = i * totalFeatures;
            
            // Read float features
            for (let j = 0; j < nFloatFeatures; j++) {
                features[featureOffset + j] = buffer.readFloatLE(sampleOffset + j * 4);
            }
            
            // Read categorical features (uint8) and convert to float
            for (let j = 0; j < nCatFeatures; j++) {
                features[featureOffset + nFloatFeatures + j] = buffer.readUInt8(sampleOffset + nFloatFeatures * 4 + j);
            }
        }
        
        offset += nSamples * bytesPerSample;
        
        // Read ground truth predictions
        const predictionsBuffer = buffer.slice(offset, offset + nSamples * 4);
        const groundTruth = new Float32Array(predictionsBuffer.buffer, predictionsBuffer.byteOffset, nSamples);
        
        return { features, groundTruth, nSamples, nFeatures: totalFeatures };
    } else {
        // Original format for backward compatibility
        const nFeatures = nFloatFeatures; // In v1, this was the total features
        console.log(`      Loading ${nSamples.toLocaleString()} samples, ${nFeatures} features each`);
        
        // Read features
        const featuresSize = nSamples * nFeatures * 4; // float32
        const featuresBuffer = buffer.slice(offset, offset + featuresSize);
        const features = new Float32Array(featuresBuffer.buffer, featuresBuffer.byteOffset, nSamples * nFeatures);
        offset += featuresSize;
        
        // Read ground truth predictions
        const predictionsBuffer = buffer.slice(offset, offset + nSamples * 4);
        const groundTruth = new Float32Array(predictionsBuffer.buffer, predictionsBuffer.byteOffset, nSamples);
        
        return { features, groundTruth, nSamples, nFeatures };
    }
}

// Run predictions in worker thread
async function runPredictionsInWorker(wasmPath, testData) {
    return new Promise((resolve, reject) => {
        const workerPath = path.join(__dirname, 'worker.js');
        const worker = new Worker(workerPath, {
            workerData: {
                wasmPath,
                testData
            }
        });
        
        let results = null;
        
        worker.on('message', (msg) => {
            if (msg.type === 'status') {
                console.log(`      ${msg.message}`);
            } else if (msg.type === 'complete') {
                results = msg.results;
            }
        });
        
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0) {
                reject(new Error(`Worker stopped with exit code ${code}`));
            } else {
                resolve(results);
            }
        });
    });
}

// Main execution
async function main() {
    try {
        // Load test data
        const testData = loadTestData(options['test-data']);
        
        // Run predictions
        console.log(`      Processing ${testData.nSamples.toLocaleString()} predictions...`);
        const startTime = Date.now();
        
        const results = await runPredictionsInWorker(options.wasm, testData);
        
        const endTime = Date.now();
        const totalTime = endTime - startTime;
        
        // Save results
        const output = {
            experiment: path.basename(options.wasm, '.js'),
            timestamp: new Date().toISOString(),
            totalSamples: testData.nSamples,
            totalTimeMs: totalTime,
            predictionsPerSecond: Math.floor(testData.nSamples / (totalTime / 1000)),
            predictions: results.predictions.slice(0, 1000), // Save first 1000 for verification
            groundTruth: Array.from(testData.groundTruth.slice(0, 1000)),
            workerTimeMs: results.executionTime
        };
        
        fs.writeFileSync(options.output, JSON.stringify(output, null, 2));
        console.log(`      ✓ Completed in ${totalTime}ms (${output.predictionsPerSecond.toLocaleString()} pred/s)`);
        
    } catch (error) {
        console.error('\n❌ Error:', error.message);
        process.exit(1);
    }
}

main();