#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
    wasm: '',
    'test-data': '',
    output: '',
    'batch-sizes': '1,10,100,1000'
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
    console.error('Usage: node run_predictions.js --wasm <module.js> --test-data <data.bin> --output <results.json> [--batch-sizes "1,10,100,1000"]');
    process.exit(1);
}

// Parse batch sizes
const batchSizes = options['batch-sizes'].split(',').map(s => parseInt(s.trim()));

// Load test data
function loadTestData(filePath) {
    const buffer = fs.readFileSync(filePath);
    let offset = 0;
    
    // Read header
    const magic = buffer.readUInt32LE(offset); offset += 4;
    const version = buffer.readUInt32LE(offset); offset += 4;
    const nSamples = buffer.readUInt32LE(offset); offset += 4;
    const nFeatures = buffer.readUInt32LE(offset); offset += 4;
    
    if (magic !== 0xCAFEBABE) {
        throw new Error('Invalid test data file format');
    }
    
    console.log(`   Loaded test data: ${nSamples} samples, ${nFeatures} features`);
    
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

// Run predictions in worker thread
async function runPredictionsInWorker(wasmPath, testData, batchSize) {
    return new Promise((resolve, reject) => {
        const workerPath = path.join(__dirname, 'worker_template.js');
        const worker = new Worker(workerPath, {
            workerData: {
                wasmPath,
                testData,
                batchSize
            }
        });
        
        let results = null;
        
        worker.on('message', (msg) => {
            if (msg.type === 'progress') {
                const percent = (msg.processed / msg.total * 100).toFixed(1);
                process.stdout.write(`\r      ⠦ Progress: ${msg.processed.toLocaleString()}/${msg.total.toLocaleString()} (${percent}%)`);
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
        console.log('📊 Running predictions...');
        
        // Load test data
        const testData = loadTestData(options['test-data']);
        
        // Results container
        const allResults = {
            experiment: path.basename(options.wasm, '.js'),
            timestamp: new Date().toISOString(),
            totalSamples: testData.nSamples,
            batchResults: []
        };
        
        // Test each batch size
        for (const batchSize of batchSizes) {
            console.log(`\n   Testing batch size: ${batchSize}`);
            
            const startTime = Date.now();
            const predictions = await runPredictionsInWorker(options.wasm, testData, batchSize);
            const endTime = Date.now();
            
            const totalTime = endTime - startTime;
            const predictionsPerSecond = Math.floor(testData.nSamples / (totalTime / 1000));
            
            console.log(`\n      ✓ Completed in ${totalTime}ms (${predictionsPerSecond.toLocaleString()} pred/s)`);
            
            allResults.batchResults.push({
                batchSize,
                totalTimeMs: totalTime,
                predictionsPerSecond,
                meanPredictionTimeUs: (totalTime * 1000) / testData.nSamples,
                predictions: predictions.slice(0, 1000) // Save first 1000 for verification
            });
        }
        
        // Calculate best performance
        allResults.bestPerformance = allResults.batchResults.reduce((best, current) => 
            current.predictionsPerSecond > best.predictionsPerSecond ? current : best
        );
        
        // Save results
        fs.writeFileSync(options.output, JSON.stringify(allResults, null, 2));
        console.log(`\n   ✓ Results saved to: ${options.output}`);
        
    } catch (error) {
        console.error('\n❌ Error:', error.message);
        process.exit(1);
    }
}

main();