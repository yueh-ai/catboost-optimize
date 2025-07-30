const { parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');

// Worker data contains: wasmPath, testData
const { wasmPath, testData } = workerData;

async function runPredictions() {
    try {
        // Load WASM module
        const wasmDir = path.dirname(wasmPath);
        const wasmFile = wasmPath.replace('.js', '.wasm');
        
        parentPort.postMessage({ type: 'status', message: 'Loading WASM module...' });
        
        // Import the module
        const CatBoostModule = require(wasmPath);
        
        // Initialize the module
        const Module = await CatBoostModule({
            wasmBinary: fs.readFileSync(wasmFile),
            locateFile: (filename) => {
                if (filename.endsWith('.wasm')) {
                    return wasmFile;
                }
                return path.join(wasmDir, filename);
            }
        });
        
        // Get the prediction function
        const catboostPredict = Module.cwrap('catboostPredict', 'number', ['number', 'number']);
        
        parentPort.postMessage({ type: 'status', message: 'Allocating memory for 1M samples...' });
        
        // Allocate memory for all features at once
        const featuresPerSample = testData.nFeatures;
        const totalSamples = testData.nSamples;
        const featureBytes = featuresPerSample * 4; // float32
        const totalBytes = totalSamples * featureBytes;
        
        // Allocate memory for one sample at a time (to avoid huge memory allocation)
        const samplePtr = Module._malloc(featureBytes);
        
        // Prepare predictions array
        const predictions = new Float32Array(totalSamples);
        
        parentPort.postMessage({ type: 'status', message: 'Running predictions...' });
        
        // Start timing
        const startTime = Date.now();
        
        // Process all samples
        const heapF32 = Module.HEAPF32;
        const ptrOffset = samplePtr >> 2;
        
        for (let i = 0; i < totalSamples; i++) {
            // Copy features to WASM memory
            const featureOffset = i * featuresPerSample;
            for (let j = 0; j < featuresPerSample; j++) {
                heapF32[ptrOffset + j] = testData.features[featureOffset + j];
            }
            
            // Run prediction
            predictions[i] = catboostPredict(samplePtr, featuresPerSample);
        }
        
        const endTime = Date.now();
        const executionTime = endTime - startTime;
        
        // Clean up
        Module._free(samplePtr);
        
        parentPort.postMessage({ type: 'status', message: `Completed ${totalSamples.toLocaleString()} predictions in ${executionTime}ms` });
        
        // Send results
        parentPort.postMessage({
            type: 'complete',
            results: {
                predictions: Array.from(predictions),
                executionTime: executionTime
            }
        });
        
    } catch (error) {
        console.error('Worker error:', error);
        throw error;
    }
}

// Run predictions
runPredictions().catch(error => {
    console.error('Fatal worker error:', error);
    process.exit(1);
});