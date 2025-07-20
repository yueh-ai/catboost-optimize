const { parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');

// Worker data contains: wasmPath, testData, batchSize
const { wasmPath, testData, batchSize } = workerData;

async function runPredictions() {
    try {
        // Load WASM module
        const wasmDir = path.dirname(wasmPath);
        const wasmFile = wasmPath.replace('.js', '.wasm');
        
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
            },
            onRuntimeInitialized: function() {
                console.log('WASM runtime initialized');
            }
        });
        
        // Ensure module is ready
        if (Module.calledRun) {
            console.log('Module is ready');
        } else {
            console.log('Waiting for module...');
            await new Promise(resolve => {
                Module.onRuntimeInitialized = resolve;
            });
        }
        
        // Get the prediction function
        const catboostPredict = Module.cwrap('catboostPredict', 'number', ['number', 'number']);
        
        // Allocate memory for features
        const featuresPerSample = testData.nFeatures;
        const featureBytes = featuresPerSample * 4; // float32
        const featuresPtr = Module._malloc(featureBytes * batchSize);
        
        // Wait a moment for heap to be fully initialized
        await new Promise(resolve => setTimeout(resolve, 100));
        
        console.log('Worker initialized:', {
            featuresPerSample,
            featureBytes,
            batchSize,
            featuresPtr,
            heapSize: Module.HEAPF32 ? Module.HEAPF32.length : 'undefined',
            moduleKeys: Object.keys(Module).filter(k => k.includes('HEAP')).join(', ')
        });
        
        // Prepare predictions array
        const predictions = new Float32Array(testData.nSamples);
        
        // Process in batches
        const totalSamples = testData.nSamples;
        let processed = 0;
        
        while (processed < totalSamples) {
            const currentBatchSize = Math.min(batchSize, totalSamples - processed);
            
            // Copy features to WASM memory
            const heapF32 = Module.HEAPF32;
            const ptrOffset = featuresPtr >> 2;
            
            for (let i = 0; i < currentBatchSize; i++) {
                const sampleIdx = processed + i;
                const featureOffset = sampleIdx * featuresPerSample;
                const destOffset = ptrOffset + i * featuresPerSample;
                
                for (let j = 0; j < featuresPerSample; j++) {
                    heapF32[destOffset + j] = testData.features[featureOffset + j];
                }
            }
            
            // Run predictions
            for (let i = 0; i < currentBatchSize; i++) {
                const samplePtr = featuresPtr + i * featureBytes;
                predictions[processed + i] = catboostPredict(samplePtr, featuresPerSample);
            }
            
            processed += currentBatchSize;
            
            // Report progress
            if (processed % 10000 === 0 || processed === totalSamples) {
                parentPort.postMessage({
                    type: 'progress',
                    processed,
                    total: totalSamples
                });
            }
        }
        
        // Clean up
        Module._free(featuresPtr);
        
        // Send results
        parentPort.postMessage({
            type: 'complete',
            results: Array.from(predictions)
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